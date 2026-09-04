"""Out-of-distribution / extrapolation detection.

A trained surrogate model will happily produce a confident-looking number for
an input far outside anything it was trained on. This module flags that case
using four independent, individually weak signals rather than one strong one:

1. **Feature range check.** Flag a numeric feature value outside
   ``[min, max]`` observed in training data, or a categorical value never
   seen in training. Cheap and exact, but only catches inputs that are
   out-of-range on *at least one feature considered alone* — a point can be
   inside every feature's individual range and still sit in a region of
   feature-space combinations the training data never covered.
2. **Nearest-neighbour distance** (Euclidean, in standardized numeric feature
   space). Flags a point far from *every* training point, which catches
   some (not all) of the combination gaps the range check misses. The
   threshold is the ``nn_distance_percentile`` quantile (default 95th) of
   each training point's distance to its own nearest other training point
   (a leave-one-out computation): if a new point is farther from its nearest
   neighbour than most training points are from theirs, it sits in a sparser
   part of feature space than the data was validated on. This threshold is a
   heuristic tied to the training set's own density, not a statistically
   derived bound — it says "this looks unusually isolated relative to how
   spread out the training data already is," nothing stronger.
3. **Mahalanobis distance**, as an alternative to Euclidean nearest-neighbour
   distance, computed on standardized numeric features. Accounts for
   correlation and scale between features (two Euclidean-equidistant points
   are not equally unusual if the training data varies more along one axis
   than another); Euclidean does not. Requires at least 2 training rows to
   estimate a covariance matrix, and the estimate is unreliable with a
   near-singular covariance (e.g. two features that are almost perfectly
   correlated in training data) — a pseudo-inverse is used to avoid crashing
   in that case, but the resulting distance should be treated with caution
   when it happens.
4. **Ensemble disagreement**, when the trained model uses the bootstrap
   uncertainty method: high spread across ensemble members at a point is
   itself evidence the model is uncertain there, independent of the distance
   checks above (see ``core.uncertainty``).

None of these guarantee an input is safe just because it passes: a point can
sit inside the training data's range and still be a real extrapolation (e.g.
a combination of feature values never actually co-occurring in training,
that neither distance check happens to flag). Treat a "within domain" result
as "no red flag raised," not "verified reliable."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from surrogate_lab.core.schema import DatasetSchema

DistanceMetric = Literal["euclidean", "mahalanobis"]


class OODAssessment(BaseModel):
    """Structured out-of-distribution assessment for a single input row."""

    within_training_domain: bool
    warnings: list[str] = Field(default_factory=list)
    out_of_range_features: list[str] = Field(default_factory=list)
    unseen_categories: dict[str, str] = Field(default_factory=dict)
    nearest_neighbor_distance: float
    nearest_neighbor_distance_threshold: float | None
    distance_metric: DistanceMetric
    ensemble_disagreement: float | None = None
    ensemble_disagreement_threshold: float | None = None


@dataclass(frozen=True)
class OODDetector:
    """Reference statistics computed once from training data, used to assess new inputs."""

    numeric_features: list[str]
    categorical_features: list[str]
    feature_min: dict[str, float]
    feature_max: dict[str, float]
    categorical_seen: dict[str, set[str]]
    scaler_mean: NDArray[np.float64]
    scaler_scale: NDArray[np.float64]
    scaled_training_points: NDArray[np.float64]
    nn_distance_percentile: float
    distance_metric: DistanceMetric
    nn_distance_threshold: float | None
    covariance_inv: NDArray[np.float64] | None
    mahalanobis_threshold: float | None
    insufficient_data_warning: str | None = field(default=None)


def _standardize(
    values: NDArray[np.float64], mean: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray((values - mean) / scale, dtype=float)


def _pairwise_euclidean(points: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.asarray(np.sqrt(np.sum(diff**2, axis=-1)), dtype=float)


def _leave_one_out_min_distances(distance_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each point's distance to its nearest *other* point, from a full distance matrix."""
    masked = distance_matrix.copy()
    np.fill_diagonal(masked, np.inf)
    return np.asarray(np.min(masked, axis=1), dtype=float)


def _mahalanobis_distances_to_point(
    training_points: NDArray[np.float64],
    query: NDArray[np.float64],
    covariance_inv: NDArray[np.float64],
) -> NDArray[np.float64]:
    diff = training_points - query
    # Mahalanobis distance: sqrt(diff^T . Sigma^-1 . diff), computed per row.
    left = diff @ covariance_inv
    squared = np.einsum("ij,ij->i", left, diff)
    squared = np.clip(squared, 0.0, None)  # guard tiny negative values from pinv round-off
    return np.asarray(np.sqrt(squared), dtype=float)


def fit_ood_detector(
    x_train: pd.DataFrame,
    schema: DatasetSchema,
    *,
    nn_distance_percentile: float = 0.95,
    distance_metric: DistanceMetric = "euclidean",
) -> OODDetector:
    """Compute reference statistics from training data for later OOD assessment.

    Args:
        x_train: Training features (must contain every column in ``schema.features``).
        schema: Declares which features are numeric vs. categorical.
        nn_distance_percentile: Quantile (0-1) of training points' own
            leave-one-out nearest-neighbour distances used as the flagging
            threshold. Higher values flag fewer points as OOD.
        distance_metric: ``"euclidean"`` or ``"mahalanobis"``, see module docstring.

    Raises:
        ValueError: If ``nn_distance_percentile`` is not in (0, 1), or ``x_train`` is empty.
    """
    if not 0 < nn_distance_percentile < 1:
        raise ValueError("nn_distance_percentile must be strictly between 0 and 1")
    if len(x_train) == 0:
        raise ValueError("cannot fit an OOD detector on an empty training set")

    numeric_features = schema.numeric_features
    categorical_features = schema.categorical_features

    feature_min = {name: float(x_train[name].min()) for name in numeric_features}
    feature_max = {name: float(x_train[name].max()) for name in numeric_features}
    categorical_seen = {
        name: set(x_train[name].astype(str).unique()) for name in categorical_features
    }

    if numeric_features:
        numeric_values = x_train[numeric_features].to_numpy(dtype=float)
        mean = numeric_values.mean(axis=0)
        std = numeric_values.std(axis=0, ddof=0)
        std_safe = np.where(std == 0, 1.0, std)
        scaled = _standardize(numeric_values, mean, std_safe)
    else:
        mean = np.array([])
        std_safe = np.array([])
        scaled = np.zeros((len(x_train), 0))

    insufficient_data_warning: str | None = None
    nn_distance_threshold: float | None = None
    covariance_inv: NDArray[np.float64] | None = None
    mahalanobis_threshold: float | None = None

    n_train = len(x_train)
    if not numeric_features:
        insufficient_data_warning = (
            "no numeric features to compute a distance in: nearest_neighbor_distance will be "
            "NaN and no distance threshold is available."
        )
    elif n_train < 2:
        insufficient_data_warning = (
            f"only {n_train} training row(s): no leave-one-out neighbour distances could be "
            "computed, so no distance threshold is available. nearest_neighbor_distance is "
            "still reported (distance to that single point) but not compared against a "
            "threshold."
        )
    else:
        euclidean_matrix = _pairwise_euclidean(scaled)
        loo_euclidean = _leave_one_out_min_distances(euclidean_matrix)
        nn_distance_threshold = float(np.quantile(loo_euclidean, nn_distance_percentile))

        if distance_metric == "mahalanobis":
            if n_train > len(numeric_features):
                covariance = np.cov(scaled, rowvar=False, ddof=1)
                covariance = np.atleast_2d(covariance)
            else:
                covariance = np.atleast_2d(np.cov(scaled, rowvar=False, ddof=0))
            covariance_inv = np.linalg.pinv(covariance)
            loo_mahalanobis = np.array(
                [
                    _mahalanobis_distances_to_point(
                        np.delete(scaled, i, axis=0), scaled[i], covariance_inv
                    ).min()
                    for i in range(n_train)
                ]
            )
            mahalanobis_threshold = float(np.quantile(loo_mahalanobis, nn_distance_percentile))

    return OODDetector(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_min=feature_min,
        feature_max=feature_max,
        categorical_seen=categorical_seen,
        scaler_mean=mean,
        scaler_scale=std_safe,
        scaled_training_points=scaled,
        nn_distance_percentile=nn_distance_percentile,
        distance_metric=distance_metric,
        nn_distance_threshold=nn_distance_threshold,
        covariance_inv=covariance_inv,
        mahalanobis_threshold=mahalanobis_threshold,
        insufficient_data_warning=insufficient_data_warning,
    )


def assess_ood(
    detector: OODDetector,
    row: pd.Series | dict[str, Any],
    *,
    ensemble_disagreement: float | None = None,
    ensemble_disagreement_threshold: float | None = None,
) -> OODAssessment:
    """Assess a single input row against the training domain captured by ``detector``."""
    warnings: list[str] = []
    out_of_range_features: list[str] = []
    unseen_categories: dict[str, str] = {}

    for name in detector.numeric_features:
        value = float(row[name])
        lo, hi = detector.feature_min[name], detector.feature_max[name]
        if value < lo or value > hi:
            out_of_range_features.append(name)
            warnings.append(
                f"{name}={value:.4g} is outside the training range [{lo:.4g}, {hi:.4g}]"
            )

    for name in detector.categorical_features:
        str_value = str(row[name])
        seen = detector.categorical_seen[name]
        if str_value not in seen:
            unseen_categories[name] = str_value
            warnings.append(
                f"{name}={str_value!r} was not seen in training data "
                f"(seen values: {sorted(seen)})"
            )

    if detector.insufficient_data_warning is not None:
        warnings.append(detector.insufficient_data_warning)

    # The metric actually used can differ from detector.distance_metric: mahalanobis was
    # requested but falls back to euclidean when covariance_inv couldn't be estimated (always
    # paired with insufficient_data_warning above). Report the metric that was actually
    # computed, not the one configured, so the label always matches the number.
    configured_metric = detector.distance_metric
    if detector.numeric_features:
        numeric_values = np.array(
            [float(row[name]) for name in detector.numeric_features], dtype=float
        )
        scaled_query = _standardize(numeric_values, detector.scaler_mean, detector.scaler_scale)

        if configured_metric == "mahalanobis" and detector.covariance_inv is not None:
            distances = _mahalanobis_distances_to_point(
                detector.scaled_training_points, scaled_query, detector.covariance_inv
            )
            threshold = detector.mahalanobis_threshold
            actual_metric: DistanceMetric = "mahalanobis"
        else:
            distances = np.sqrt(
                np.sum((detector.scaled_training_points - scaled_query) ** 2, axis=1)
            )
            threshold = detector.nn_distance_threshold
            actual_metric = "euclidean"

        nearest_distance = float(distances.min()) if len(distances) else float("nan")
    else:
        nearest_distance = float("nan")
        threshold = None
        actual_metric = configured_metric

    if threshold is not None and nearest_distance > threshold:
        warnings.append(
            f"nearest-neighbor distance {nearest_distance:.4g} ({actual_metric}) exceeds the "
            f"training-set threshold {threshold:.4g} (the {detector.nn_distance_percentile:.0%} "
            "percentile of training points' own nearest-neighbour distances)"
        )

    if ensemble_disagreement is not None and ensemble_disagreement_threshold is not None:
        if ensemble_disagreement > ensemble_disagreement_threshold:
            warnings.append(
                f"bootstrap ensemble members disagree by {ensemble_disagreement:.4g}, more than "
                f"the typical {ensemble_disagreement_threshold:.4g} observed on validation data "
                "— treat this prediction's interval with extra caution"
            )

    distance_flagged = threshold is not None and nearest_distance > threshold
    disagreement_flagged = (
        ensemble_disagreement is not None
        and ensemble_disagreement_threshold is not None
        and ensemble_disagreement > ensemble_disagreement_threshold
    )
    within_training_domain = not (
        out_of_range_features or unseen_categories or distance_flagged or disagreement_flagged
    )

    return OODAssessment(
        within_training_domain=within_training_domain,
        warnings=warnings,
        out_of_range_features=out_of_range_features,
        unseen_categories=unseen_categories,
        nearest_neighbor_distance=nearest_distance,
        nearest_neighbor_distance_threshold=threshold,
        distance_metric=actual_metric,
        ensemble_disagreement=ensemble_disagreement,
        ensemble_disagreement_threshold=ensemble_disagreement_threshold,
    )
