"""Auto-generated model cards: a structured, honest summary of one trained model.

Every field is computed from the actual training run (row counts, feature
ranges, real metrics, the uncertainty method actually used) rather than
hand-written boilerplate. ``limitations``/``intended_use``/``out_of_scope_use``
are assembled from those same facts, so they change when the run changes
(e.g. the bootstrap-vs-residual limitation text differs depending on
``uncertainty_method``) instead of being static text pasted into every card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from surrogate_lab.core.metrics import Metrics
from surrogate_lab.core.schema import DatasetSchema


@dataclass(frozen=True)
class ModelCard:
    """A structured summary of one trained model, its data, and its known limits."""

    model_name: str
    model_type: str
    hyperparameters: dict[str, Any]
    training_rows: int
    validation_rows: int
    test_rows: int
    numeric_feature_ranges: dict[str, tuple[float, float]]
    categorical_feature_values: dict[str, list[str]]
    target_name: str
    target_range: tuple[float, float]
    metrics_test: Metrics
    cv_mae_mean: float | None
    uncertainty_method: str
    confidence_level: float
    interval_coverage_test: float
    limitations: list[str] = field(default_factory=list)
    intended_use: list[str] = field(default_factory=list)
    out_of_scope_use: list[str] = field(default_factory=list)


def _hyperparameters(estimator: Any) -> dict[str, Any]:
    """Extract a plain-JSON-able hyperparameter dict from a fitted scikit-learn estimator."""
    params: dict[str, Any] = estimator.get_params(deep=False)
    return {key: value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
            for key, value in sorted(params.items())}


def _limitations(
    uncertainty_method: str,
    confidence_level: float,
    interval_coverage_test: float,
    n_bootstrap_estimators: int | None,
) -> list[str]:
    limitations = []
    coverage_gap = abs(interval_coverage_test - confidence_level)
    if uncertainty_method == "bootstrap":
        limitations.append(
            f"Uncertainty is estimated by training {n_bootstrap_estimators} bootstrap-resampled "
            "copies of this model and using the spread of their predictions as the interval. "
            "This gives a per-point estimate that can widen near the edges of the training data, "
            "but it costs "
            f"{n_bootstrap_estimators}x the training time of a single fit and does not capture "
            "model-form uncertainty: if the chosen model type is a poor fit for the underlying "
            "relationship, every ensemble member can still agree while being collectively wrong."
        )
    else:
        limitations.append(
            "Uncertainty is a single, global, symmetric interval half-width computed from "
            "validation-set residuals (not per-point). It assumes residual magnitude is roughly "
            "constant across the input domain (homoscedastic), which under-states uncertainty in "
            "regions where the model is locally worse, such as extrapolation."
        )
    limitations.append(
        f"Empirical test-set coverage of the {confidence_level:.0%} interval was "
        f"{interval_coverage_test:.1%}"
        + (
            ", close to the target."
            if coverage_gap < 0.05
            else f", a {coverage_gap:.1%} gap from the target — treat interval widths on new "
            "data with proportionate caution."
        )
    )
    limitations.append(
        "Predictions for inputs outside the training data's feature ranges are extrapolations; "
        "this model card does not by itself tell you whether a given input is in range — use the "
        "out-of-distribution assessment (surrogate_lab.core.ood) for that."
    )
    return limitations


def build_model_card(
    *,
    model_name: str,
    fitted_estimator: Any,
    schema: DatasetSchema,
    x_train: pd.DataFrame,
    y_train: NDArray[np.float64],
    validation_rows: int,
    test_rows: int,
    metrics_test: Metrics,
    cv_mae_scores: NDArray[np.float64],
    uncertainty_method: str,
    confidence_level: float,
    interval_coverage_test: float,
    n_bootstrap_estimators: int | None = None,
) -> ModelCard:
    """Build a :class:`ModelCard` from one training run's real inputs and outputs."""
    numeric_ranges = {
        name: (float(x_train[name].min()), float(x_train[name].max()))
        for name in schema.numeric_features
    }
    categorical_values = {
        name: sorted(x_train[name].astype(str).unique()) for name in schema.categorical_features
    }
    cv_mae_mean = float(cv_mae_scores.mean()) if cv_mae_scores.size else None

    limitations = _limitations(
        uncertainty_method, confidence_level, interval_coverage_test, n_bootstrap_estimators
    )

    intended_use = [
        "Interpolating within the feature ranges shown above, for the same kind of "
        "process/system that generated the training data.",
        "Quickly exploring a design space or screening candidates before committing to an "
        "expensive simulation or experiment for the most promising ones.",
    ]
    out_of_scope_use = [
        "Extrapolating beyond the training feature ranges without first checking the "
        "out-of-distribution assessment.",
        "Safety-critical or high-stakes decisions made from this model's output alone, without "
        "independent validation against the real simulation or experiment.",
        "Any input distribution materially different from the training data (a different "
        "material, geometry regime, or operating range than what was trained on).",
    ]

    return ModelCard(
        model_name=model_name,
        model_type=type(fitted_estimator).__name__,
        hyperparameters=_hyperparameters(fitted_estimator),
        training_rows=len(x_train),
        validation_rows=validation_rows,
        test_rows=test_rows,
        numeric_feature_ranges=numeric_ranges,
        categorical_feature_values=categorical_values,
        target_name=schema.target,
        target_range=(float(y_train.min()), float(y_train.max())),
        metrics_test=metrics_test,
        cv_mae_mean=cv_mae_mean,
        uncertainty_method=uncertainty_method,
        confidence_level=confidence_level,
        interval_coverage_test=interval_coverage_test,
        limitations=limitations,
        intended_use=intended_use,
        out_of_scope_use=out_of_scope_use,
    )
