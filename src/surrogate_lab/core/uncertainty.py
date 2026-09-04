"""Prediction-interval methods: residual-based (default) and bootstrap-ensemble.

Two independent, honestly-scoped methods live here, selected via
``ExperimentConfig.uncertainty.method``:

**Residual-based** (v0.1.0, default) is the simplest honest approach: fit the model
once, compute residuals on a held-out validation set, and take the empirical
quantile of the *absolute* residuals as a single, symmetric, global half-width.
It assumes residual magnitude is roughly constant across the input domain
(homoscedasticity) and gives no per-point estimate.

**Bootstrap-ensemble** (v0.2.0) trains N resamples of the same model on
bootstrap-resampled training data (bagging) and uses the spread of the N
predictions at each point as that point's interval. This is a genuine
alternative, not a replacement: it gives a per-point, heteroscedastic estimate
(wider where the ensemble disagrees, narrower where it agrees), which the
residual method cannot. It costs N times the training time of a single fit,
and it is still only as good as the base model and the resampling procedure —
it does not capture uncertainty about whether the *model class itself* is
wrong for the data (model-form uncertainty), and with a small or unrepresentative
training set, all N resamples can still agree while being collectively wrong.
Neither method detects out-of-distribution inputs; see ``core.ood`` for that,
including how ensemble disagreement doubles as one of its signals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class PredictionInterval:
    """A single global half-width for a symmetric, residual-based prediction interval."""

    half_width: float
    confidence_level: float

    def bounds(
        self, predictions: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(lower, upper)`` bound arrays for ``predictions``."""
        lower = predictions - self.half_width
        upper = predictions + self.half_width
        return lower, upper


def fit_prediction_interval(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    confidence_level: float = 0.9,
) -> PredictionInterval:
    """Compute a residual-based prediction interval half-width.

    Args:
        y_true: Observed values on a held-out (validation) set.
        y_pred: Model predictions on the same set.
        confidence_level: Target coverage, in (0, 1). For example, 0.9 asks
            for an interval that should contain roughly 90% of future
            residuals, *if* the homoscedasticity assumption above holds.

    Raises:
        ValueError: If ``confidence_level`` is not strictly between 0 and 1,
            or if ``y_true``/``y_pred`` are empty.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if len(y_true) == 0:
        raise ValueError("cannot fit a prediction interval on an empty array")

    residuals = np.abs(y_true - y_pred)
    half_width = float(np.quantile(residuals, confidence_level))
    return PredictionInterval(half_width=half_width, confidence_level=confidence_level)


def empirical_coverage(
    interval: PredictionInterval,
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Fraction of ``y_true`` values that fall inside the interval around
    ``y_pred``. Compare against ``interval.confidence_level`` to judge
    calibration on a held-out set the interval was *not* fit on (e.g. test).
    """
    lower, upper = interval.bounds(y_pred)
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


@dataclass(frozen=True)
class BootstrapPredictionInterval:
    """A per-point prediction interval derived from bootstrap-ensemble spread."""

    confidence_level: float
    n_estimators: int

    def bounds(
        self, ensemble_predictions: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(lower, upper)`` per point from ``ensemble_predictions``.

        Args:
            ensemble_predictions: Array of shape ``(n_estimators, n_points)``,
                one row per ensemble member's predictions on the same inputs.
        """
        alpha = 1 - self.confidence_level
        lower = np.quantile(ensemble_predictions, alpha / 2, axis=0)
        upper = np.quantile(ensemble_predictions, 1 - alpha / 2, axis=0)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def half_widths(self, ensemble_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-point half-width, i.e. half the ``(lower, upper)`` gap at each point."""
        lower, upper = self.bounds(ensemble_predictions)
        return (upper - lower) / 2.0


def fit_bootstrap_ensemble(
    build_pipeline: Callable[[], Pipeline],
    x_train: pd.DataFrame,
    y_train: NDArray[np.float64],
    n_estimators: int = 30,
    random_state: int = 42,
) -> list[Pipeline]:
    """Train ``n_estimators`` copies of a pipeline, each on a bootstrap resample of the data.

    Each resample draws ``len(x_train)`` rows from ``x_train``/``y_train`` with
    replacement (the standard bagging procedure), then fits a freshly
    constructed pipeline (from ``build_pipeline``) on that resample. The
    pipeline's own randomness (e.g. a random forest's tree construction) is
    left at whatever ``build_pipeline`` sets it to; the resampling itself is
    the main source of ensemble diversity, which is what makes this a bagging
    procedure and not just N identical fits.

    Args:
        build_pipeline: Returns a fresh, unfitted pipeline each call.
        x_train: Training features.
        y_train: Training targets, same length as ``x_train``.
        n_estimators: Number of bootstrap resamples to train. More members
            give a smoother, more stable interval estimate at N times the
            training cost; 30 is a reasonable default for tabular data with
            fast-to-fit models, not a value derived from theory.
        random_state: Seed for the resampling draws. Does not seed each
            member's own model randomness (that comes from ``build_pipeline``).

    Raises:
        ValueError: If ``n_estimators`` is less than 2 or ``x_train`` is empty.
    """
    if n_estimators < 2:
        raise ValueError("n_estimators must be at least 2 to form an ensemble")
    if len(x_train) == 0:
        raise ValueError("cannot fit a bootstrap ensemble on an empty training set")

    rng = np.random.default_rng(random_state)
    n = len(x_train)
    pipelines = []
    for _ in range(n_estimators):
        resample_idx = rng.integers(0, n, size=n)
        pipeline = build_pipeline()
        pipeline.fit(x_train.iloc[resample_idx], y_train[resample_idx])
        pipelines.append(pipeline)
    return pipelines


def ensemble_predict(pipelines: list[Pipeline], x: pd.DataFrame) -> NDArray[np.float64]:
    """Predict with every ensemble member. Returns shape ``(len(pipelines), len(x))``."""
    return np.array([np.asarray(p.predict(x), dtype=float) for p in pipelines])


def ensemble_disagreement(ensemble_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-point standard deviation across ensemble members.

    A simple, direct measure of how much the ensemble disagrees at each
    point; high disagreement is itself a signal that the input may be poorly
    covered by the training data (see ``core.ood``), independent of the
    interval width computed from the same predictions.
    """
    return np.asarray(np.std(ensemble_predictions, axis=0), dtype=float)
