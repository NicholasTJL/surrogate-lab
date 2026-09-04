"""Residual-based prediction intervals.

This is deliberately the simplest honest approach to uncertainty for v0.1.0,
not a substitute for real predictive uncertainty quantification. See the
"Uncertainty" section of the README for the method and its limitations.

Method: fit the model, compute residuals on a held-out validation set, and
take the empirical quantile of the *absolute* residuals at the requested
confidence level as a single, symmetric, global half-width. A prediction
interval is then ``[prediction - half_width, prediction + half_width]`` for
every point, regardless of where that point sits in feature space.

This assumes residual magnitude is roughly constant across the input domain
(homoscedasticity) and that the validation set is representative of future
inputs. It will under-cover in regions where the model is locally worse
(e.g. extrapolation) and over-cover where it is locally better. Bootstrap
ensembles or Gaussian-process variance (planned for v0.2.0, see
docs/vision.md) would give per-point, heteroscedastic estimates instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictionInterval:
    """A single global half-width for a symmetric prediction interval."""

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
