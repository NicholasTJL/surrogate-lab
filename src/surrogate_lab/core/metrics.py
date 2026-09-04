"""Regression metrics: MAE, RMSE, R^2, and MAPE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Metrics:
    """A bundle of regression metrics computed on one set of predictions."""

    mae: float
    rmse: float
    r2: float
    mape: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2, "mape": self.mape}


def mean_absolute_percentage_error(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """MAPE as a percentage. Rows where ``y_true`` is zero are excluded to
    avoid division by zero; raises if every row would be excluded.
    """
    nonzero = y_true != 0
    if not np.any(nonzero):
        raise ValueError("cannot compute MAPE: every y_true value is zero")
    errors = np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
    return float(np.mean(errors) * 100.0)


def compute_metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> Metrics:
    """Compute MAE, RMSE, R^2, and MAPE for a set of predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return Metrics(mae=mae, rmse=rmse, r2=r2, mape=mape)
