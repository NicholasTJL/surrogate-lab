"""Error analysis by data region: is model error uniform, or concentrated somewhere?

A single test-set MAE/RMSE can hide a model that is excellent on most of the
input space and much worse in one region (commonly the tails of the target
distribution, or the edges of a feature's range). This module bins a chosen
variable (a feature, or the target itself) into quantile-based regions and
reports MAE/RMSE per region, so that kind of concentration is visible instead
of averaged away.

This is a descriptive diagnostic, not a statistical test: with few test
points per bin, per-bin MAE/RMSE estimates are noisy, and no significance
test is applied to whether one bin's error is "really" higher than another's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class RegionBin:
    """Error metrics for one quantile bin of the binning variable."""

    label: str
    lower: float
    upper: float
    count: int
    mae: float
    rmse: float


@dataclass(frozen=True)
class RegionErrorTable:
    """Error broken down by quantile bin of ``region_name``."""

    region_name: str
    bins: list[RegionBin]

    @property
    def max_mae(self) -> float:
        return max((b.mae for b in self.bins), default=float("nan"))

    @property
    def min_mae(self) -> float:
        return min((b.mae for b in self.bins), default=float("nan"))


def error_by_region(
    values: NDArray[np.float64],
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    region_name: str,
    n_bins: int = 5,
) -> RegionErrorTable:
    """Break MAE/RMSE down by quantile bin of ``values``.

    Args:
        values: The variable to bin by (a feature column, or the target
            itself), same length as ``y_true``/``y_pred``.
        y_true: Observed target values.
        y_pred: Model predictions, same length as ``y_true``.
        region_name: Label for the binning variable, used in the returned table.
        n_bins: Requested number of quantile bins. Fewer bins are produced
            when ``values`` has too few distinct values to support this many
            (duplicate quantile edges are collapsed rather than raising).

    Raises:
        ValueError: If the arrays are empty, of mismatched length, or ``n_bins`` is less than 1.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if len(values) == 0:
        raise ValueError("cannot compute error-by-region on an empty array")
    if not (len(values) == len(y_true) == len(y_pred)):
        raise ValueError("values, y_true, and y_pred must have the same length")

    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)  # collapse duplicate edges from ties / few distinct values

    bin_index: NDArray[np.intp]
    if len(edges) < 2:
        # Every value is identical: a single bin covering that one value.
        bin_index = np.zeros(len(values), dtype=np.intp)
        edges = np.array([edges[0], edges[0]])
    else:
        # Bin by the internal edges only, right-open: values equal to the overall max
        # (which equals edges[-1]) land in the last bin rather than overflowing past it.
        bin_index = np.digitize(values, edges[1:-1], right=False)

    bins: list[RegionBin] = []
    n_actual_bins = len(edges) - 1
    for i in range(n_actual_bins):
        mask = bin_index == i
        count = int(mask.sum())
        lower, upper = float(edges[i]), float(edges[i + 1])
        if count == 0:
            continue
        bin_true = y_true[mask]
        bin_pred = y_pred[mask]
        mae = float(mean_absolute_error(bin_true, bin_pred))
        rmse = float(np.sqrt(mean_squared_error(bin_true, bin_pred)))
        bins.append(
            RegionBin(
                label=f"[{lower:.4g}, {upper:.4g}]",
                lower=lower,
                upper=upper,
                count=count,
                mae=mae,
                rmse=rmse,
            )
        )

    return RegionErrorTable(region_name=region_name, bins=bins)
