"""Matplotlib plots rendered to base64-encoded PNGs for embedding in HTML."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402


def _figure_to_base64_png(fig: Figure) -> str:
    """Render a matplotlib figure to a base64-encoded PNG data string and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parity_plot(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    title: str,
    interval_half_width: float | NDArray[np.float64] | None = None,
) -> str:
    """Predicted-vs-actual scatter plot with a y=x reference line.

    ``interval_half_width`` controls how the prediction interval is shown:

    * A single float (the residual-based method's constant half-width) draws
      a shaded band of that half-width around the y=x reference line.
    * A per-point array (the bootstrap method's half-width at each test
      point) draws vertical error bars on each point instead, since a
      heteroscedastic interval has no single band to shade.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad

    per_point_half_width: NDArray[np.float64] | None = None
    if isinstance(interval_half_width, np.ndarray):
        per_point_half_width = interval_half_width
    elif interval_half_width is not None:
        line = np.linspace(lo, hi, 100)
        ax.fill_between(
            line,
            line - interval_half_width,
            line + interval_half_width,
            color="tab:blue",
            alpha=0.15,
            label=f"±{interval_half_width:.3g} prediction interval",
        )

    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="y = x")
    if per_point_half_width is not None:
        ax.errorbar(
            y_true,
            y_pred,
            yerr=per_point_half_width,
            fmt="o",
            alpha=0.6,
            markersize=4,
            elinewidth=1,
            ecolor="tab:blue",
            color="tab:blue",
            capsize=2,
            label="prediction ± bootstrap interval",
        )
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none", color="tab:blue", s=20)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _figure_to_base64_png(fig)


def residual_plot(y_true: NDArray[np.float64], y_pred: NDArray[np.float64], title: str) -> str:
    """Residuals (actual - predicted) against predicted values."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolor="none", color="tab:orange", s=20)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    fig.tight_layout()
    return _figure_to_base64_png(fig)


def residual_histogram(residuals: NDArray[np.float64], title: str) -> str:
    """Histogram of residuals."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=30, color="tab:green", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return _figure_to_base64_png(fig)


def error_by_region_chart(
    bin_labels: list[str],
    mae_per_bin: NDArray[np.float64],
    counts_per_bin: NDArray[np.int_],
    region_name: str,
    title: str,
) -> str:
    """Bar chart of MAE per quantile bin, with point counts annotated per bar."""
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(bin_labels))
    ax.bar(positions, mae_per_bin, color="tab:purple", alpha=0.8)
    for pos, mae, count in zip(positions, mae_per_bin, counts_per_bin, strict=True):
        ax.text(float(pos), float(mae), f"n={count}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(f"{region_name} (quantile bin)")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    fig.tight_layout()
    return _figure_to_base64_png(fig)
