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
    interval_half_width: float | None = None,
) -> str:
    """Predicted-vs-actual scatter plot with a y=x reference line.

    If ``interval_half_width`` is given, a shaded band of that half-width
    around the reference line is drawn to show the residual-based prediction
    interval.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad

    if interval_half_width is not None:
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
