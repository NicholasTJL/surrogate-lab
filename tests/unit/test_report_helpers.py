"""Unit tests for reporting/report.py's private HTML-rendering helpers.

These exercise branches (mahalanobis vs. euclidean vs. insufficient-data OOD
summaries, an empty error-by-region table) that a single end-to-end report
generation run doesn't reach, faster than a full training run per branch.
"""

import numpy as np

from surrogate_lab.core.error_analysis import RegionBin, RegionErrorTable
from surrogate_lab.core.ood import OODDetector
from surrogate_lab.reporting.report import _error_by_region_html, _ood_summary_html


def _detector(**overrides: object) -> OODDetector:
    defaults: dict[str, object] = dict(
        numeric_features=["x1"],
        categorical_features=[],
        feature_min={"x1": 0.0},
        feature_max={"x1": 10.0},
        categorical_seen={},
        scaler_mean=np.array([5.0]),
        scaler_scale=np.array([2.0]),
        scaled_training_points=np.zeros((3, 1)),
        nn_distance_percentile=0.95,
        distance_metric="euclidean",
        nn_distance_threshold=1.5,
        covariance_inv=None,
        mahalanobis_threshold=None,
        insufficient_data_warning=None,
    )
    defaults.update(overrides)
    return OODDetector(**defaults)  # type: ignore[arg-type]


def test_ood_summary_shows_euclidean_threshold_by_default() -> None:
    html_out = _ood_summary_html(_detector())

    assert "Euclidean nearest-neighbour distance threshold" in html_out
    assert "1.5" in html_out


def test_ood_summary_shows_mahalanobis_threshold_when_configured() -> None:
    html_out = _ood_summary_html(
        _detector(distance_metric="mahalanobis", mahalanobis_threshold=2.25)
    )

    assert "Mahalanobis nearest-neighbour distance threshold" in html_out
    assert "2.25" in html_out


def test_ood_summary_shows_insufficient_data_warning() -> None:
    html_out = _ood_summary_html(
        _detector(
            nn_distance_threshold=None,
            insufficient_data_warning="only 1 training row(s): no threshold available.",
        )
    )

    assert "only 1 training row" in html_out


def test_ood_summary_handles_no_threshold_available_at_all() -> None:
    # Distance metric says mahalanobis but neither threshold nor the insufficient-data
    # warning is set (defensive fallback branch, not reachable via fit_ood_detector itself).
    html_out = _ood_summary_html(
        _detector(
            distance_metric="mahalanobis",
            nn_distance_threshold=None,
            mahalanobis_threshold=None,
            insufficient_data_warning=None,
        )
    )

    assert "Out-of-distribution detection" in html_out


def test_error_by_region_html_empty_table_renders_nothing() -> None:
    empty_table = RegionErrorTable(region_name="target", bins=[])

    assert _error_by_region_html(empty_table, "linear_regression") == ""


def test_error_by_region_html_notes_spread_with_ratio_when_min_mae_positive() -> None:
    table = RegionErrorTable(
        region_name="target",
        bins=[
            RegionBin(label="[0, 1)", lower=0, upper=1, count=5, mae=1.0, rmse=1.0),
            RegionBin(label="[1, 2)", lower=1, upper=2, count=5, mae=4.0, rmse=4.0),
        ],
    )

    html_out = _error_by_region_html(table, "linear_regression")

    assert "4x the lowest-error bin" in html_out


def test_error_by_region_html_notes_spread_when_min_mae_is_zero() -> None:
    # min_mae == 0 (perfect predictions in one bin) means a ratio is undefined; the note must
    # still say something rather than silently disappear.
    table = RegionErrorTable(
        region_name="target",
        bins=[
            RegionBin(label="[0, 1)", lower=0, upper=1, count=5, mae=0.0, rmse=0.0),
            RegionBin(label="[1, 2)", lower=1, upper=2, count=5, mae=4.0, rmse=4.0),
        ],
    )

    html_out = _error_by_region_html(table, "linear_regression")

    assert "~zero error" in html_out
    assert "4" in html_out


def test_error_by_region_html_omits_note_when_all_bins_zero_error() -> None:
    table = RegionErrorTable(
        region_name="target",
        bins=[RegionBin(label="[0, 1)", lower=0, upper=1, count=5, mae=0.0, rmse=0.0)],
    )

    html_out = _error_by_region_html(table, "linear_regression")

    assert "~zero error" not in html_out
    assert "lowest-error bin" not in html_out
