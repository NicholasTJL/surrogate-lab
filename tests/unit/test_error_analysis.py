import numpy as np
import pytest

from surrogate_lab.core.error_analysis import error_by_region


def test_error_uniform_across_bins_when_error_is_constant() -> None:
    values = np.arange(100, dtype=float)
    y_true = values.copy()
    y_pred = values + 1.0  # constant absolute error of 1 everywhere

    table = error_by_region(values, y_true, y_pred, "target", n_bins=4)

    assert len(table.bins) == 4
    for b in table.bins:
        assert b.mae == pytest.approx(1.0)
    assert table.max_mae == pytest.approx(table.min_mae)


def test_error_concentrated_in_top_bin_is_detected() -> None:
    # Model is perfect except badly wrong in the top 10% of the target range.
    values = np.linspace(0, 100, 200)
    y_true = values.copy()
    y_pred = values.copy()
    top_mask = values > 90
    y_pred[top_mask] = y_true[top_mask] + 50.0  # large error only at the top

    table = error_by_region(values, y_true, y_pred, "target", n_bins=10)

    mae_by_bin = [b.mae for b in table.bins]
    assert mae_by_bin[-1] > 10 * mae_by_bin[0]
    assert table.max_mae == pytest.approx(mae_by_bin[-1])


def test_bin_counts_sum_to_total_points() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_true = values.copy()
    y_pred = values + 0.5

    table = error_by_region(values, y_true, y_pred, "target", n_bins=3)

    assert sum(b.count for b in table.bins) == len(values)


def test_hand_computed_two_bins() -> None:
    # 4 points, 2 bins, split at the median (edges: [1, 2, 4] after quantile(0,0.5,1)).
    values = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, 22.0, 33.0, 44.0])  # errors: 0, 2, 3, 4

    table = error_by_region(values, y_true, y_pred, "target", n_bins=2)

    assert len(table.bins) == 2
    # bin 0: values <= median (2.0) -> points 1,2 -> errors 0,2 -> MAE 1.0
    # bin 1: values > median -> points 3,4 -> errors 3,4 -> MAE 3.5
    assert table.bins[0].mae == pytest.approx(1.0)
    assert table.bins[1].mae == pytest.approx(3.5)
    assert table.bins[0].count == 2
    assert table.bins[1].count == 2


def test_fewer_distinct_values_than_requested_bins_collapses_gracefully() -> None:
    # Only 2 distinct values but 5 bins requested: duplicate quantile edges collapse.
    values = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    y_true = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    y_pred = np.array([1.0, 1.1, 0.9, 2.0, 2.2])

    table = error_by_region(values, y_true, y_pred, "target", n_bins=5)

    assert len(table.bins) <= 2
    assert sum(b.count for b in table.bins) == len(values)


def test_all_identical_values_produce_single_bin() -> None:
    values = np.array([5.0, 5.0, 5.0])
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.5])

    table = error_by_region(values, y_true, y_pred, "target", n_bins=4)

    assert len(table.bins) == 1
    assert table.bins[0].count == 3


def test_region_name_is_preserved() -> None:
    values = np.array([1.0, 2.0])
    table = error_by_region(values, values, values, "length_m", n_bins=1)

    assert table.region_name == "length_m"


def test_empty_arrays_raise() -> None:
    with pytest.raises(ValueError, match="empty array"):
        error_by_region(np.array([]), np.array([]), np.array([]), "target")


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        error_by_region(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]), "target")


def test_invalid_n_bins_raises() -> None:
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        error_by_region(np.array([1.0]), np.array([1.0]), np.array([1.0]), "target", n_bins=0)


def test_gap_in_data_produces_empty_bins_that_are_dropped() -> None:
    # A big gap between two clusters: with many requested bins, quantile interpolation
    # can place two edges entirely inside the gap, producing an empty bin that must be
    # dropped from the output rather than reported with 0 rows.
    values = np.array([1.0, 2.0, 3.0, 96.0, 97.0, 98.0, 99.0, 100.0])
    y_true = values.copy()
    y_pred = values + 1.0

    table = error_by_region(values, y_true, y_pred, "target", n_bins=10)

    assert all(b.count > 0 for b in table.bins)
    assert sum(b.count for b in table.bins) == len(values)
    assert len(table.bins) < 10  # fewer bins than requested because some were empty


def test_single_point_produces_one_bin() -> None:
    table = error_by_region(np.array([1.0]), np.array([2.0]), np.array([3.0]), "target", n_bins=5)

    assert len(table.bins) == 1
    assert table.bins[0].mae == pytest.approx(1.0)
