import numpy as np
import pytest

from surrogate_lab.core.uncertainty import empirical_coverage, fit_prediction_interval


def test_half_width_grows_with_confidence_level() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=500)
    y_pred = y_true + rng.normal(scale=0.1, size=500)

    low_confidence = fit_prediction_interval(y_true, y_pred, confidence_level=0.5)
    high_confidence = fit_prediction_interval(y_true, y_pred, confidence_level=0.95)

    assert high_confidence.half_width > low_confidence.half_width


def test_perfect_predictions_have_zero_half_width() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    interval = fit_prediction_interval(y_true, y_pred, confidence_level=0.9)

    assert interval.half_width == pytest.approx(0.0)


def test_invalid_confidence_level_raises() -> None:
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        fit_prediction_interval(y_true, y_pred, confidence_level=1.5)


def test_empty_arrays_raise() -> None:
    with pytest.raises(ValueError, match="empty array"):
        fit_prediction_interval(np.array([]), np.array([]))


def test_empirical_coverage_matches_expected_fraction() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    y_pred = rng.normal(size=n)
    y_true = y_pred + rng.normal(scale=1.0, size=n)

    interval = fit_prediction_interval(y_true, y_pred, confidence_level=0.9)
    coverage = empirical_coverage(interval, y_true, y_pred)

    # Fit and evaluated on the same distribution: coverage should track the
    # nominal confidence level reasonably closely.
    assert coverage == pytest.approx(0.9, abs=0.03)


def test_coverage_is_one_when_half_width_covers_everything() -> None:
    y_true = np.array([1.0, 2.0, 100.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    interval = fit_prediction_interval(y_true, y_pred, confidence_level=0.9999)
    huge_interval = type(interval)(half_width=1000.0, confidence_level=0.9)

    coverage = empirical_coverage(huge_interval, y_true, y_pred)

    assert coverage == pytest.approx(1.0)
