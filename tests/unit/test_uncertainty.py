import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from surrogate_lab.core.uncertainty import (
    BootstrapPredictionInterval,
    empirical_coverage,
    ensemble_disagreement,
    ensemble_predict,
    fit_bootstrap_ensemble,
    fit_prediction_interval,
)


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


def _linear_pipeline() -> Pipeline:
    return Pipeline(steps=[("model", LinearRegression())])


def test_bootstrap_interval_bounds_hand_computed() -> None:
    # 5 ensemble members' predictions at 1 point: [1, 2, 3, 4, 5].
    # confidence_level=0.8 -> alpha=0.2 -> quantiles at 0.1 and 0.9.
    # np.quantile linear interpolation: q(0.1) = 1 + 0.4*(2-1) = 1.4,
    # q(0.9) = 4 + 0.6*(5-4) = 4.6.
    ensemble_predictions = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    interval = BootstrapPredictionInterval(confidence_level=0.8, n_estimators=5)

    lower, upper = interval.bounds(ensemble_predictions)

    assert lower[0] == pytest.approx(1.4)
    assert upper[0] == pytest.approx(4.6)
    assert interval.half_widths(ensemble_predictions)[0] == pytest.approx(1.6)


def test_fit_bootstrap_ensemble_rejects_too_few_estimators() -> None:
    x_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="at least 2"):
        fit_bootstrap_ensemble(_linear_pipeline, x_train, y_train, n_estimators=1)


def test_fit_bootstrap_ensemble_rejects_empty_training_set() -> None:
    x_train = pd.DataFrame({"x": []})
    y_train = np.array([])

    with pytest.raises(ValueError, match="empty training set"):
        fit_bootstrap_ensemble(_linear_pipeline, x_train, y_train, n_estimators=5)


def test_ensemble_predict_shape() -> None:
    x_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_train = np.array([2.0, 4.0, 6.0, 8.0])
    pipelines = fit_bootstrap_ensemble(_linear_pipeline, x_train, y_train, n_estimators=6)

    x_query = pd.DataFrame({"x": [5.0, 6.0]})
    predictions = ensemble_predict(pipelines, x_query)

    assert predictions.shape == (6, 2)


def test_ensemble_disagreement_hand_computed() -> None:
    # Column 0: [1, 3, 5] -> mean 3, population std sqrt(((2)^2+0+(2)^2)/3) = sqrt(8/3).
    ensemble_predictions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    disagreement = ensemble_disagreement(ensemble_predictions)

    assert disagreement == pytest.approx([np.sqrt(8 / 3), np.sqrt(8 / 3)])


def test_bootstrap_ensemble_perfect_agreement_on_noiseless_linear_data() -> None:
    # A perfectly linear relationship with no noise: every bootstrap resample still
    # contains only points on the same line, so ordinary least squares recovers that
    # exact line every time and all ensemble members agree exactly.
    x_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y_train = np.array([2.0 * v for v in x_train["x"]])

    pipelines = fit_bootstrap_ensemble(
        _linear_pipeline, x_train, y_train, n_estimators=20, random_state=0
    )
    predictions = ensemble_predict(pipelines, pd.DataFrame({"x": [4.5]}))
    disagreement = ensemble_disagreement(predictions)

    assert disagreement[0] == pytest.approx(0.0, abs=1e-8)


def test_bootstrap_ensemble_strong_disagreement_on_noisy_small_data() -> None:
    # A tiny, noisy dataset: bootstrap resamples of just a few points can differ a lot
    # (some resamples omit a point entirely, others duplicate it), so fitted lines vary
    # noticeably, especially when queried outside the training x-range.
    x_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 5.0, 2.0])  # deliberately non-linear/noisy

    pipelines = fit_bootstrap_ensemble(
        _linear_pipeline, x_train, y_train, n_estimators=30, random_state=1
    )
    # Query well outside the training range, where small slope differences are amplified.
    predictions = ensemble_predict(pipelines, pd.DataFrame({"x": [20.0]}))
    disagreement = ensemble_disagreement(predictions)

    noiseless_x = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    noiseless_pipelines = fit_bootstrap_ensemble(
        _linear_pipeline,
        noiseless_x,
        np.array([2.0 * v for v in noiseless_x["x"]]),
        n_estimators=30,
        random_state=1,
    )
    noiseless_disagreement = ensemble_disagreement(
        ensemble_predict(noiseless_pipelines, pd.DataFrame({"x": [20.0]}))
    )

    assert disagreement[0] > 1.0
    assert disagreement[0] > noiseless_disagreement[0]
