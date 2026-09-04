import numpy as np
import pytest

from surrogate_lab.core.metrics import compute_metrics, mean_absolute_percentage_error


def test_perfect_predictions_have_zero_error_and_r2_one() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.mape == pytest.approx(0.0)


def test_known_mae_and_rmse() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 3.0, 8.0])

    metrics = compute_metrics(y_true, y_pred)

    # errors: 1, 0, 0, 4 -> MAE = 5/4 = 1.25, RMSE = sqrt((1+0+0+16)/4) = sqrt(4.25)
    assert metrics.mae == pytest.approx(1.25)
    assert metrics.rmse == pytest.approx(np.sqrt(4.25))


def test_mape_known_value() -> None:
    y_true = np.array([10.0, 20.0, 50.0])
    y_pred = np.array([12.0, 18.0, 50.0])

    mape = mean_absolute_percentage_error(y_true, y_pred)

    # |2/10| + |2/20| + |0/50| = 0.2 + 0.1 + 0 -> mean 0.1 -> 10%
    assert mape == pytest.approx(10.0)


def test_mape_excludes_zero_targets() -> None:
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([5.0, 11.0])

    mape = mean_absolute_percentage_error(y_true, y_pred)

    assert mape == pytest.approx(10.0)


def test_mape_all_zero_targets_raises() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="every y_true value is zero"):
        mean_absolute_percentage_error(y_true, y_pred)
