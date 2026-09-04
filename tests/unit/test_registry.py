import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from surrogate_lab.core.registry import (
    UnknownModelError,
    get_model,
    list_models,
    register_model,
)


def test_list_models_includes_expected_defaults() -> None:
    models = list_models()

    assert models == sorted(models)
    assert {"linear_regression", "random_forest", "gradient_boosting"} <= set(models)


def test_get_model_returns_expected_types() -> None:
    assert isinstance(get_model("linear_regression"), LinearRegression)
    assert isinstance(get_model("random_forest", random_state=1), RandomForestRegressor)
    assert isinstance(get_model("gradient_boosting", random_state=1), GradientBoostingRegressor)


def test_get_model_passes_random_state() -> None:
    model = get_model("random_forest", random_state=123)

    assert model.random_state == 123


def test_unknown_model_raises_with_available_list() -> None:
    with pytest.raises(UnknownModelError, match="unknown model 'nope'"):
        get_model("nope")


def test_register_model_adds_new_entry() -> None:
    register_model("always_linear", lambda seed: LinearRegression())

    assert "always_linear" in list_models()
    assert isinstance(get_model("always_linear"), LinearRegression)
