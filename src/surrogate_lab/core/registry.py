"""A small registry mapping model names to scikit-learn-compatible estimators.

Kept intentionally light-dependency for v0.1.0: only scikit-learn estimators
are registered by default (linear regression, random forest, gradient
boosting). Gaussian-process regression, PyTorch networks, and other model
families from the product vision are deferred to later releases — see
docs/vision.md.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

ModelFactory = Callable[[int], BaseEstimator]


class UnknownModelError(KeyError):
    """Raised when a model name is not present in the registry."""


def _linear_regression(_random_state: int) -> BaseEstimator:
    return LinearRegression()


def _random_forest(random_state: int) -> BaseEstimator:
    return RandomForestRegressor(n_estimators=200, random_state=random_state)


def _gradient_boosting(random_state: int) -> BaseEstimator:
    return GradientBoostingRegressor(random_state=random_state)


_REGISTRY: dict[str, ModelFactory] = {
    "linear_regression": _linear_regression,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
}


def list_models() -> list[str]:
    """Return the names of all registered models, sorted alphabetically."""
    return sorted(_REGISTRY)


def register_model(name: str, factory: ModelFactory) -> None:
    """Register a new model factory under ``name``, overwriting any existing entry."""
    _REGISTRY[name] = factory


def get_model(name: str, random_state: int = 42) -> BaseEstimator:
    """Instantiate a fresh, unfitted estimator for ``name``.

    Raises:
        UnknownModelError: If ``name`` is not registered. The error message
            lists the available model names.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise UnknownModelError(
            f"unknown model {name!r}; available models: {list_models()}"
        ) from exc
    return factory(random_state)
