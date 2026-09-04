import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from surrogate_lab.core.metrics import compute_metrics
from surrogate_lab.core.model_card import build_model_card
from surrogate_lab.core.schema import DatasetSchema


def _fitted_model(x_train: pd.DataFrame, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def test_model_card_captures_real_training_data_facts() -> None:
    schema = DatasetSchema(features=["x1", "x2", "cat"], target="y", categorical_features=["cat"])
    x_train = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "cat": ["a", "b", "a", "c"],
        }
    )
    y_train = np.array([5.0, 6.0, 7.0, 8.0])
    model = _fitted_model(x_train[["x1", "x2"]], y_train)
    y_pred = model.predict(x_train[["x1", "x2"]])

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=2,
        test_rows=3,
        metrics_test=compute_metrics(y_train, y_pred),
        cv_mae_scores=np.array([0.1, 0.2, 0.3]),
        uncertainty_method="residual",
        confidence_level=0.9,
        interval_coverage_test=0.85,
    )

    assert card.model_name == "linear_regression"
    assert card.model_type == "LinearRegression"
    assert card.training_rows == 4
    assert card.validation_rows == 2
    assert card.test_rows == 3
    assert card.numeric_feature_ranges == {"x1": (1.0, 4.0), "x2": (10.0, 40.0)}
    assert card.categorical_feature_values == {"cat": ["a", "b", "c"]}
    assert card.target_name == "y"
    assert card.target_range == (5.0, 8.0)
    assert card.cv_mae_mean == pytest.approx(0.2)
    assert card.interval_coverage_test == 0.85


def test_model_card_without_cross_validation_has_no_cv_mean() -> None:
    schema = DatasetSchema(features=["x1"], target="y")
    x_train = pd.DataFrame({"x1": [1.0, 2.0]})
    y_train = np.array([1.0, 2.0])
    model = _fitted_model(x_train, y_train)

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=1,
        test_rows=1,
        metrics_test=compute_metrics(y_train, y_train),
        cv_mae_scores=np.array([]),
        uncertainty_method="residual",
        confidence_level=0.9,
        interval_coverage_test=0.9,
    )

    assert card.cv_mae_mean is None


def test_hyperparameters_reflect_actual_fitted_estimator_params() -> None:
    schema = DatasetSchema(features=["x1"], target="y")
    x_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])
    model = LinearRegression(fit_intercept=False)
    model.fit(x_train, y_train)

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=0,
        test_rows=0,
        metrics_test=compute_metrics(y_train, y_train),
        cv_mae_scores=np.array([]),
        uncertainty_method="residual",
        confidence_level=0.9,
        interval_coverage_test=0.9,
    )

    assert card.hyperparameters["fit_intercept"] is False


def test_bootstrap_limitation_text_mentions_n_estimators_and_cost() -> None:
    schema = DatasetSchema(features=["x1"], target="y")
    x_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])
    model = _fitted_model(x_train, y_train)

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=1,
        test_rows=1,
        metrics_test=compute_metrics(y_train, y_train),
        cv_mae_scores=np.array([]),
        uncertainty_method="bootstrap",
        confidence_level=0.9,
        interval_coverage_test=0.88,
        n_bootstrap_estimators=25,
    )

    joined = " ".join(card.limitations)
    assert "25" in joined
    assert "model-form uncertainty" in joined


def test_residual_limitation_text_mentions_homoscedasticity() -> None:
    schema = DatasetSchema(features=["x1"], target="y")
    x_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])
    model = _fitted_model(x_train, y_train)

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=1,
        test_rows=1,
        metrics_test=compute_metrics(y_train, y_train),
        cv_mae_scores=np.array([]),
        uncertainty_method="residual",
        confidence_level=0.9,
        interval_coverage_test=0.9,
    )

    joined = " ".join(card.limitations)
    assert "homoscedastic" in joined


def test_low_coverage_gap_flagged_in_limitations() -> None:
    schema = DatasetSchema(features=["x1"], target="y")
    x_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])
    model = _fitted_model(x_train, y_train)

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=1,
        test_rows=1,
        metrics_test=compute_metrics(y_train, y_train),
        cv_mae_scores=np.array([]),
        uncertainty_method="residual",
        confidence_level=0.9,
        interval_coverage_test=0.5,  # big gap from 0.9 target
    )

    joined = " ".join(card.limitations)
    assert "gap from the target" in joined


def test_intended_use_and_out_of_scope_use_are_non_empty() -> None:
    schema = DatasetSchema(features=["x1"], target="y")
    x_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])
    model = _fitted_model(x_train, y_train)

    card = build_model_card(
        model_name="linear_regression",
        fitted_estimator=model,
        schema=schema,
        x_train=x_train,
        y_train=y_train,
        validation_rows=1,
        test_rows=1,
        metrics_test=compute_metrics(y_train, y_train),
        cv_mae_scores=np.array([]),
        uncertainty_method="residual",
        confidence_level=0.9,
        interval_coverage_test=0.9,
    )

    assert len(card.intended_use) > 0
    assert len(card.out_of_scope_use) > 0
