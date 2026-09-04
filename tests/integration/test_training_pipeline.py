from pathlib import Path

import numpy as np

from surrogate_lab.core.config import ExperimentConfig, UncertaintyConfig, load_config
from surrogate_lab.core.data import load_dataset
from surrogate_lab.core.ood import assess_ood
from surrogate_lab.core.training import train_and_evaluate


def test_train_and_evaluate_end_to_end() -> None:
    config = load_config(Path("tests/fixtures/sample_config.yaml"))
    df = load_dataset(config.data.path, config.data.format)

    result = train_and_evaluate(df, config)

    assert set(result.models) == {"linear_regression", "random_forest"}
    for trained in result.models.values():
        assert trained.metrics_test.mae >= 0
        assert trained.metrics_test.rmse >= 0
        assert trained.cv_mae_scores.size == config.cross_validation.n_splits
        assert len(trained.y_test_true) == len(result.split.test)
        assert trained.uncertainty_method == "residual"
        assert trained.prediction_interval is not None
        assert trained.prediction_interval.half_width >= 0
        assert trained.bootstrap_interval is None
        assert trained.ensemble_pipelines is None
        assert len(trained.y_test_lower) == len(result.split.test)
        assert (trained.y_test_lower <= trained.y_test_upper).all()
        assert trained.error_by_region.region_name == result.schema.target
        assert len(trained.error_by_region.bins) > 0
        assert trained.model_card.model_name == trained.name
        assert trained.model_card.training_rows == len(result.split.train)
        # Fitted pipeline can predict on the held-out test split without error.
        predictions = trained.pipeline.predict(result.split.test[result.schema.features])
        assert len(predictions) == len(result.split.test)

    # The OOD detector is fit once from the training split, shared across all models.
    assert result.ood_detector.numeric_features == result.schema.numeric_features
    training_point = result.split.train.iloc[0][result.schema.features]
    assessment = assess_ood(result.ood_detector, training_point)
    assert assessment.within_training_domain is True


def test_train_and_evaluate_is_reproducible() -> None:
    config = load_config(Path("tests/fixtures/sample_config.yaml"))
    df = load_dataset(config.data.path, config.data.format)

    result_a = train_and_evaluate(df, config)
    result_b = train_and_evaluate(df, config)

    for name in result_a.models:
        metrics_a = result_a.models[name].metrics_test
        metrics_b = result_b.models[name].metrics_test
        assert metrics_a.mae == metrics_b.mae
        assert metrics_a.rmse == metrics_b.rmse


def test_bootstrap_uncertainty_method_produces_per_point_intervals() -> None:
    base_config = load_config(Path("tests/fixtures/sample_config.yaml"))
    df = load_dataset(base_config.data.path, base_config.data.format)
    config = ExperimentConfig(
        **{
            **base_config.model_dump(),
            "models": ["linear_regression"],
            "uncertainty": UncertaintyConfig(method="bootstrap", n_bootstrap_estimators=8),
        }
    )

    result = train_and_evaluate(df, config)
    trained = result.models["linear_regression"]

    assert trained.uncertainty_method == "bootstrap"
    assert trained.prediction_interval is None
    assert trained.bootstrap_interval is not None
    assert trained.bootstrap_interval.n_estimators == 8
    assert trained.ensemble_pipelines is not None
    assert len(trained.ensemble_pipelines) == 8
    assert trained.ensemble_disagreement_threshold is not None
    assert trained.ensemble_disagreement_threshold >= 0
    # Per-point interval widths, unlike the residual method's single scalar half-width.
    half_widths = (trained.y_test_upper - trained.y_test_lower) / 2.0
    assert len(half_widths) == len(result.split.test)
    assert not np.allclose(half_widths, half_widths[0])  # genuinely varies point to point
    assert 0.0 <= trained.interval_coverage_test <= 1.0
    assert trained.model_card.uncertainty_method == "bootstrap"
