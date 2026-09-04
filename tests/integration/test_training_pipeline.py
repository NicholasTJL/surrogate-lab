from pathlib import Path

from surrogate_lab.core.config import load_config
from surrogate_lab.core.data import load_dataset
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
        assert trained.prediction_interval.half_width >= 0
        # Fitted pipeline can predict on the held-out test split without error.
        predictions = trained.pipeline.predict(result.split.test[result.schema.features])
        assert len(predictions) == len(result.split.test)


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
