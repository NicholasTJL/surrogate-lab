from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from surrogate_lab.core.artifacts import ModelMetadata, load_artifact, save_artifact
from surrogate_lab.core.ood import fit_ood_detector
from surrogate_lab.core.schema import DatasetSchema


def _fitted_pipeline() -> Pipeline:
    pipeline = Pipeline(steps=[("model", LinearRegression())])
    pipeline.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([1.0, 2.0, 3.0]))
    return pipeline


def _metadata(**overrides: object) -> ModelMetadata:
    defaults: dict[str, object] = dict(
        model_name="linear_regression",
        features=["x"],
        categorical_features=[],
        target="y",
        uncertainty_method="residual",
        confidence_level=0.9,
        prediction_interval_half_width=0.5,
        n_bootstrap_estimators=None,
        ensemble_disagreement_threshold=None,
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)  # type: ignore[arg-type]


def test_save_and_load_residual_artifact_round_trips(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    metadata = _metadata()
    model_path = tmp_path / "model.joblib"

    save_artifact(pipeline, metadata, model_path)
    loaded_pipeline, loaded_metadata, ensemble, ood_detector = load_artifact(model_path)

    assert loaded_metadata == metadata
    assert ensemble is None
    assert ood_detector is None
    np.testing.assert_array_equal(
        loaded_pipeline.predict(pd.DataFrame({"x": [4.0]})),
        pipeline.predict(pd.DataFrame({"x": [4.0]})),
    )


def test_save_and_load_bootstrap_ensemble_round_trips(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    ensemble_pipelines = [_fitted_pipeline() for _ in range(5)]
    metadata = _metadata(
        uncertainty_method="bootstrap",
        prediction_interval_half_width=None,
        n_bootstrap_estimators=5,
        ensemble_disagreement_threshold=0.2,
    )
    model_path = tmp_path / "model.joblib"

    save_artifact(pipeline, metadata, model_path, ensemble_pipelines=ensemble_pipelines)
    _, loaded_metadata, loaded_ensemble, ood_detector = load_artifact(model_path)

    assert loaded_metadata.uncertainty_method == "bootstrap"
    assert loaded_metadata.n_bootstrap_estimators == 5
    assert loaded_ensemble is not None
    assert len(loaded_ensemble) == 5
    assert ood_detector is None


def test_save_and_load_ood_detector_round_trips(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    x_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    schema = DatasetSchema(features=["x"], target="y")
    detector = fit_ood_detector(x_train, schema)
    metadata = _metadata()
    model_path = tmp_path / "model.joblib"

    save_artifact(pipeline, metadata, model_path, ood_detector=detector)
    _, _, ensemble, loaded_detector = load_artifact(model_path)

    assert ensemble is None
    assert loaded_detector is not None
    assert loaded_detector.numeric_features == ["x"]
    assert loaded_detector.feature_min == {"x": 1.0}
    assert loaded_detector.feature_max == {"x": 4.0}


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        load_artifact(tmp_path / "missing.joblib")


def test_load_missing_metadata_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    pipeline = _fitted_pipeline()
    import joblib

    joblib.dump(pipeline, model_path)

    with pytest.raises(FileNotFoundError, match="model metadata not found"):
        load_artifact(model_path)


def test_metadata_as_dict_and_from_dict_round_trip() -> None:
    metadata = _metadata(uncertainty_method="bootstrap", n_bootstrap_estimators=10)

    restored = ModelMetadata.from_dict(metadata.as_dict())

    assert restored == metadata


def test_from_dict_defaults_uncertainty_method_for_old_artifacts() -> None:
    # Metadata saved by a pre-v0.2.0 artifact has no uncertainty_method key.
    legacy = {
        "model_name": "linear_regression",
        "features": ["x"],
        "categorical_features": [],
        "target": "y",
        "confidence_level": 0.9,
        "prediction_interval_half_width": 0.5,
    }

    restored = ModelMetadata.from_dict(legacy)

    assert restored.uncertainty_method == "residual"
    assert restored.n_bootstrap_estimators is None
