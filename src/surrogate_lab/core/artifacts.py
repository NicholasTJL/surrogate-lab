"""Save and load fitted pipelines and their metadata with joblib."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata saved alongside a fitted pipeline, needed to interpret predictions."""

    model_name: str
    features: list[str]
    categorical_features: list[str]
    target: str
    prediction_interval_half_width: float | None
    confidence_level: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "features": self.features,
            "categorical_features": self.categorical_features,
            "target": self.target,
            "prediction_interval_half_width": self.prediction_interval_half_width,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            model_name=data["model_name"],
            features=list(data["features"]),
            categorical_features=list(data["categorical_features"]),
            target=data["target"],
            prediction_interval_half_width=data["prediction_interval_half_width"],
            confidence_level=data["confidence_level"],
        )


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def save_artifact(pipeline: Pipeline, metadata: ModelMetadata, model_path: Path | str) -> None:
    """Save a fitted pipeline to ``model_path`` and its metadata alongside it."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    _metadata_path(model_path).write_text(json.dumps(metadata.as_dict(), indent=2))


def load_artifact(model_path: Path | str) -> tuple[Pipeline, ModelMetadata]:
    """Load a fitted pipeline and its metadata previously saved with :func:`save_artifact`.

    Raises:
        FileNotFoundError: If the model file or its metadata sidecar is missing.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    meta_path = _metadata_path(model_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"model metadata not found: {meta_path}")

    pipeline: Pipeline = joblib.load(model_path)
    metadata = ModelMetadata.from_dict(json.loads(meta_path.read_text()))
    return pipeline, metadata
