"""Save and load fitted pipelines and their metadata with joblib."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from surrogate_lab.core.ood import OODDetector


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata saved alongside a fitted pipeline, needed to interpret predictions."""

    model_name: str
    features: list[str]
    categorical_features: list[str]
    target: str
    uncertainty_method: str
    confidence_level: float | None
    prediction_interval_half_width: float | None
    n_bootstrap_estimators: int | None
    ensemble_disagreement_threshold: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "features": self.features,
            "categorical_features": self.categorical_features,
            "target": self.target,
            "uncertainty_method": self.uncertainty_method,
            "confidence_level": self.confidence_level,
            "prediction_interval_half_width": self.prediction_interval_half_width,
            "n_bootstrap_estimators": self.n_bootstrap_estimators,
            "ensemble_disagreement_threshold": self.ensemble_disagreement_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            model_name=data["model_name"],
            features=list(data["features"]),
            categorical_features=list(data["categorical_features"]),
            target=data["target"],
            uncertainty_method=data.get("uncertainty_method", "residual"),
            confidence_level=data["confidence_level"],
            prediction_interval_half_width=data["prediction_interval_half_width"],
            n_bootstrap_estimators=data.get("n_bootstrap_estimators"),
            ensemble_disagreement_threshold=data.get("ensemble_disagreement_threshold"),
        )


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def _ensemble_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".ensemble.joblib")


def _ood_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".ood.joblib")


def save_artifact(
    pipeline: Pipeline,
    metadata: ModelMetadata,
    model_path: Path | str,
    *,
    ensemble_pipelines: list[Pipeline] | None = None,
    ood_detector: OODDetector | None = None,
) -> None:
    """Save a fitted pipeline to ``model_path``, its metadata alongside it, and optionally
    the bootstrap ensemble members and/or the out-of-distribution detector needed to
    reproduce prediction intervals and OOD assessments at ``predict`` time.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    _metadata_path(model_path).write_text(json.dumps(metadata.as_dict(), indent=2))
    if ensemble_pipelines is not None:
        joblib.dump(ensemble_pipelines, _ensemble_path(model_path))
    if ood_detector is not None:
        joblib.dump(ood_detector, _ood_path(model_path))


def load_artifact(
    model_path: Path | str,
) -> tuple[Pipeline, ModelMetadata, list[Pipeline] | None, OODDetector | None]:
    """Load a fitted pipeline, its metadata, and any bootstrap ensemble / OOD detector
    previously saved with :func:`save_artifact`. The ensemble and OOD detector are ``None``
    when the model was not trained with them (e.g. ``uncertainty_method="residual"`` never
    saves an ensemble).

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

    ensemble_pipelines: list[Pipeline] | None = None
    ensemble_path = _ensemble_path(model_path)
    if ensemble_path.exists():
        ensemble_pipelines = joblib.load(ensemble_path)

    ood_detector: OODDetector | None = None
    ood_path = _ood_path(model_path)
    if ood_path.exists():
        ood_detector = joblib.load(ood_path)

    return pipeline, metadata, ensemble_pipelines, ood_detector
