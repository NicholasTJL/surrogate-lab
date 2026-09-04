"""YAML/dict-based experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from surrogate_lab.core.schema import DatasetSchema


class ConfigError(Exception):
    """Raised when a config file cannot be read or fails validation."""


class DataConfig(BaseModel):
    """Where to load the dataset from."""

    path: str
    format: Literal["csv", "parquet"] | None = None


class SplitConfig(BaseModel):
    """Train/validation/test split parameters."""

    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class CrossValidationConfig(BaseModel):
    """Cross-validation settings applied during training."""

    enabled: bool = True
    n_splits: int = 5


class UncertaintyConfig(BaseModel):
    """Prediction interval settings.

    ``method`` selects between the two v0.2.0 alternatives: ``"residual"`` (the
    original v0.1.0 approach — a single global half-width from validation
    residuals) or ``"bootstrap"`` (a per-point interval from the spread of an
    N-member bootstrap ensemble; see ``surrogate_lab.core.uncertainty`` for
    the real tradeoff between the two). ``n_bootstrap_estimators`` is only
    used when ``method == "bootstrap"``.
    """

    method: Literal["residual", "bootstrap"] = "residual"
    confidence_level: float = Field(default=0.9, gt=0, lt=1)
    n_bootstrap_estimators: int = Field(default=30, ge=2)


class OODConfig(BaseModel):
    """Out-of-distribution / extrapolation detection settings.

    ``nn_distance_percentile`` calibrates the nearest-neighbour distance
    threshold from the training set's own leave-one-out neighbour distances
    (see ``surrogate_lab.core.ood`` for exactly how). ``distance_metric``
    chooses Euclidean (default) or Mahalanobis distance in standardized
    numeric feature space. ``ensemble_disagreement_percentile`` calibrates
    the bootstrap-ensemble disagreement warning threshold (only used when
    ``uncertainty.method == "bootstrap"``) from the spread observed across
    ensemble members on the validation set.
    """

    enabled: bool = True
    nn_distance_percentile: float = Field(default=0.95, gt=0, lt=1)
    distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean"
    ensemble_disagreement_percentile: float = Field(default=0.95, gt=0, lt=1)


class ErrorAnalysisConfig(BaseModel):
    """Error-by-region breakdown settings.

    ``region_feature`` names a feature to bin by; when omitted (the default),
    error is broken down by quantile bin of the target itself.
    """

    enabled: bool = True
    region_feature: str | None = None
    n_bins: int = Field(default=5, ge=1)


class ExperimentConfig(BaseModel):
    """Full configuration for a training experiment."""

    data: DataConfig
    dataset_schema: DatasetSchema
    models: list[str]
    split: SplitConfig = Field(default_factory=SplitConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    ood: OODConfig = Field(default_factory=OODConfig)
    error_analysis: ErrorAnalysisConfig = Field(default_factory=ErrorAnalysisConfig)
    output_dir: str = "outputs"
    random_state: int = 42

    @model_validator(mode="after")
    def error_analysis_region_feature_must_be_numeric(self) -> ExperimentConfig:
        region_feature = self.error_analysis.region_feature
        numeric_features = self.dataset_schema.numeric_features
        if region_feature is not None and region_feature not in numeric_features:
            raise ValueError(
                f"error_analysis.region_feature {region_feature!r} must be one of "
                f"dataset_schema's numeric features (quantile binning needs a numeric "
                f"variable): {numeric_features}"
            )
        return self


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Raises:
        ConfigError: If the file does not exist, is not valid YAML, or does
            not match :class:`ExperimentConfig`.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised as a project error
        raise ConfigError(f"invalid config in {path}: {exc}") from exc
