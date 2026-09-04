"""YAML/dict-based experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

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
    """Residual-based prediction interval settings."""

    confidence_level: float = 0.9


class ExperimentConfig(BaseModel):
    """Full configuration for a training experiment."""

    data: DataConfig
    dataset_schema: DatasetSchema
    models: list[str]
    split: SplitConfig = Field(default_factory=SplitConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    output_dir: str = "outputs"
    random_state: int = 42


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
