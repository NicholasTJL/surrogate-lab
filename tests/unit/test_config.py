from pathlib import Path

import pytest

from surrogate_lab.core.config import ConfigError, ExperimentConfig, load_config


def test_load_valid_config() -> None:
    config = load_config(Path("tests/fixtures/sample_config.yaml"))

    assert isinstance(config, ExperimentConfig)
    assert config.models == ["linear_regression", "random_forest"]
    assert config.dataset_schema.target == "deflection_m"
    assert config.split.random_state == 42


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data: [unclosed")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_file)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- just\n- a\n- list\n")

    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(config_file)


def test_schema_validation_error_wrapped(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: []
  target: y
models: [linear_regression]
"""
    )

    with pytest.raises(ConfigError, match="invalid config"):
        load_config(config_file)


def test_defaults_applied(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
"""
    )

    config = load_config(config_file)

    assert config.split.test_size == 0.2
    assert config.cross_validation.n_splits == 5
    assert config.uncertainty.confidence_level == 0.9
    assert config.output_dir == "outputs"
