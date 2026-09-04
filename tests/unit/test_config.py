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
    assert config.uncertainty.method == "residual"
    assert config.uncertainty.n_bootstrap_estimators == 30
    assert config.ood.enabled is True
    assert config.ood.nn_distance_percentile == 0.95
    assert config.ood.distance_metric == "euclidean"
    assert config.error_analysis.region_feature is None
    assert config.error_analysis.n_bins == 5
    assert config.output_dir == "outputs"


def test_bootstrap_uncertainty_method_parses(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
uncertainty:
  method: bootstrap
  n_bootstrap_estimators: 10
"""
    )

    config = load_config(config_file)

    assert config.uncertainty.method == "bootstrap"
    assert config.uncertainty.n_bootstrap_estimators == 10


def test_mahalanobis_distance_metric_parses(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
ood:
  distance_metric: mahalanobis
  nn_distance_percentile: 0.99
"""
    )

    config = load_config(config_file)

    assert config.ood.distance_metric == "mahalanobis"
    assert config.ood.nn_distance_percentile == 0.99


def test_error_analysis_region_feature_must_be_a_declared_feature(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
error_analysis:
  region_feature: not_a_feature
"""
    )

    with pytest.raises(ConfigError, match="must be one of dataset_schema's numeric features"):
        load_config(config_file)


def test_error_analysis_region_feature_rejects_categorical_feature(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a, material]
  target: y
  categorical_features: [material]
models: [linear_regression]
error_analysis:
  region_feature: material
"""
    )

    with pytest.raises(ConfigError, match="must be one of dataset_schema's numeric features"):
        load_config(config_file)


def test_error_analysis_region_feature_accepts_declared_feature(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a, b]
  target: y
models: [linear_regression]
error_analysis:
  region_feature: b
"""
    )

    config = load_config(config_file)

    assert config.error_analysis.region_feature == "b"


def test_n_bootstrap_estimators_below_two_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
uncertainty:
  method: bootstrap
  n_bootstrap_estimators: 1
"""
    )

    with pytest.raises(ConfigError, match="invalid config"):
        load_config(config_file)


def test_confidence_level_outside_open_interval_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
uncertainty:
  confidence_level: 1.5
"""
    )

    with pytest.raises(ConfigError, match="invalid config"):
        load_config(config_file)


def test_nn_distance_percentile_outside_open_interval_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
ood:
  nn_distance_percentile: 0
"""
    )

    with pytest.raises(ConfigError, match="invalid config"):
        load_config(config_file)


def test_n_bins_below_one_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: data.csv
dataset_schema:
  features: [a]
  target: y
models: [linear_regression]
error_analysis:
  n_bins: 0
"""
    )

    with pytest.raises(ConfigError, match="invalid config"):
        load_config(config_file)
