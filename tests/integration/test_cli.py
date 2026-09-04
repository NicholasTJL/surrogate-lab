from pathlib import Path

from typer.testing import CliRunner

from surrogate_lab.cli.main import app

runner = CliRunner()


def test_train_command_writes_artifacts_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
data:
  path: tests/fixtures/sample_data.csv
  format: csv
dataset_schema:
  features: [length_m, load_n, moment_of_inertia_m4, material]
  target: deflection_m
  categorical_features: [material]
models: [linear_regression]
cross_validation:
  enabled: false
output_dir: {output_dir.as_posix()}
"""
    )

    result = runner.invoke(app, ["train", str(config_file)])

    assert result.exit_code == 0, result.output
    assert (output_dir / "report.html").exists()
    assert (output_dir / "models" / "linear_regression.joblib").exists()
    assert (output_dir / "models" / "linear_regression.joblib.meta.json").exists()
    assert "trained" in result.output.lower()


def test_train_command_reports_unknown_model(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
data:
  path: tests/fixtures/sample_data.csv
dataset_schema:
  features: [length_m, load_n, moment_of_inertia_m4, material]
  target: deflection_m
  categorical_features: [material]
models: [not_a_real_model]
"""
    )

    result = runner.invoke(app, ["train", str(config_file)])

    assert result.exit_code == 1
    assert "unknown model" in result.output.lower()


def test_train_then_predict_round_trip(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
data:
  path: tests/fixtures/sample_data.csv
dataset_schema:
  features: [length_m, load_n, moment_of_inertia_m4, material]
  target: deflection_m
  categorical_features: [material]
models: [linear_regression]
cross_validation:
  enabled: false
output_dir: {output_dir.as_posix()}
"""
    )
    train_result = runner.invoke(app, ["train", str(config_file)])
    assert train_result.exit_code == 0, train_result.output

    predictions_file = tmp_path / "predictions.csv"
    predict_result = runner.invoke(
        app,
        [
            "predict",
            str(output_dir / "models" / "linear_regression.joblib"),
            "tests/fixtures/sample_data.csv",
            str(predictions_file),
            "--interval",
        ],
    )

    assert predict_result.exit_code == 0, predict_result.output
    assert predictions_file.exists()
    header = predictions_file.read_text().splitlines()[0]
    assert "deflection_m_predicted" in header
    assert "deflection_m_predicted_lower" in header
    assert "deflection_m_predicted_upper" in header


def test_predict_missing_feature_column_fails(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
data:
  path: tests/fixtures/sample_data.csv
dataset_schema:
  features: [length_m, load_n, moment_of_inertia_m4, material]
  target: deflection_m
  categorical_features: [material]
models: [linear_regression]
cross_validation:
  enabled: false
output_dir: {output_dir.as_posix()}
"""
    )
    runner.invoke(app, ["train", str(config_file)])

    bad_input = tmp_path / "bad_input.csv"
    bad_input.write_text("length_m,load_n\n1.0,2.0\n")

    result = runner.invoke(
        app,
        [
            "predict",
            str(output_dir / "models" / "linear_regression.joblib"),
            str(bad_input),
            str(tmp_path / "out.csv"),
        ],
    )

    assert result.exit_code == 1
    assert "missing declared feature columns" in result.output
