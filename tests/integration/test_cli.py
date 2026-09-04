import json
from pathlib import Path

import pandas as pd
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


def test_predict_includes_ood_assessment_by_default(tmp_path: Path) -> None:
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
    train_result = runner.invoke(app, ["train", str(config_file)])
    assert train_result.exit_code == 0, train_result.output

    # One in-domain row (copied straight from training data) and one far-out-of-range row.
    input_file = tmp_path / "input.csv"
    input_file.write_text(
        "length_m,load_n,moment_of_inertia_m4,material,deflection_m\n"
        "1.5,500,1.2e-6,steel,0.01\n"
        "10000.0,500,1.2e-6,steel,0.01\n"
    )
    predictions_file = tmp_path / "predictions.csv"

    result = runner.invoke(
        app,
        [
            "predict",
            str(output_dir / "models" / "linear_regression.joblib"),
            str(input_file),
            str(predictions_file),
        ],
    )

    assert result.exit_code == 0, result.output
    content = predictions_file.read_text()
    header = content.splitlines()[0]
    assert "within_training_domain" in header
    assert "ood_warnings" in header
    assert "nearest_neighbor_distance" in header
    rows = content.splitlines()[1:]
    assert "False" in rows[1]  # the out-of-range row is flagged
    assert "flagged as outside the training domain" in result.output


def test_predict_no_ood_flag_omits_ood_columns(tmp_path: Path) -> None:
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

    predictions_file = tmp_path / "predictions.csv"
    result = runner.invoke(
        app,
        [
            "predict",
            str(output_dir / "models" / "linear_regression.joblib"),
            "tests/fixtures/sample_data.csv",
            str(predictions_file),
            "--no-ood",
        ],
    )

    assert result.exit_code == 0, result.output
    header = predictions_file.read_text().splitlines()[0]
    assert "within_training_domain" not in header


def test_train_and_predict_with_bootstrap_uncertainty(tmp_path: Path) -> None:
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
uncertainty:
  method: bootstrap
  n_bootstrap_estimators: 6
output_dir: {output_dir.as_posix()}
"""
    )

    train_result = runner.invoke(app, ["train", str(config_file)])
    assert train_result.exit_code == 0, train_result.output
    assert "bootstrap" in train_result.output

    model_path = output_dir / "models" / "linear_regression.joblib"
    assert (output_dir / "models" / "linear_regression.joblib.ensemble.joblib").exists()
    assert (output_dir / "models" / "linear_regression.joblib.ood.joblib").exists()

    predictions_file = tmp_path / "predictions.csv"
    predict_result = runner.invoke(
        app,
        [
            "predict",
            str(model_path),
            "tests/fixtures/sample_data.csv",
            str(predictions_file),
            "--interval",
        ],
    )

    assert predict_result.exit_code == 0, predict_result.output
    predictions_df = pd.read_csv(predictions_file)
    lower = predictions_df["deflection_m_predicted_lower"]
    upper = predictions_df["deflection_m_predicted_upper"]
    assert (lower <= upper).all()
    # Bootstrap gives a per-point interval width, not a single constant one.
    assert (upper - lower).nunique() > 1


def test_ood_ensemble_disagreement_check_runs_without_interval_flag(tmp_path: Path) -> None:
    # Regression test: ensemble-disagreement is an OOD signal independent of --interval, so
    # it must be computed on `predict --ood` (the default) even when --interval is *not*
    # passed. Force the disagreement threshold to 0 after training so any nonzero disagreement
    # (near-certain across bootstrap-resampled random forest members) trips the flag, proving
    # the ensemble predictions were actually computed in this code path.
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
models: [random_forest]
cross_validation:
  enabled: false
uncertainty:
  method: bootstrap
  n_bootstrap_estimators: 5
output_dir: {output_dir.as_posix()}
"""
    )
    train_result = runner.invoke(app, ["train", str(config_file)])
    assert train_result.exit_code == 0, train_result.output

    meta_path = output_dir / "models" / "random_forest.joblib.meta.json"
    metadata = json.loads(meta_path.read_text())
    metadata["ensemble_disagreement_threshold"] = 0.0
    meta_path.write_text(json.dumps(metadata))

    predictions_file = tmp_path / "predictions.csv"
    predict_result = runner.invoke(
        app,
        [
            "predict",
            str(output_dir / "models" / "random_forest.joblib"),
            "tests/fixtures/sample_data.csv",
            str(predictions_file),
            # deliberately no --interval
        ],
    )

    assert predict_result.exit_code == 0, predict_result.output
    predictions_df = pd.read_csv(predictions_file)
    assert "deflection_m_predicted_lower" not in predictions_df.columns  # --interval not passed
    assert predictions_df["ood_warnings"].str.contains("disagree").any()
