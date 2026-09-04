"""surrogate-lab command-line interface: ``train`` and ``predict``."""

from __future__ import annotations

from pathlib import Path

import typer

from surrogate_lab.core.artifacts import ModelMetadata, load_artifact, save_artifact
from surrogate_lab.core.config import ConfigError, load_config
from surrogate_lab.core.data import DataError, load_dataset
from surrogate_lab.core.registry import UnknownModelError
from surrogate_lab.core.training import train_and_evaluate
from surrogate_lab.reporting.report import generate_report

app = typer.Typer(help="surrogate-lab: train, compare, and validate surrogate models.")


@app.command()
def train(config_file: Path) -> None:
    """Train every model in a config file, save artifacts, and generate an HTML report."""
    try:
        config = load_config(config_file)
        df = load_dataset(config.data.path, config.data.format)
        result = train_and_evaluate(df, config)
    except (ConfigError, DataError, UnknownModelError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    output_dir = Path(config.output_dir)
    models_dir = output_dir / "models"
    for name, trained in result.models.items():
        metadata = ModelMetadata(
            model_name=name,
            features=result.schema.features,
            categorical_features=result.schema.categorical_features,
            target=result.schema.target,
            prediction_interval_half_width=trained.prediction_interval.half_width,
            confidence_level=trained.prediction_interval.confidence_level,
        )
        save_artifact(trained.pipeline, metadata, models_dir / f"{name}.joblib")

    report_path = generate_report(result, output_dir / "report.html")

    typer.secho(f"{len(result.models)} model(s) trained.", fg=typer.colors.GREEN)
    for name, trained in result.models.items():
        metrics = trained.metrics_test
        typer.echo(
            f"  {name}: MAE={metrics.mae:.4g} RMSE={metrics.rmse:.4g} "
            f"R2={metrics.r2:.4g} MAPE={metrics.mape:.4g}%"
        )
    typer.echo(f"Artifacts saved under {models_dir}")
    typer.echo(f"Report written to {report_path}")


@app.command()
def predict(
    model_file: Path,
    input_file: Path,
    output_file: Path,
    interval: bool = typer.Option(
        False, "--interval", help="Include prediction interval bounds in the output."
    ),
) -> None:
    """Run a saved pipeline on new data and write predictions to a CSV file."""
    try:
        pipeline, metadata = load_artifact(model_file)
        df = load_dataset(input_file)
    except (FileNotFoundError, DataError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    missing = [column for column in metadata.features if column not in df.columns]
    if missing:
        typer.secho(f"input is missing declared feature columns: {missing}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    predictions = pipeline.predict(df[metadata.features])
    result_df = df.copy()
    result_df[f"{metadata.target}_predicted"] = predictions

    if interval and metadata.prediction_interval_half_width is not None:
        half_width = metadata.prediction_interval_half_width
        result_df[f"{metadata.target}_predicted_lower"] = predictions - half_width
        result_df[f"{metadata.target}_predicted_upper"] = predictions + half_width

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    typer.secho(f"Wrote {len(result_df)} prediction(s) to {output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
