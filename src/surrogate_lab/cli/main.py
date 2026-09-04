"""surrogate-lab command-line interface: ``train`` and ``predict``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from surrogate_lab.core.artifacts import ModelMetadata, load_artifact, save_artifact
from surrogate_lab.core.config import ConfigError, load_config
from surrogate_lab.core.data import DataError, load_dataset
from surrogate_lab.core.ood import assess_ood
from surrogate_lab.core.registry import UnknownModelError
from surrogate_lab.core.training import train_and_evaluate
from surrogate_lab.core.uncertainty import (
    BootstrapPredictionInterval,
    ensemble_disagreement,
    ensemble_predict,
)
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
            uncertainty_method=trained.uncertainty_method,
            confidence_level=trained.confidence_level,
            prediction_interval_half_width=(
                trained.prediction_interval.half_width
                if trained.prediction_interval is not None
                else None
            ),
            n_bootstrap_estimators=(
                trained.bootstrap_interval.n_estimators
                if trained.bootstrap_interval is not None
                else None
            ),
            ensemble_disagreement_threshold=trained.ensemble_disagreement_threshold,
        )
        save_artifact(
            trained.pipeline,
            metadata,
            models_dir / f"{name}.joblib",
            ensemble_pipelines=trained.ensemble_pipelines,
            ood_detector=result.ood_detector,
        )

    report_path = generate_report(result, output_dir / "report.html")

    typer.secho(f"{len(result.models)} model(s) trained.", fg=typer.colors.GREEN)
    for name, trained in result.models.items():
        metrics = trained.metrics_test
        typer.echo(
            f"  {name}: MAE={metrics.mae:.4g} RMSE={metrics.rmse:.4g} "
            f"R2={metrics.r2:.4g} MAPE={metrics.mape:.4g}% "
            f"(uncertainty: {trained.uncertainty_method})"
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
    ood: bool = typer.Option(
        True,
        "--ood/--no-ood",
        help="Include an out-of-distribution assessment per row (default: on, when the "
        "saved model has an OOD detector).",
    ),
) -> None:
    """Run a saved pipeline on new data and write predictions to a CSV file."""
    try:
        pipeline, metadata, ensemble_pipelines, ood_detector = load_artifact(model_file)
        df = load_dataset(input_file)
    except (FileNotFoundError, DataError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    missing = [column for column in metadata.features if column not in df.columns]
    if missing:
        typer.secho(f"input is missing declared feature columns: {missing}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    feature_df = df[metadata.features]
    predictions = pipeline.predict(feature_df)
    result_df = df.copy()
    result_df[f"{metadata.target}_predicted"] = predictions

    # Ensemble disagreement is an OOD signal independent of --interval (a bootstrap-trained
    # model's ensemble disagreement should still be checked even if the caller doesn't want
    # interval bounds in the output), so it's computed whenever either --interval or --ood
    # needs it, not gated behind --interval alone.
    disagreement: np.ndarray | None = None
    if metadata.uncertainty_method == "bootstrap" and ensemble_pipelines and (interval or ood):
        ensemble_preds = ensemble_predict(ensemble_pipelines, feature_df)
        disagreement = ensemble_disagreement(ensemble_preds)
        if interval:
            bootstrap_interval = BootstrapPredictionInterval(
                confidence_level=metadata.confidence_level or 0.9,
                n_estimators=len(ensemble_pipelines),
            )
            lower, upper = bootstrap_interval.bounds(ensemble_preds)
            result_df[f"{metadata.target}_predicted_lower"] = lower
            result_df[f"{metadata.target}_predicted_upper"] = upper
    elif interval and metadata.prediction_interval_half_width is not None:
        half_width = metadata.prediction_interval_half_width
        result_df[f"{metadata.target}_predicted_lower"] = predictions - half_width
        result_df[f"{metadata.target}_predicted_upper"] = predictions + half_width

    if ood and ood_detector is not None:
        within_domain = []
        warnings_joined = []
        nn_distances = []
        for i in range(len(feature_df)):
            row = feature_df.iloc[i]
            row_disagreement = float(disagreement[i]) if disagreement is not None else None
            assessment = assess_ood(
                ood_detector,
                row,
                ensemble_disagreement=row_disagreement,
                ensemble_disagreement_threshold=metadata.ensemble_disagreement_threshold,
            )
            within_domain.append(assessment.within_training_domain)
            warnings_joined.append("; ".join(assessment.warnings))
            nn_distances.append(assessment.nearest_neighbor_distance)
        result_df["within_training_domain"] = within_domain
        result_df["ood_warnings"] = warnings_joined
        result_df["nearest_neighbor_distance"] = nn_distances

        n_flagged = sum(1 for flag in within_domain if not flag)
        if n_flagged:
            typer.secho(
                f"{n_flagged} of {len(within_domain)} prediction(s) flagged as outside the "
                "training domain (see the ood_warnings column).",
                fg=typer.colors.YELLOW,
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    typer.secho(f"Wrote {len(result_df)} prediction(s) to {output_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
