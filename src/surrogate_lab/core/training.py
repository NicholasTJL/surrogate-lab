"""Ties data, schema, preprocessing, the model registry, and metrics together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from surrogate_lab.core.config import ExperimentConfig
from surrogate_lab.core.data import DatasetSplit, split_dataset, validate_schema
from surrogate_lab.core.error_analysis import RegionErrorTable, error_by_region
from surrogate_lab.core.metrics import Metrics, compute_metrics
from surrogate_lab.core.model_card import ModelCard, build_model_card
from surrogate_lab.core.ood import OODDetector, fit_ood_detector
from surrogate_lab.core.preprocessing import build_preprocessor
from surrogate_lab.core.registry import get_model
from surrogate_lab.core.schema import DatasetSchema
from surrogate_lab.core.uncertainty import (
    BootstrapPredictionInterval,
    PredictionInterval,
    empirical_coverage,
    ensemble_disagreement,
    ensemble_predict,
    fit_bootstrap_ensemble,
    fit_prediction_interval,
)


@dataclass(frozen=True)
class TrainedModel:
    """A fitted pipeline plus everything needed to report on it."""

    name: str
    pipeline: Pipeline
    metrics_train: Metrics
    metrics_validation: Metrics
    metrics_test: Metrics
    cv_mae_scores: NDArray[np.float64]
    y_test_true: NDArray[np.float64]
    y_test_pred: NDArray[np.float64]
    uncertainty_method: str
    confidence_level: float
    y_test_lower: NDArray[np.float64]
    y_test_upper: NDArray[np.float64]
    interval_coverage_test: float
    error_by_region: RegionErrorTable
    model_card: ModelCard
    prediction_interval: PredictionInterval | None = None
    bootstrap_interval: BootstrapPredictionInterval | None = None
    ensemble_pipelines: list[Pipeline] | None = None
    ensemble_disagreement_threshold: float | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """The outcome of training every model requested in an :class:`ExperimentConfig`."""

    schema: DatasetSchema
    split: DatasetSplit
    models: dict[str, TrainedModel]
    ood_detector: OODDetector


def _xy(df: pd.DataFrame, schema: DatasetSchema) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    x = df[schema.features]
    y: NDArray[np.float64] = df[schema.target].to_numpy(dtype=float)
    return x, y


def _build_pipeline(schema: DatasetSchema, model_name: str, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(schema)),
            ("model", get_model(model_name, random_state=random_state)),
        ]
    )


def train_and_evaluate(df: pd.DataFrame, config: ExperimentConfig) -> ExperimentResult:
    """Validate, split, train, cross-validate, and evaluate every model in ``config.models``.

    Raises:
        DataError: If ``df`` does not match ``config.dataset_schema``.
        UnknownModelError: If ``config.models`` names an unregistered model.
    """
    schema = config.dataset_schema
    validate_schema(df, schema)

    split = split_dataset(
        df,
        test_size=config.split.test_size,
        val_size=config.split.val_size,
        random_state=config.split.random_state,
    )

    x_train, y_train = _xy(split.train, schema)
    x_val, y_val = _xy(split.validation, schema)
    x_test, y_test = _xy(split.test, schema)

    ood_detector = fit_ood_detector(
        x_train,
        schema,
        nn_distance_percentile=config.ood.nn_distance_percentile,
        distance_metric=config.ood.distance_metric,
    )

    region_feature = config.error_analysis.region_feature
    region_values_test = (
        split.test[region_feature].to_numpy(dtype=float) if region_feature else y_test
    )
    region_name = region_feature or schema.target

    trained: dict[str, TrainedModel] = {}
    for model_name in config.models:
        cv_scores: NDArray[np.float64] = np.array([], dtype=float)
        if config.cross_validation.enabled:
            cv_pipeline = _build_pipeline(schema, model_name, config.random_state)
            kfold = KFold(
                n_splits=config.cross_validation.n_splits,
                shuffle=True,
                random_state=config.random_state,
            )
            raw_scores = cross_val_score(
                cv_pipeline, x_train, y_train, cv=kfold, scoring="neg_mean_absolute_error"
            )
            cv_scores = -np.asarray(raw_scores, dtype=float)

        pipeline = _build_pipeline(schema, model_name, config.random_state)
        pipeline.fit(x_train, y_train)

        y_train_pred: NDArray[np.float64] = np.asarray(pipeline.predict(x_train), dtype=float)
        y_val_pred: NDArray[np.float64] = np.asarray(pipeline.predict(x_val), dtype=float)
        y_test_pred: NDArray[np.float64] = np.asarray(pipeline.predict(x_test), dtype=float)

        confidence_level = config.uncertainty.confidence_level
        prediction_interval: PredictionInterval | None = None
        bootstrap_interval: BootstrapPredictionInterval | None = None
        ensemble_pipelines: list[Pipeline] | None = None
        ensemble_disagreement_threshold: float | None = None
        n_bootstrap_estimators: int | None = None

        if config.uncertainty.method == "bootstrap":
            n_bootstrap_estimators = config.uncertainty.n_bootstrap_estimators

            def _build_this_pipeline(
                schema: DatasetSchema = schema,
                model_name: str = model_name,
                random_state: int = config.random_state,
            ) -> Pipeline:
                return _build_pipeline(schema, model_name, random_state)

            ensemble_pipelines = fit_bootstrap_ensemble(
                _build_this_pipeline,
                x_train,
                y_train,
                n_estimators=n_bootstrap_estimators,
                random_state=config.random_state,
            )
            val_ensemble_preds = ensemble_predict(ensemble_pipelines, x_val)
            test_ensemble_preds = ensemble_predict(ensemble_pipelines, x_test)

            bootstrap_interval = BootstrapPredictionInterval(
                confidence_level=confidence_level, n_estimators=n_bootstrap_estimators
            )
            y_test_lower, y_test_upper = bootstrap_interval.bounds(test_ensemble_preds)
            within = (y_test >= y_test_lower) & (y_test <= y_test_upper)
            coverage = float(np.mean(within))

            val_disagreement = ensemble_disagreement(val_ensemble_preds)
            ensemble_disagreement_threshold = float(
                np.quantile(val_disagreement, config.ood.ensemble_disagreement_percentile)
            )
        else:
            prediction_interval = fit_prediction_interval(
                y_val, y_val_pred, confidence_level=confidence_level
            )
            y_test_lower, y_test_upper = prediction_interval.bounds(y_test_pred)
            coverage = empirical_coverage(prediction_interval, y_test, y_test_pred)

        card = build_model_card(
            model_name=model_name,
            fitted_estimator=pipeline.named_steps["model"],
            schema=schema,
            x_train=x_train,
            y_train=y_train,
            validation_rows=len(x_val),
            test_rows=len(x_test),
            metrics_test=compute_metrics(y_test, y_test_pred),
            cv_mae_scores=cv_scores,
            uncertainty_method=config.uncertainty.method,
            confidence_level=confidence_level,
            interval_coverage_test=coverage,
            n_bootstrap_estimators=n_bootstrap_estimators,
        )

        trained[model_name] = TrainedModel(
            name=model_name,
            pipeline=pipeline,
            metrics_train=compute_metrics(y_train, y_train_pred),
            metrics_validation=compute_metrics(y_val, y_val_pred),
            metrics_test=compute_metrics(y_test, y_test_pred),
            cv_mae_scores=cv_scores,
            y_test_true=y_test,
            y_test_pred=y_test_pred,
            uncertainty_method=config.uncertainty.method,
            confidence_level=confidence_level,
            y_test_lower=y_test_lower,
            y_test_upper=y_test_upper,
            interval_coverage_test=coverage,
            error_by_region=error_by_region(
                region_values_test,
                y_test,
                y_test_pred,
                region_name,
                n_bins=config.error_analysis.n_bins,
            ),
            model_card=card,
            prediction_interval=prediction_interval,
            bootstrap_interval=bootstrap_interval,
            ensemble_pipelines=ensemble_pipelines,
            ensemble_disagreement_threshold=ensemble_disagreement_threshold,
        )

    return ExperimentResult(schema=schema, split=split, models=trained, ood_detector=ood_detector)
