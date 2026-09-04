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
from surrogate_lab.core.metrics import Metrics, compute_metrics
from surrogate_lab.core.preprocessing import build_preprocessor
from surrogate_lab.core.registry import get_model
from surrogate_lab.core.schema import DatasetSchema
from surrogate_lab.core.uncertainty import (
    PredictionInterval,
    empirical_coverage,
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
    prediction_interval: PredictionInterval
    interval_coverage_test: float


@dataclass(frozen=True)
class ExperimentResult:
    """The outcome of training every model requested in an :class:`ExperimentConfig`."""

    schema: DatasetSchema
    split: DatasetSplit
    models: dict[str, TrainedModel]


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

        interval = fit_prediction_interval(
            y_val, y_val_pred, confidence_level=config.uncertainty.confidence_level
        )
        coverage = empirical_coverage(interval, y_test, y_test_pred)

        trained[model_name] = TrainedModel(
            name=model_name,
            pipeline=pipeline,
            metrics_train=compute_metrics(y_train, y_train_pred),
            metrics_validation=compute_metrics(y_val, y_val_pred),
            metrics_test=compute_metrics(y_test, y_test_pred),
            cv_mae_scores=cv_scores,
            y_test_true=y_test,
            y_test_pred=y_test_pred,
            prediction_interval=interval,
            interval_coverage_test=coverage,
        )

    return ExperimentResult(schema=schema, split=split, models=trained)
