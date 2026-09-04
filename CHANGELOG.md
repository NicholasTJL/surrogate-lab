# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Bootstrap-ensemble prediction intervals (`uncertainty.method: bootstrap`), a genuine
  alternative to the existing residual-based method: trains N bootstrap-resampled copies of a
  model and uses the spread of their predictions as a per-point, heteroscedastic interval, at N
  times the training cost (`surrogate_lab.core.uncertainty`).
- Out-of-distribution / extrapolation detection (`surrogate_lab.core.ood`): feature range
  checks, Euclidean nearest-neighbour distance with a training-set-calibrated threshold, an
  optional Mahalanobis-distance metric, and ensemble-disagreement warnings when the bootstrap
  uncertainty method is used. Structured via a pydantic `OODAssessment` model, surfaced by
  `surrogate-lab predict` (new `--ood`/`--no-ood` flag, on by default) and in the HTML report.
- Auto-generated model cards per trained model in the HTML report: training data summary,
  model type and hyperparameters, validation metrics, and limitations/intended-use/out-of-scope
  sections assembled from the actual training run, not hand-written boilerplate
  (`surrogate_lab.core.model_card`).
- Error analysis by data region: test-set MAE/RMSE broken down by quantile bin of the target or
  a chosen feature (`error_analysis.region_feature` in config), shown as a table and chart in
  the report (`surrogate_lab.core.error_analysis`).
- New config sections: `uncertainty.method`/`n_bootstrap_estimators`, `ood.*`
  (`enabled`, `nn_distance_percentile`, `distance_metric`, `ensemble_disagreement_percentile`),
  `error_analysis.*` (`region_feature`, `n_bins`).
- The demo FastAPI app's `/train` endpoint now accepts an `uncertainty_method` field
  (`residual` or `bootstrap`), with a lower bootstrap member count than the CLI default to
  bound live-request latency.
- `examples/beam_deflection/config_bootstrap.yaml`: a variant of the example config
  demonstrating bootstrap uncertainty, Mahalanobis-distance OOD detection, and error-by-feature
  analysis.
- Project scaffold: package layout, CI, contribution templates.
- `DatasetSchema` (pydantic) for user-defined feature/target columns.
- CSV/Parquet ingestion and reproducible train/validation/test splitting
  (`surrogate_lab.core.data`).
- Scikit-learn preprocessing pipelines (`StandardScaler` + `OneHotEncoder` via
  `ColumnTransformer`).
- Model registry covering linear regression, random forest, and gradient boosting
  (`surrogate_lab.core.registry`).
- Cross-validation and MAE/RMSE/R²/MAPE metrics.
- Residual-based prediction intervals with empirical coverage reporting
  (`surrogate_lab.core.uncertainty`).
- Self-contained HTML report generation: parity plots, residual plots, residual
  histograms, and a metrics comparison table, all embedded as base64 PNGs
  (`surrogate_lab.reporting`).
- YAML experiment configuration (`surrogate_lab.core.config`).
- Joblib-based model + metadata artifact persistence (`surrogate_lab.core.artifacts`).
- CLI: `surrogate-lab train` and `surrogate-lab predict`.
- `examples/beam_deflection`: a synthetic cantilever-beam dataset and end-to-end example.

### Changed

- `surrogate_lab.core.artifacts.save_artifact`/`load_artifact` now also persist (and return) the
  bootstrap ensemble members and the fitted out-of-distribution detector alongside the primary
  pipeline and metadata. `load_artifact` now returns a 4-tuple
  (`pipeline, metadata, ensemble_pipelines, ood_detector`) instead of a 2-tuple; artifacts saved
  by v0.1.0 still load (the two new elements come back `None`), but any code calling
  `load_artifact` directly needs to be updated for the new return shape.
- `ModelMetadata` gained `uncertainty_method`, `n_bootstrap_estimators`, and
  `ensemble_disagreement_threshold` fields; `prediction_interval_half_width` is now `None` for
  bootstrap-method models (the ensemble file carries the interval information instead).

### Known limitations

- Gaussian-process uncertainty was considered for v0.2.0 and deliberately deferred: it only
  makes mathematical sense paired with a GP model (not bolted onto e.g. random forest
  predictions), which would mean adding a fourth model type to the registry as well — judged
  disproportionate scope for this release next to bootstrap ensembles and out-of-distribution
  detection. See docs/vision.md.
- Model registry still covers scikit-learn estimators only; polynomial regression and PyTorch
  networks remain deferred (see docs/vision.md).
- Out-of-distribution detection catches some, not all, forms of extrapolation: a point can pass
  every configured check (range, distance, ensemble disagreement) and still be a real
  extrapolation in a feature combination none of the checks happen to flag. See the README's
  "Out-of-distribution detection" section.
- No served inference API, Docker image, ONNX export, or batch prediction service yet — planned
  for v0.3.0.
