# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Known limitations

- Uncertainty quantification is residual-based only (a single global interval per model), not
  bootstrap ensembles or Gaussian-process variance — see the README's "Uncertainty" section.
- Model registry covers scikit-learn estimators only; polynomial regression, Gaussian-process
  regression, and PyTorch networks are deferred to later releases (see docs/vision.md).
- No out-of-distribution detection, model cards, or served inference API yet — planned for
  v0.2.0/v0.3.0.
