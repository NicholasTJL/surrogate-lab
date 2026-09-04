# Product Vision and Non-Goals

## Problem

Expensive simulations and experiments (CFD sweeps, FEA cases, physical lab tests) are too slow
or costly to run exhaustively across a design space. A surrogate model — a cheap statistical
stand-in trained on a sample of real runs — lets you explore the rest of the space quickly. But a
surrogate that reports a prediction without any sense of how much to trust it is dangerous: it
will extrapolate into regions it has never seen with the same apparent confidence as it has in
the middle of its training data. surrogate-lab is a reusable toolkit for training, comparing,
validating, exporting, and serving surrogate models, built around reliability rather than only
predictive accuracy.

## Who it's for

Engineers and researchers who need a fast, repeatable approximation of an expensive simulation or
experiment, and who currently cobble this together with ad hoc notebooks that don't track how
the model was validated, how confident it is, or whether it's being asked to extrapolate.

## What makes it different

surrogate-lab treats reliability as a first-class output, not an afterthought: every training
run produces an explicit metrics comparison, residual diagnostics, and an honestly-labeled
uncertainty estimate with its assumptions stated, not just a leaderboard of error scores. It is
built on scikit-learn-compatible tabular regression — general enough for many simulation domains,
without trying to be a full AutoML platform.

## v0.1.0 scope

- CSV and Parquet ingestion.
- User-defined feature/target schema (pydantic), validated against the dataset.
- Reproducible train/validation/test split with a fixed random seed.
- Scikit-learn preprocessing pipelines (`StandardScaler` + `OneHotEncoder` via
  `ColumnTransformer`).
- A model registry covering linear regression, random forest, and gradient boosting.
- Cross-validation support.
- MAE, RMSE, R², and MAPE metrics.
- Residual and parity plots.
- YAML-based experiment configuration.
- Reproducible random seeds throughout (split, models, cross-validation).
- Saved preprocessing + model artifacts (joblib) with metadata sidecars.
- A CLI: `surrogate-lab train`, `surrogate-lab predict`.
- A self-contained HTML report (parity/residual plots, metrics table, residual-based prediction
  intervals with a documented, honest description of what they do and do not provide).

## v0.1.0 non-goals

These are excluded from the first release to protect delivery speed and API quality — not ruled
out permanently:

- Polynomial regression, Gaussian-process regression, multilayer perceptrons, and PyTorch
  networks in the model registry (v0.1.0 is scikit-learn only, to keep dependencies light).
- Bootstrap ensembles or Gaussian-process uncertainty (v0.1.0 uses residual-based prediction
  intervals only — see the README's "Uncertainty" section for why, and what that trades off).
- Out-of-distribution / extrapolation detection (feature range checks, nearest-neighbour or
  Mahalanobis distance, ensemble-disagreement warnings).
- Model cards and automated error analysis by data region.
- A served inference API, Docker image, ONNX export, or batch prediction service.
- Hyperparameter tuning beyond scikit-learn defaults.

## v0.2.0: reliability release

Delivered:

- Bootstrap-ensemble prediction intervals (`uncertainty.method: bootstrap`), selectable in
  config alongside the existing residual-based method — a genuine alternative (per-point,
  heteroscedastic estimate at N times the training cost), not a replacement.
- Feature range checks, nearest-neighbour distance (Euclidean by default, with a Mahalanobis
  option), and ensemble-disagreement warnings for out-of-distribution detection, behind a
  structured pydantic `OODAssessment` response, surfaced in `surrogate-lab predict` and the
  HTML report.
- Model cards and error analysis by data region, auto-generated per trained model in the HTML
  report from the actual training run's numbers.

Deliberately deferred, not attempted this release:

- **Gaussian-process uncertainty.** Considered and dropped: it only makes mathematical sense
  paired with a Gaussian-process model, not applied to e.g. random forest or gradient boosting
  predictions. Doing it properly means adding `gaussian_process` as a fourth model type to the
  registry as well as a third uncertainty method — judged disproportionate new complexity next
  to bootstrap ensembles and out-of-distribution detection, which were more clearly in scope and
  well-defined. Candidate for a future release once there's a concrete need for a model type
  that supports it natively.

## v0.3.0: deployment release

- FastAPI inference service with batch prediction, health/readiness endpoints, and a model
  version endpoint.
- Docker image and ONNX export where supported.
- Request validation and prediction logging without sensitive input retention.
- A latency benchmark.

## Current status

v0.1.0's foundation (ingestion, schema validation, splitting, preprocessing, the model registry,
cross-validation, metrics, artifact persistence, the CLI, and HTML reporting) is in place, and
v0.2.0's reliability work is built on top of it: bootstrap-ensemble uncertainty alongside the
original residual-based method, out-of-distribution detection (feature range, nearest-neighbour
and Mahalanobis distance, ensemble disagreement), model cards, and error-by-region analysis.
Gaussian-process uncertainty was scoped for v0.2.0 and deliberately deferred (see above). A
served inference API, Docker image, and ONNX export remain the v0.3.0 milestone.

A small FastAPI demo (`src/surrogate_lab/api/`, `GET /demo`, `POST /train` for
linear-regression-only fits on small uploads) was added ahead of the v0.3.0 schedule. It works
correctly — verified locally end to end — but is **not deployed live**: the dependency bundle
(scikit-learn + pandas + matplotlib + scipy + pyarrow, ~478MB) sits right at Vercel's 500MB
standard Python bundle limit, which triggers an automatic "dependency optimization" step that
drops the project's own package from the deployed bundle
(`ModuleNotFoundError: No module named 'surrogate_lab'` at runtime, despite the build itself
reporting success). Neither `excludeFiles` bundle trimming nor enabling Fluid compute changed
the outcome — the larger-bundle allowance Vercel's docs mention for Fluid compute appears to be
a separate, non-self-serve beta. Run the API locally instead: see the root README's quickstart.
