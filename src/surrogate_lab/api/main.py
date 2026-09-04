"""FastAPI application: a report-only web demo for surrogate-lab.

Scope (deliberate, see project README / task notes for the reasoning):

* ``GET /`` serves a small dependency-free HTML demo page.
* ``GET /demo`` serves a *pre-trained* report baked into the package at
  build time (``static/demo_report.html``) — no training happens at
  request time, so there is no timeout risk.
* ``POST /train`` lets a visitor upload a small CSV and get back a real,
  freshly generated HTML report, but trains *only* ``linear_regression``
  (the cheapest model in the registry) and enforces hard caps on upload
  size and row count, keeping the request well under Vercel's serverless
  function timeout.
* ``GET /health`` is a trivial liveness/version check.

``random_forest`` and ``gradient_boosting`` are intentionally never
reachable through this API — they remain CLI-only (``surrogate-lab train``).
"""

from __future__ import annotations

import html
import io
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from surrogate_lab import __version__
from surrogate_lab.core.config import (
    CrossValidationConfig,
    DataConfig,
    ExperimentConfig,
    SplitConfig,
    UncertaintyConfig,
)
from surrogate_lab.core.data import DataError
from surrogate_lab.core.schema import DatasetSchema
from surrogate_lab.core.training import train_and_evaluate
from surrogate_lab.reporting.report import generate_report

app = FastAPI(title="surrogate-lab demo API")

_STATIC_DIR = Path(__file__).parent / "static"
_DEMO_REPORT_PATH = _STATIC_DIR / "demo_report.html"

# Hard caps on what /train will accept, chosen to keep a live linear-regression
# training run (schema validation -> split -> preprocess -> fit -> CV -> report)
# comfortably under Vercel's 10s Hobby-plan serverless function timeout even
# accounting for cold starts.
MAX_UPLOAD_BYTES = 1_000_000  # ~1 MB
MAX_ROWS = 5_000
MIN_ROWS = 20  # enough rows to leave a sane train/val/test split and 3-fold CV
CV_N_SPLITS = 3
# Kept low (vs. the CLI default of 30) to bound live-request latency: bootstrap trains this
# many linear_regression fits in addition to the primary one.
BOOTSTRAP_N_ESTIMATORS = 15

_ONLY_MODEL = "linear_regression"

_PAGE_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem auto;
       max-width: 720px; color: #1a1a1a; line-height: 1.5; padding: 0 1rem; }
h1 { margin-bottom: 0.25rem; }
p.tagline { color: #555; margin-top: 0; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem 1.5rem; margin: 1.5rem 0; }
.card h2 { margin-top: 0; }
a.button, button { display: inline-block; background: #1a56db; color: #fff; border: none;
       padding: 0.6rem 1.2rem; border-radius: 6px; font-size: 1rem; text-decoration: none;
       cursor: pointer; }
a.button:hover, button:hover { background: #1443ad; }
label { display: block; font-weight: 600; margin-top: 0.9rem; margin-bottom: 0.25rem; }
input[type=text], input[type=file] { width: 100%; box-sizing: border-box; padding: 0.5rem;
       border: 1px solid #ccc; border-radius: 4px; font-size: 0.95rem; }
.hint { color: #666; font-size: 0.85rem; margin-top: 0.2rem; }
.limits { background: #f7f7f7; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.85rem;
       color: #444; margin-top: 1rem; }
footer { color: #888; font-size: 0.8rem; margin-top: 3rem; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _error_page(message: str, *, detail: str | None = None) -> HTMLResponse:
    body_detail = f"<p>{html.escape(detail)}</p>" if detail else ""
    body = f"""
<h1>Could not train a model</h1>
<div class="card">
  <p><strong>{html.escape(message)}</strong></p>
  {body_detail}
  <p><a href="/">&larr; Back to the demo</a></p>
</div>
"""
    return HTMLResponse(content=_page("surrogate-lab: error", body), status_code=400)


def _index_html() -> str:
    body = f"""
<h1>surrogate-lab</h1>
<p class="tagline">Train, compare, and validate surrogate models &mdash; web demo.</p>

<div class="card">
  <h2>1. See a pre-trained example report</h2>
  <p>A cantilever-beam deflection surrogate, trained offline with all three registered
  models (linear regression, random forest, gradient boosting) and cross-validated.</p>
  <a class="button" href="/demo">View the beam_deflection example report</a>
</div>

<div class="card">
  <h2>2. Train on your own CSV (linear regression only)</h2>
  <p>Upload a small CSV, tell us which columns are features and which is the target, and
  we'll train a fresh <strong>linear regression</strong> model and hand back a full HTML
  report &mdash; live, in this request.</p>
  <form method="post" action="/train" enctype="multipart/form-data">
    <label for="file">CSV file</label>
    <input type="file" id="file" name="file" accept=".csv,text/csv" required>

    <label for="features">Feature columns</label>
    <input type="text" id="features" name="features" placeholder="length_m, load_n, material"
           required>
    <p class="hint">Comma-separated column names. Non-numeric columns are treated as
    categorical automatically.</p>

    <label for="target">Target column</label>
    <input type="text" id="target" name="target" placeholder="deflection_m" required>
    <p class="hint">Must be a numeric column.</p>

    <label for="uncertainty_method">Uncertainty method</label>
    <select id="uncertainty_method" name="uncertainty_method">
      <option value="residual" selected>Residual-based (single global interval)</option>
      <option value="bootstrap">Bootstrap ensemble ({BOOTSTRAP_N_ESTIMATORS} members,
        per-point interval, slower)</option>
    </select>
    <p class="hint">See the report's uncertainty note for what each method does and does not
    guarantee.</p>

    <p style="margin-top: 1.25rem;"><button type="submit">Train &amp; view report</button></p>
  </form>
  <div class="limits">
    Uses <strong>linear regression only</strong> (the fastest model available) to stay
    within serverless request time limits. Datasets are capped at
    {MAX_ROWS:,} rows / {MAX_UPLOAD_BYTES // 1_000_000} MB, and need at least
    {MIN_ROWS} rows. Random forest and gradient boosting are available via the
    <code>surrogate-lab</code> command-line tool, not this API.
  </div>
</div>

<footer>surrogate-lab v{html.escape(__version__)} &middot;
<a href="https://github.com/NicholasTJL/surrogate-lab">source on GitHub</a></footer>
"""
    return _page("surrogate-lab demo", body)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Demo landing page: link to the pre-baked report + CSV upload form."""
    return HTMLResponse(content=_index_html())


@app.get("/health")
def health() -> dict[str, str]:
    """Trivial liveness and version check."""
    return {"status": "ok", "version": __version__}


@app.get("/demo", response_class=HTMLResponse)
def demo() -> HTMLResponse:
    """Serve the pre-trained beam_deflection example report, baked into the package."""
    if not _DEMO_REPORT_PATH.exists():
        return _error_page(
            "The pre-baked demo report is missing from this deployment.",
            detail=f"Expected to find it at {_DEMO_REPORT_PATH}.",
        )
    content = _DEMO_REPORT_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


def _parse_columns(raw: str, kind: str) -> list[str]:
    columns = [name.strip() for name in raw.split(",") if name.strip()]
    if not columns:
        raise ValueError(f"no {kind} column names given")
    return columns


def _build_schema(df: pd.DataFrame, features: list[str], target: str) -> DatasetSchema:
    """Build a :class:`DatasetSchema`, auto-detecting categorical features.

    A feature is treated as categorical if its column is not a numeric pandas
    dtype; everything else is treated as numeric. This mirrors what a user
    filling in only "feature names" + "target name" (no separate categorical
    picker, by design) would expect.
    """
    categorical = [
        name
        for name in features
        if name in df.columns and not pd.api.types.is_numeric_dtype(df[name])
    ]
    return DatasetSchema(features=features, target=target, categorical_features=categorical)


def _build_config(schema: DatasetSchema, uncertainty_method: str) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(path="<uploaded>", format="csv"),
        dataset_schema=schema,
        models=[_ONLY_MODEL],
        split=SplitConfig(test_size=0.2, val_size=0.15, random_state=42),
        cross_validation=CrossValidationConfig(enabled=True, n_splits=CV_N_SPLITS),
        uncertainty=UncertaintyConfig(
            method="bootstrap" if uncertainty_method == "bootstrap" else "residual",
            confidence_level=0.9,
            n_bootstrap_estimators=BOOTSTRAP_N_ESTIMATORS,
        ),
        output_dir="<unused>",
        random_state=42,
    )


@app.post("/train", response_class=HTMLResponse)
def train(
    file: UploadFile,
    features: str = Form(...),
    target: str = Form(...),
    uncertainty_method: str = Form("residual"),
) -> HTMLResponse:
    """Train a linear-regression model on an uploaded CSV and return the HTML report.

    Only ``linear_regression`` is ever used here, deliberately: random forest and
    gradient boosting are not exposed through this API to keep requests within
    Vercel's serverless function time limit. ``uncertainty_method`` may be
    ``"residual"`` (default) or ``"bootstrap"``; bootstrap trains
    ``BOOTSTRAP_N_ESTIMATORS`` additional linear regressions to keep request
    latency bounded, a lower count than the CLI's default of 30.
    """
    raw_bytes = file.file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // 1_000_000
        return _error_page(
            f"File is too large ({len(raw_bytes):,} bytes).",
            detail=f"The limit is {MAX_UPLOAD_BYTES:,} bytes (~{limit_mb} MB).",
        )
    if not raw_bytes:
        return _error_page("The uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean error page
        return _error_page("Could not parse the uploaded file as CSV.", detail=str(exc))

    if len(df) > MAX_ROWS:
        return _error_page(
            f"Dataset has too many rows ({len(df):,}).",
            detail=f"The limit is {MAX_ROWS:,} rows for the live-training demo.",
        )
    if len(df) < MIN_ROWS:
        return _error_page(
            f"Dataset has too few rows ({len(df)}).",
            detail=f"At least {MIN_ROWS} rows are needed for a meaningful train/validation/"
            "test split and cross-validation.",
        )

    try:
        feature_names = _parse_columns(features, "feature")
        target_name = target.strip()
        if not target_name:
            raise ValueError("no target column name given")
    except ValueError as exc:
        return _error_page("Invalid column specification.", detail=str(exc))

    missing = [name for name in [*feature_names, target_name] if name not in df.columns]
    if missing:
        return _error_page(
            "Some requested columns are not in the uploaded CSV.",
            detail=f"Missing: {', '.join(missing)}. Available columns: {', '.join(df.columns)}.",
        )

    try:
        schema = _build_schema(df, feature_names, target_name)
    except ValidationError as exc:
        return _error_page("Invalid feature/target specification.", detail=str(exc))

    config = _build_config(schema, uncertainty_method)

    try:
        result = train_and_evaluate(df, config)
    except DataError as exc:
        return _error_page("Dataset does not match the declared schema.", detail=str(exc))
    except ValueError as exc:
        # Most commonly: the target column (or a numeric feature) contains
        # non-numeric values that couldn't be coerced to float.
        return _error_page(
            "Could not train on this data.",
            detail=f"{exc} (is the target column, and every numeric feature column, "
            "actually numeric?)",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = generate_report(result, Path(tmpdir) / "report.html")
        report_html = report_path.read_text(encoding="utf-8")

    return HTMLResponse(content=report_html)
