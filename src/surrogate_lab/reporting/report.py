"""Self-contained HTML report generation.

The report embeds every plot as a base64 PNG and has no linked CSS/JS or
external file dependencies, so it can be opened directly from disk or
attached to an email/issue as a single file.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

from surrogate_lab.core.training import ExperimentResult, TrainedModel
from surrogate_lab.reporting.plots import parity_plot, residual_histogram, residual_plot

_UNCERTAINTY_NOTE = """
<section class="note">
  <h2>About the uncertainty estimate</h2>
  <p>Prediction intervals here are <strong>residual-based, not full Bayesian or ensemble
  uncertainty</strong>. Each model gets a single, global, symmetric half-width computed from the
  distribution of absolute residuals on the validation set, at the configured confidence level.
  The same half-width is applied everywhere in feature space.</p>
  <p><strong>Limitations:</strong> this assumes residual magnitude is roughly constant across the
  input domain (homoscedastic) and that the validation set represents the inputs you will predict
  on later. It will under-cover where the model is locally worse (e.g. extrapolation beyond the
  training domain) and over-cover where it is locally better. It does not detect out-of-distribution
  inputs on its own. Treat the coverage numbers above as a rough, global calibration check, not a
  per-prediction guarantee.</p>
</section>
"""

_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem auto;
       max-width: 960px; color: #1a1a1a; }
h1 { margin-bottom: 0; }
.meta { color: #666; margin-top: 0.25rem; }
table.metrics { border-collapse: collapse; width: 100%; margin: 1rem 0; }
table.metrics th, table.metrics td { border: 1px solid #ccc; padding: 0.4rem 0.6rem;
       text-align: right; }
table.metrics th:first-child, table.metrics td:first-child { text-align: left; }
section.model { border-top: 1px solid #ddd; padding-top: 1rem; margin-top: 1.5rem; }
div.plots { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; }
figure img { max-width: 300px; display: block; }
figcaption { text-align: center; font-size: 0.85rem; color: #555; }
section.note { background: #f7f7f7; padding: 1rem; border-radius: 6px; margin-top: 2rem; }
p.uncertainty, p.cv { font-size: 0.9rem; color: #333; }
"""


def _metrics_table_html(models: dict[str, TrainedModel]) -> str:
    rows = []
    for name, trained in models.items():
        metrics = trained.metrics_test
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics.mae:.4g}</td>"
            f"<td>{metrics.rmse:.4g}</td>"
            f"<td>{metrics.r2:.4g}</td>"
            f"<td>{metrics.mape:.4g}%</td>"
            f"<td>{trained.interval_coverage_test * 100:.1f}%</td>"
            "</tr>"
        )
    return (
        '<table class="metrics"><thead><tr>'
        "<th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE</th>"
        "<th>Interval coverage (test)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _model_section_html(name: str, trained: TrainedModel) -> str:
    residuals = trained.y_test_true - trained.y_test_pred
    parity_png = parity_plot(
        trained.y_test_true,
        trained.y_test_pred,
        f"{name}: parity (test)",
        trained.prediction_interval.half_width,
    )
    residual_png = residual_plot(
        trained.y_test_true, trained.y_test_pred, f"{name}: residuals (test)"
    )
    hist_png = residual_histogram(residuals, f"{name}: residual distribution (test)")

    cv_html = ""
    if trained.cv_mae_scores.size:
        scores = ", ".join(f"{score:.4g}" for score in trained.cv_mae_scores)
        cv_html = (
            f'<p class="cv">Cross-validated MAE per fold: {scores} '
            f"(mean {trained.cv_mae_scores.mean():.4g})</p>"
        )

    safe_name = html.escape(name)
    return f"""
    <section class="model">
      <h2>{safe_name}</h2>
      {cv_html}
      <div class="plots">
        <figure><img src="data:image/png;base64,{parity_png}" alt="{safe_name} parity plot">
          <figcaption>Parity</figcaption></figure>
        <figure><img src="data:image/png;base64,{residual_png}" alt="{safe_name} residual plot">
          <figcaption>Residuals</figcaption></figure>
        <figure><img src="data:image/png;base64,{hist_png}" alt="{safe_name} residual histogram">
          <figcaption>Residual distribution</figcaption></figure>
      </div>
      <p class="uncertainty">
        Residual-based {trained.prediction_interval.confidence_level * 100:.0f}% prediction
        interval half-width: {trained.prediction_interval.half_width:.4g}. Empirical coverage on
        the held-out test set: {trained.interval_coverage_test * 100:.1f}%. See "About the
        uncertainty estimate" below for what this does and does not mean.
      </p>
    </section>
    """


def generate_report(result: ExperimentResult, output_path: Path | str) -> Path:
    """Render a single self-contained HTML report and write it to ``output_path``.

    Contains, per trained model: a parity plot with the prediction-interval
    band, a residual plot, and a residual histogram, plus a metrics
    comparison table across all models and a note on what the uncertainty
    estimate does and does not mean.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections = "\n".join(
        _model_section_html(name, trained) for name, trained in result.models.items()
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>surrogate-lab report</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>surrogate-lab training report</h1>
<p class="meta">Generated {generated_at} &middot; {len(result.models)} model(s) &middot;
target: {html.escape(result.schema.target)}</p>
<h2>Metrics comparison (test set)</h2>
{_metrics_table_html(result.models)}
{sections}
{_UNCERTAINTY_NOTE}
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
