"""Self-contained HTML report generation.

The report embeds every plot as a base64 PNG and has no linked CSS/JS or
external file dependencies, so it can be opened directly from disk or
attached to an email/issue as a single file.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from surrogate_lab.core.error_analysis import RegionErrorTable
from surrogate_lab.core.model_card import ModelCard
from surrogate_lab.core.ood import OODDetector
from surrogate_lab.core.training import ExperimentResult, TrainedModel
from surrogate_lab.reporting.plots import (
    error_by_region_chart,
    parity_plot,
    residual_histogram,
    residual_plot,
)

_RESIDUAL_UNCERTAINTY_NOTE = """
<section class="note">
  <h2>About the uncertainty estimate: residual-based</h2>
  <p>Prediction intervals here are <strong>residual-based, not ensemble or Bayesian
  uncertainty</strong>. Each model gets a single, global, symmetric half-width computed from the
  distribution of absolute residuals on the validation set, at the configured confidence level.
  The same half-width is applied everywhere in feature space.</p>
  <p><strong>Limitations:</strong> this assumes residual magnitude is roughly constant across the
  input domain (homoscedastic) and that the validation set represents the inputs you will predict
  on later. It will under-cover where the model is locally worse (e.g. extrapolation beyond the
  training domain) and over-cover where it is locally better. It does not detect out-of-distribution
  inputs on its own &mdash; see the out-of-distribution section below for that. Treat the coverage
  numbers above as a rough, global calibration check, not a per-prediction guarantee.</p>
</section>
"""

_BOOTSTRAP_UNCERTAINTY_NOTE = """
<section class="note">
  <h2>About the uncertainty estimate: bootstrap ensemble</h2>
  <p>Prediction intervals here come from an <strong>ensemble of models trained on bootstrap
  resamples</strong> of the training data (bagging). At each point, the interval is the spread of
  the ensemble members' predictions at the configured confidence level &mdash; a genuinely
  per-point, heteroscedastic estimate: it can widen where the ensemble disagrees and narrow where
  it agrees.</p>
  <p><strong>Limitations:</strong> this costs N times the training time of a single fit (N =
  the number of ensemble members). It does not capture model-form uncertainty &mdash; if the
  chosen model type is a poor fit for the underlying relationship, every ensemble member can
  still agree with each other while being collectively wrong, producing a falsely narrow
  interval. High ensemble disagreement is itself used as one signal in the out-of-distribution
  assessment below, but low disagreement is not proof an input is well-covered by training data.</p>
</section>
"""

_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 2rem auto;
       max-width: 960px; color: #1a1a1a; }
h1 { margin-bottom: 0; }
.meta { color: #666; margin-top: 0.25rem; }
table.metrics, table.card, table.region { border-collapse: collapse; width: 100%; margin: 1rem 0; }
table.metrics th, table.metrics td, table.card th, table.card td, table.region th,
table.region td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }
table.metrics th:first-child, table.metrics td:first-child, table.card th:first-child,
table.card td:first-child, table.region th:first-child, table.region td:first-child {
       text-align: left; }
section.model { border-top: 1px solid #ddd; padding-top: 1rem; margin-top: 1.5rem; }
div.plots { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; }
figure img { max-width: 300px; display: block; }
figcaption { text-align: center; font-size: 0.85rem; color: #555; }
section.note { background: #f7f7f7; padding: 1rem; border-radius: 6px; margin-top: 2rem; }
p.uncertainty, p.cv { font-size: 0.9rem; color: #333; }
details.card { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
       padding: 0.75rem 1rem; margin-top: 1rem; }
details.card summary { font-weight: 600; cursor: pointer; }
ul.limits { font-size: 0.9rem; margin: 0.5rem 0; }
"""


def _metrics_table_html(models: dict[str, TrainedModel]) -> str:
    rows = []
    for name, trained in models.items():
        metrics = trained.metrics_test
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(trained.uncertainty_method)}</td>"
            f"<td>{metrics.mae:.4g}</td>"
            f"<td>{metrics.rmse:.4g}</td>"
            f"<td>{metrics.r2:.4g}</td>"
            f"<td>{metrics.mape:.4g}%</td>"
            f"<td>{trained.interval_coverage_test * 100:.1f}%</td>"
            "</tr>"
        )
    return (
        '<table class="metrics"><thead><tr>'
        "<th>Model</th><th>Uncertainty method</th><th>MAE</th><th>RMSE</th><th>R²</th>"
        "<th>MAPE</th><th>Interval coverage (test)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _model_card_html(card: ModelCard) -> str:
    numeric_ranges = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{lo:.4g} to {hi:.4g}</td></tr>"
        for name, (lo, hi) in card.numeric_feature_ranges.items()
    )
    categorical_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(', '.join(values))}</td></tr>"
        for name, values in card.categorical_feature_values.items()
    )
    hyperparam_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in card.hyperparameters.items()
    )
    cv_line = (
        f"<p>Cross-validated MAE (mean): {card.cv_mae_mean:.4g}</p>"
        if card.cv_mae_mean is not None
        else "<p>Cross-validation was disabled for this run.</p>"
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in card.limitations)
    intended = "".join(f"<li>{html.escape(item)}</li>" for item in card.intended_use)
    out_of_scope = "".join(f"<li>{html.escape(item)}</li>" for item in card.out_of_scope_use)

    return f"""
    <details class="card">
      <summary>Model card: {html.escape(card.model_name)}</summary>
      <h3>Training data</h3>
      <p>{card.training_rows} training rows, {card.validation_rows} validation rows,
      {card.test_rows} test rows. Target <code>{html.escape(card.target_name)}</code> range:
      {card.target_range[0]:.4g} to {card.target_range[1]:.4g}.</p>
      <table class="card"><tbody>{numeric_ranges}{categorical_rows}</tbody></table>
      <h3>Model</h3>
      <p>Type: <code>{html.escape(card.model_type)}</code></p>
      <table class="card"><tbody>{hyperparam_rows}</tbody></table>
      <h3>Validation metrics</h3>
      <p>Test MAE {card.metrics_test.mae:.4g}, RMSE {card.metrics_test.rmse:.4g}, R²
      {card.metrics_test.r2:.4g}, MAPE {card.metrics_test.mape:.4g}%.</p>
      {cv_line}
      <p>Uncertainty method: <code>{html.escape(card.uncertainty_method)}</code>,
      {card.confidence_level * 100:.0f}% target coverage, {card.interval_coverage_test * 100:.1f}%
      empirical test coverage.</p>
      <h3>Known limitations</h3>
      <ul class="limits">{limitations}</ul>
      <h3>Intended use</h3>
      <ul class="limits">{intended}</ul>
      <h3>Out-of-scope use</h3>
      <ul class="limits">{out_of_scope}</ul>
    </details>
    """


def _error_by_region_html(table: RegionErrorTable, model_name: str) -> str:
    if not table.bins:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(b.label)}</td><td>{b.count}</td><td>{b.mae:.4g}</td>"
        f"<td>{b.rmse:.4g}</td></tr>"
        for b in table.bins
    )
    chart_png = error_by_region_chart(
        [b.label for b in table.bins],
        np.array([b.mae for b in table.bins]),
        np.array([b.count for b in table.bins]),
        table.region_name,
        f"{model_name}: error by {table.region_name} (test)",
    )
    if table.min_mae > 0:
        ratio = table.max_mae / table.min_mae
        spread_note = (
            f"<p>Highest-error bin's MAE is {ratio:.2g}x the lowest-error bin's MAE on this "
            "test set.</p>"
        )
    elif table.max_mae > 0:
        # min_mae == 0 (a bin with perfect predictions): a ratio is undefined, but the gap is
        # still worth naming explicitly rather than silently omitting the note.
        spread_note = (
            f"<p>The lowest-error bin has ~zero error (perfect predictions on this test set), "
            f"while the highest-error bin's MAE is {table.max_mae:.4g} &mdash; error is highly "
            "concentrated in specific regions.</p>"
        )
    else:
        spread_note = ""
    return f"""
    <h3>Error by {html.escape(table.region_name)} (test set)</h3>
    <p>Test-set MAE/RMSE broken down by quantile bin of {html.escape(table.region_name)}, to
    show whether error is uniform or concentrated in one region. Bin counts are small on a
    typical test split, so treat differences between bins as suggestive, not conclusive.</p>
    <img src="data:image/png;base64,{chart_png}" alt="Error by {html.escape(table.region_name)}"
         style="max-width: 500px;">
    <table class="region"><thead><tr><th>{html.escape(table.region_name)} range</th>
    <th>n</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>{rows}</tbody></table>
    {spread_note}
    """


def _model_section_html(name: str, trained: TrainedModel) -> str:
    residuals = trained.y_test_true - trained.y_test_pred
    interval_arg: float | np.ndarray
    if trained.bootstrap_interval is not None:
        half_widths = (trained.y_test_upper - trained.y_test_lower) / 2.0
        interval_arg = half_widths
        half_width_summary = (
            f"mean {half_widths.mean():.4g} (min {half_widths.min():.4g}, "
            f"max {half_widths.max():.4g})"
        )
        method_label = f"bootstrap ensemble ({trained.bootstrap_interval.n_estimators} members)"
    else:
        assert trained.prediction_interval is not None
        interval_arg = trained.prediction_interval.half_width
        half_width_summary = f"{trained.prediction_interval.half_width:.4g}"
        method_label = "residual-based"

    parity_png = parity_plot(
        trained.y_test_true,
        trained.y_test_pred,
        f"{name}: parity (test)",
        interval_arg,
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
        {method_label} {trained.confidence_level * 100:.0f}% prediction interval half-width:
        {half_width_summary}. Empirical coverage on the held-out test set:
        {trained.interval_coverage_test * 100:.1f}%. See the uncertainty-method note below for
        what this does and does not mean.
      </p>
      {_error_by_region_html(trained.error_by_region, name)}
      {_model_card_html(trained.model_card)}
    </section>
    """


def _ood_summary_html(detector: OODDetector) -> str:
    if detector.insufficient_data_warning is not None:
        threshold_line = f"<p>{html.escape(detector.insufficient_data_warning)}</p>"
    elif detector.distance_metric == "mahalanobis" and detector.mahalanobis_threshold is not None:
        threshold_line = (
            f"<p>Mahalanobis nearest-neighbour distance threshold (training-set "
            f"{detector.nn_distance_percentile:.0%} percentile): "
            f"{detector.mahalanobis_threshold:.4g}.</p>"
        )
    elif detector.nn_distance_threshold is not None:
        threshold_line = (
            f"<p>Euclidean nearest-neighbour distance threshold (training-set "
            f"{detector.nn_distance_percentile:.0%} percentile): "
            f"{detector.nn_distance_threshold:.4g} (standardized feature space).</p>"
        )
    else:
        threshold_line = ""

    ranges = "".join(
        f"<li>{html.escape(name)}: {detector.feature_min[name]:.4g} to "
        f"{detector.feature_max[name]:.4g}</li>"
        for name in detector.numeric_features
    )
    categorical = "".join(
        f"<li>{html.escape(name)}: {html.escape(', '.join(sorted(values)))}</li>"
        for name, values in detector.categorical_seen.items()
    )

    return f"""
<section class="note">
  <h2>Out-of-distribution detection</h2>
  <p>These are the feature ranges, categories, and nearest-neighbour distance threshold
  learned from the training data. <code>surrogate-lab predict</code> checks each new input
  against these by default (<code>--no-ood</code> to skip it) and reports whether it falls
  outside the training domain &mdash; see the README's "Out-of-distribution detection" section
  for the full method and its limitations. A "within training domain" result means no
  configured check raised a flag, not that the prediction is verified reliable: these checks
  catch some, not all, forms of extrapolation.</p>
  <ul>{ranges}{categorical}</ul>
  {threshold_line}
</section>
"""


def generate_report(result: ExperimentResult, output_path: Path | str) -> Path:
    """Render a single self-contained HTML report and write it to ``output_path``.

    Contains, per trained model: a parity plot with the prediction-interval
    visualization, a residual plot, a residual histogram, an error-by-region
    breakdown, and an auto-generated model card, plus a metrics comparison
    table across all models and notes on what the uncertainty estimate and
    out-of-distribution checks do and do not mean.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = "\n".join(
        _model_section_html(name, trained) for name, trained in result.models.items()
    )

    methods_used = {trained.uncertainty_method for trained in result.models.values()}
    uncertainty_notes = ""
    if "residual" in methods_used:
        uncertainty_notes += _RESIDUAL_UNCERTAINTY_NOTE
    if "bootstrap" in methods_used:
        uncertainty_notes += _BOOTSTRAP_UNCERTAINTY_NOTE

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
{uncertainty_notes}
{_ood_summary_html(result.ood_detector)}
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
