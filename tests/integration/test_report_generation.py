from pathlib import Path

from surrogate_lab.core.config import load_config
from surrogate_lab.core.data import load_dataset
from surrogate_lab.core.training import train_and_evaluate
from surrogate_lab.reporting.report import generate_report


def test_generate_report_produces_valid_html_with_expected_content(tmp_path: Path) -> None:
    config = load_config(Path("tests/fixtures/sample_config.yaml"))
    df = load_dataset(config.data.path, config.data.format)
    result = train_and_evaluate(df, config)

    report_path = generate_report(result, tmp_path / "report.html")

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    assert content.startswith("<!doctype html>")
    assert "<html" in content and "</html>" in content

    # Each model gets its own section with parity/residual/histogram plots plus an
    # error-by-region chart, all embedded as base64 PNGs.
    for name in result.models:
        assert name in content
    assert content.count("data:image/png;base64,") == len(result.models) * 4

    # Metrics table and uncertainty note are present.
    assert "Metrics comparison" in content
    assert "MAE" in content and "RMSE" in content and "MAPE" in content
    assert "residual-based" in content.lower() or "residual based" in content.lower()
    assert "About the uncertainty estimate" in content

    # Model card and error-by-region sections are present per model.
    assert "Model card:" in content
    assert "Known limitations" in content
    assert "Error by" in content

    # Out-of-distribution calibration summary is present.
    assert "Out-of-distribution detection" in content

    # No external file dependencies: no linked stylesheet/script/image files.
    assert "<link " not in content
    assert 'src="http' not in content
    assert "<script " not in content
