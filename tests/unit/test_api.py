"""Tests for the report-only demo API (src/surrogate_lab/api/main.py).

Covers: health check, the pre-baked /demo report, a real live-trained
/train round trip (linear_regression only), and that bad/oversized
uploads come back as clean 4xx responses rather than a 500 or a hang.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from surrogate_lab import __version__
from surrogate_lab.api.main import MAX_ROWS, MAX_UPLOAD_BYTES, MIN_ROWS, app

client = TestClient(app)

_FIXTURE_CSV = Path("tests/fixtures/sample_data.csv")


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_index_links_to_demo_and_has_upload_form() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'href="/demo"' in response.text
    assert 'action="/train"' in response.text
    assert "linear regression" in response.text.lower()


def test_demo_returns_baked_report_content() -> None:
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The baked report is a real surrogate-lab report for the beam_deflection example.
    assert "surrogate-lab training report" in response.text
    assert "linear_regression" in response.text
    assert "random_forest" in response.text


def test_train_with_valid_small_csv_returns_real_report() -> None:
    csv_bytes = _FIXTURE_CSV.read_bytes()

    response = client.post(
        "/train",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={
            "features": "length_m, load_n, moment_of_inertia_m4, material",
            "target": "deflection_m",
        },
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "surrogate-lab training report" in response.text
    assert "linear_regression" in response.text
    # Only linear_regression should ever be trained via the API.
    assert "random_forest" not in response.text
    assert "gradient_boosting" not in response.text
    # A real metrics table, not a placeholder.
    assert "MAE" in response.text and "RMSE" in response.text


def test_train_rejects_missing_columns_cleanly() -> None:
    csv_bytes = _FIXTURE_CSV.read_bytes()

    response = client.post(
        "/train",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={"features": "length_m, not_a_real_column", "target": "deflection_m"},
    )

    assert response.status_code == 400
    assert "text/html" in response.headers["content-type"]
    assert "not_a_real_column" in response.text
    assert "<h1>Could not train a model</h1>" in response.text


def test_train_rejects_non_numeric_target_cleanly() -> None:
    csv_bytes = _FIXTURE_CSV.read_bytes()

    response = client.post(
        "/train",
        files={"file": ("sample_data.csv", csv_bytes, "text/csv")},
        data={"features": "length_m, load_n", "target": "material"},
    )

    assert response.status_code == 400
    assert "<h1>Could not train a model</h1>" in response.text


def test_train_rejects_too_few_rows() -> None:
    df = pd.read_csv(_FIXTURE_CSV).head(MIN_ROWS - 1)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    response = client.post(
        "/train",
        files={"file": ("tiny.csv", buffer.getvalue().encode(), "text/csv")},
        data={
            "features": "length_m, load_n, moment_of_inertia_m4, material",
            "target": "deflection_m",
        },
    )

    assert response.status_code == 400
    assert "too few rows" in response.text.lower()


def test_train_rejects_too_many_rows_without_hanging() -> None:
    base = pd.read_csv(_FIXTURE_CSV)
    # Duplicate rows well past MAX_ROWS; this must be rejected on the row-count
    # check before any training is attempted (and must not hang or 500).
    repeats = (MAX_ROWS // len(base)) + 2
    big = pd.concat([base] * repeats, ignore_index=True)
    buffer = io.StringIO()
    big.to_csv(buffer, index=False)
    assert len(big) > MAX_ROWS

    response = client.post(
        "/train",
        files={"file": ("big.csv", buffer.getvalue().encode(), "text/csv")},
        data={
            "features": "length_m, load_n, moment_of_inertia_m4, material",
            "target": "deflection_m",
        },
    )

    assert response.status_code == 400
    assert "too many rows" in response.text.lower()


def test_train_rejects_oversized_file_without_hanging() -> None:
    # Bytes-sized cap check should fire before pandas even parses the file.
    oversized = b"a,b\n" + b"1,2\n" * (MAX_UPLOAD_BYTES // 4 + 1000)
    assert len(oversized) > MAX_UPLOAD_BYTES

    response = client.post(
        "/train",
        files={"file": ("oversized.csv", oversized, "text/csv")},
        data={"features": "a", "target": "b"},
    )

    assert response.status_code == 400
    assert "too large" in response.text.lower()


def test_train_rejects_empty_file() -> None:
    response = client.post(
        "/train",
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"features": "a", "target": "b"},
    )

    assert response.status_code == 400


def test_train_rejects_garbage_that_is_not_csv() -> None:
    # Enough rows/bytes to pass the size gate, but not parseable as tabular data
    # with the requested columns.
    garbage = ("not,a,valid,header\n" + "x\n" * (MIN_ROWS + 5)).encode()

    response = client.post(
        "/train",
        files={"file": ("garbage.csv", garbage, "text/csv")},
        data={"features": "length_m", "target": "deflection_m"},
    )

    assert response.status_code == 400
