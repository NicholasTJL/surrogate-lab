from pathlib import Path

import pandas as pd
import pytest

from surrogate_lab.core.data import DataError, load_dataset, split_dataset, validate_schema
from surrogate_lab.core.schema import DatasetSchema


def test_load_csv() -> None:
    df = load_dataset(Path("tests/fixtures/sample_data.csv"))

    assert len(df) == 120
    assert "deflection_m" in df.columns


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_dataset(tmp_path / "missing.csv")


def test_load_unknown_extension_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("a,b\n1,2\n")

    with pytest.raises(DataError, match="cannot infer format"):
        load_dataset(bad_file)


def test_load_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    loaded = load_dataset(parquet_path)

    pd.testing.assert_frame_equal(loaded, df)


def test_validate_schema_missing_column_raises() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    schema = DatasetSchema(features=["a", "b"], target="y")

    with pytest.raises(DataError, match="missing declared columns"):
        validate_schema(df, schema)


def test_validate_schema_missing_target_values_raises() -> None:
    df = pd.DataFrame({"a": [1, 2], "y": [1.0, None]})
    schema = DatasetSchema(features=["a"], target="y")

    with pytest.raises(DataError, match="contains missing values"):
        validate_schema(df, schema)


def test_validate_schema_passes_for_matching_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "y": [1.0, 2.0]})
    schema = DatasetSchema(features=["a"], target="y")

    validate_schema(df, schema)  # no exception


def test_split_is_reproducible_for_same_seed() -> None:
    df = load_dataset(Path("tests/fixtures/sample_data.csv"))

    split_a = split_dataset(df, test_size=0.2, val_size=0.1, random_state=42)
    split_b = split_dataset(df, test_size=0.2, val_size=0.1, random_state=42)

    pd.testing.assert_frame_equal(split_a.train, split_b.train)
    pd.testing.assert_frame_equal(split_a.validation, split_b.validation)
    pd.testing.assert_frame_equal(split_a.test, split_b.test)


def test_split_differs_for_different_seed() -> None:
    df = load_dataset(Path("tests/fixtures/sample_data.csv"))

    split_a = split_dataset(df, test_size=0.2, val_size=0.1, random_state=42)
    split_b = split_dataset(df, test_size=0.2, val_size=0.1, random_state=7)

    assert not split_a.train.equals(split_b.train)


def test_split_sizes_and_no_overlap() -> None:
    df = load_dataset(Path("tests/fixtures/sample_data.csv"))

    split = split_dataset(df, test_size=0.2, val_size=0.1, random_state=42)

    assert len(split.train) + len(split.validation) + len(split.test) == len(df)
    assert abs(len(split.test) / len(df) - 0.2) < 0.03
    assert abs(len(split.validation) / len(df) - 0.1) < 0.03


def test_split_rejects_sizes_summing_to_one_or_more() -> None:
    df = load_dataset(Path("tests/fixtures/sample_data.csv"))

    with pytest.raises(ValueError, match="less than 1"):
        split_dataset(df, test_size=0.6, val_size=0.5, random_state=42)
