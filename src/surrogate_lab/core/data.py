"""Dataset ingestion, schema validation, and reproducible splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from surrogate_lab.core.schema import DatasetSchema

DataFormat = Literal["csv", "parquet"]


class DataError(Exception):
    """Raised when a dataset cannot be loaded or does not match its schema."""


def _infer_format(path: Path) -> DataFormat:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    raise DataError(f"cannot infer format from extension {suffix!r}; pass format explicitly")


def load_dataset(path: Path | str, data_format: DataFormat | None = None) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame.

    Args:
        path: Path to the dataset file.
        data_format: ``"csv"`` or ``"parquet"``. Inferred from the file extension
            when omitted.

    Raises:
        DataError: If the file does not exist or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"dataset file not found: {path}")

    fmt = data_format or _infer_format(path)
    try:
        if fmt == "csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a project error
        raise DataError(f"failed to parse {path} as {fmt}: {exc}") from exc


def validate_schema(df: pd.DataFrame, schema: DatasetSchema) -> None:
    """Check that ``df`` contains every column declared in ``schema``.

    Raises:
        DataError: If any declared column is missing, or if the target column
            contains missing values.
    """
    missing = [column for column in schema.all_columns if column not in df.columns]
    if missing:
        raise DataError(f"dataset is missing declared columns: {missing}")
    if df[schema.target].isna().any():
        raise DataError(f"target column {schema.target!r} contains missing values")


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test partitions of a dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DatasetSplit:
    """Split a dataset into train/validation/test partitions.

    The split is deterministic: the same ``random_state`` on the same input
    always produces identical partitions (row indices and contents), because
    ``train_test_split`` is called twice with a fixed seed and no other
    source of randomness is involved.

    Args:
        df: The full dataset.
        test_size: Fraction of rows held out for the test partition.
        val_size: Fraction of rows held out for the validation partition
            (taken from the remainder after the test split).
        random_state: Seed controlling both split operations.

    Raises:
        ValueError: If ``test_size + val_size >= 1``.
    """
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    relative_val_size = val_size / (1 - test_size)
    train, validation = train_test_split(
        train_val, test_size=relative_val_size, random_state=random_state
    )
    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
