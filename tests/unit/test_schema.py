import pytest
from pydantic import ValidationError

from surrogate_lab.core.schema import DatasetSchema


def test_valid_schema_computes_numeric_features() -> None:
    schema = DatasetSchema(features=["a", "b", "c"], target="y", categorical_features=["c"])

    assert schema.numeric_features == ["a", "b"]
    assert schema.all_columns == ["a", "b", "c", "y"]


def test_empty_features_raises() -> None:
    with pytest.raises(ValidationError, match="features must not be empty"):
        DatasetSchema(features=[], target="y")


def test_duplicate_features_raises() -> None:
    with pytest.raises(ValidationError, match="duplicate feature names"):
        DatasetSchema(features=["a", "a"], target="y")


def test_target_in_features_raises() -> None:
    with pytest.raises(ValidationError, match="must not also be a feature"):
        DatasetSchema(features=["a", "y"], target="y")


def test_unknown_categorical_feature_raises() -> None:
    with pytest.raises(ValidationError, match="categorical_features not in features"):
        DatasetSchema(features=["a", "b"], target="y", categorical_features=["c"])
