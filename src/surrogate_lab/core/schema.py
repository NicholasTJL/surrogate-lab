"""Pydantic schema describing the feature/target layout of a dataset."""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class DatasetSchema(BaseModel):
    """User-declared feature and target columns for a tabular dataset.

    ``categorical_features`` must be a subset of ``features``; every remaining
    feature is treated as numeric.
    """

    features: list[str]
    target: str
    categorical_features: list[str] = []

    @field_validator("features")
    @classmethod
    def features_must_be_non_empty_and_unique(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("features must not be empty")
        duplicates = sorted({name for name in value if value.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate feature names: {duplicates}")
        return value

    @model_validator(mode="after")
    def target_not_in_features(self) -> DatasetSchema:
        if self.target in self.features:
            raise ValueError(f"target {self.target!r} must not also be a feature")
        return self

    @model_validator(mode="after")
    def categorical_features_are_subset(self) -> DatasetSchema:
        unknown = sorted(set(self.categorical_features) - set(self.features))
        if unknown:
            raise ValueError(f"categorical_features not in features: {unknown}")
        return self

    @property
    def numeric_features(self) -> list[str]:
        """Features not declared categorical, in original order."""
        categorical = set(self.categorical_features)
        return [name for name in self.features if name not in categorical]

    @property
    def all_columns(self) -> list[str]:
        """All columns required in a dataset: features plus target."""
        return [*self.features, self.target]
