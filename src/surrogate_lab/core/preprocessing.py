"""Scikit-learn preprocessing pipelines built from a :class:`DatasetSchema`."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from surrogate_lab.core.schema import DatasetSchema


def build_preprocessor(schema: DatasetSchema) -> ColumnTransformer:
    """Build a :class:`ColumnTransformer` that scales numeric features and
    one-hot encodes categorical features declared in ``schema``.

    Unknown categories seen at prediction time are ignored (encoded as all
    zeros) rather than raising, so batch prediction on new data does not
    crash on a previously unseen category.
    """
    transformers = []
    if schema.numeric_features:
        transformers.append(("numeric", StandardScaler(), schema.numeric_features))
    if schema.categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                schema.categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers)
