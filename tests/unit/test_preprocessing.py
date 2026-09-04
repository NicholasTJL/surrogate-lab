import pandas as pd

from surrogate_lab.core.preprocessing import build_preprocessor
from surrogate_lab.core.schema import DatasetSchema


def test_preprocessor_scales_numeric_and_encodes_categorical() -> None:
    schema = DatasetSchema(features=["x1", "x2", "cat"], target="y", categorical_features=["cat"])
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "cat": ["a", "b", "a", "b"],
        }
    )

    transformer = build_preprocessor(schema)
    transformed = transformer.fit_transform(df)

    # 2 scaled numeric columns + 2 one-hot columns (a, b)
    assert transformed.shape == (4, 4)


def test_preprocessor_handles_numeric_only_schema() -> None:
    schema = DatasetSchema(features=["x1", "x2"], target="y")
    df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})

    transformer = build_preprocessor(schema)
    transformed = transformer.fit_transform(df)

    assert transformed.shape == (2, 2)


def test_preprocessor_ignores_unseen_category_at_transform_time() -> None:
    schema = DatasetSchema(features=["cat"], target="y", categorical_features=["cat"])
    train_df = pd.DataFrame({"cat": ["a", "b"]})
    new_df = pd.DataFrame({"cat": ["c"]})

    transformer = build_preprocessor(schema)
    transformer.fit(train_df)
    transformed = transformer.transform(new_df)

    # Unknown category encodes as all zeros rather than raising.
    assert transformed.shape == (1, 2)
    assert transformed.sum() == 0
