import pandas as pd

from src.data_utils import normalize_earthquake_columns


def test_normalize_usgs_magnitude_column():
    df = pd.DataFrame({"time": ["2020-01-01T00:00:00Z"], "mag": [3.4]})

    normalized = normalize_earthquake_columns(df)

    assert "magnitude" in normalized.columns
    assert normalized["magnitude"].tolist() == [3.4]


def test_keep_canonical_magnitude_column():
    df = pd.DataFrame({"magnitude": [3.4]})

    normalized = normalize_earthquake_columns(df)

    assert normalized is df
    assert normalized["magnitude"].tolist() == [3.4]
