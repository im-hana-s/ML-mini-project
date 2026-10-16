import pandas as pd


def normalize_earthquake_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "magnitude" not in df.columns and "mag" in df.columns:
        return df.rename(columns={"mag": "magnitude"})
    return df
