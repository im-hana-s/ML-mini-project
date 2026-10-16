import pandas as pd

def test_prior_window_counts_exclude_current_timestamp():
    # The feature builder's rule: count only events strictly before current time.
    idx = pd.to_datetime([
        "2020-01-01T00:00:00Z",
        "2020-01-01T12:00:00Z",
        "2020-01-02T00:00:00Z",
    ], utc=True)
    ones = pd.Series(1.0, index=idx)
    counts = ones.rolling("1D", closed="left").sum()
    assert pd.isna(counts.iloc[0])
    assert counts.iloc[1] == 1
    assert counts.iloc[2] == 2
