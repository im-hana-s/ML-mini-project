import pandas as pd

def test_eruption_interval_membership():
    events = pd.to_datetime([
        "2020-01-01T00:00:00Z",
        "2020-01-02T00:00:00Z",
        "2020-01-04T00:00:00Z",
    ], utc=True)
    start = pd.Timestamp("2020-01-02T00:00:00Z")
    end = pd.Timestamp("2020-01-03T00:00:00Z")
    labels = [int(start <= t <= end) for t in events]
    assert labels == [0, 1, 0]
