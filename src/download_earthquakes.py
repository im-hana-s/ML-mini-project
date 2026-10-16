from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from data_utils import normalize_earthquake_columns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EARTHQUAKE_FILE = RAW / "earthquakes.csv"
ERUPTION_FILE = RAW / "eruptions.csv"

QUERY = {
    "format": "csv",
    "starttime": "1950-01-01",
    "endtime": "2018-12-31T23:59:59",
    "latitude": 19.421,
    "longitude": -155.287,
    "maxradius": 0.5,
    "minmagnitude": 3,
    "orderby": "time-asc",
}


def main() -> None:
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?{urlencode(QUERY)}"
    try:
        with urlopen(url, timeout=120) as response:
            csv_data = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download the USGS earthquake catalog: {exc}") from exc

    eq = normalize_earthquake_columns(pd.read_csv(BytesIO(csv_data)))
    required = ["time", "latitude", "longitude", "depth", "magnitude"]
    missing = sorted(set(required) - set(eq.columns))
    if missing:
        raise ValueError(f"Downloaded earthquake CSV is missing required columns: {missing}")
    if eq.empty:
        raise ValueError("The USGS query returned no earthquake records.")

    eq["time"] = pd.to_datetime(eq["time"], utc=True, errors="coerce")
    for column in required[1:]:
        eq[column] = pd.to_numeric(eq[column], errors="coerce")
    invalid_rows = eq[required].isna().any(axis=1)
    invalid_count = int(invalid_rows.sum())
    eq = eq.dropna(subset=required).copy()
    if eq.empty:
        raise ValueError("No USGS rows have valid values in all required columns.")

    eruptions = pd.read_csv(ERUPTION_FILE)
    eruption_start = pd.to_datetime(eruptions["start_time"], utc=True, errors="coerce")
    eruption_end = pd.to_datetime(eruptions["end_time"], utc=True, errors="coerce")
    overlaps = (eruption_start <= eq["time"].max()) & (eruption_end >= eq["time"].min())
    if not overlaps.any():
        raise ValueError(
            "The downloaded earthquake catalog does not overlap the eruption intervals; "
            "refusing to replace the existing CSV."
        )

    eq["time"] = eq["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    RAW.mkdir(parents=True, exist_ok=True)
    eq.to_csv(EARTHQUAKE_FILE, index=False)
    print(f"Saved {len(eq)} USGS earthquakes to {EARTHQUAKE_FILE}")
    if invalid_count:
        print(f"Dropped {invalid_count} USGS row(s) with missing or invalid required values.")
    print(f"Time range: {eq['time'].min()} to {eq['time'].max()}")
    print(
        "Query: Kilauea center (19.421, -155.287), radius 0.5 degrees, "
        "1950-01-01 through 2018-12-31, magnitude >= 3"
    )


if __name__ == "__main__":
    main()
