from pathlib import Path
import pandas as pd

from data_utils import normalize_earthquake_columns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

EQ_FILE = RAW / "earthquakes.csv"
ER_FILE = RAW / "eruptions.csv"
if not EQ_FILE.exists() or not ER_FILE.exists():
    raise FileNotFoundError(
        "Add data/raw/earthquakes.csv and data/raw/eruptions.csv first. "
        "See README.md for required schemas."
    )

eq = normalize_earthquake_columns(pd.read_csv(EQ_FILE))
er = pd.read_csv(ER_FILE)
required_eq = ["time", "latitude", "longitude", "depth", "magnitude"]
required_er = ["start_time", "end_time"]
if set(required_eq) - set(eq.columns):
    raise ValueError(f"Earthquake CSV requires columns: {required_eq}")
if set(required_er) - set(er.columns):
    raise ValueError(f"Eruption CSV requires columns: {required_er}")

eq["time"] = pd.to_datetime(eq["time"], utc=True, errors="coerce")
for col in ["latitude", "longitude", "depth", "magnitude"]:
    eq[col] = pd.to_numeric(eq[col], errors="coerce")
eq = eq.dropna(subset=required_eq).copy()
eq = eq.sort_values("time").drop_duplicates(subset=required_eq).reset_index(drop=True)
if eq.empty:
    raise ValueError("No earthquake rows have valid values in all required columns.")

er["start_time"] = pd.to_datetime(er["start_time"], utc=True, errors="coerce")
er["end_time"] = pd.to_datetime(er["end_time"], utc=True, errors="coerce")
er = er.dropna(subset=["start_time"]).sort_values("start_time").copy()
if er.empty:
    raise ValueError("No valid eruption start times found.")
if er["end_time"].isna().any():
    raise ValueError(
        "Every eruption interval needs an end_time. Resolve ongoing/open intervals "
        "using a documented cutoff before running this starter pipeline."
    )
if (er["end_time"] < er["start_time"]).any():
    raise ValueError("Found eruption end_time earlier than start_time.")
if not (
    (er["start_time"] <= eq["time"].max())
    & (er["end_time"] >= eq["time"].min())
).any():
    raise ValueError(
        "The earthquake catalog and eruption intervals do not overlap in time. "
        f"Earthquake range: {eq['time'].min()} to {eq['time'].max()}; "
        f"eruption range: {er['start_time'].min()} to {er['end_time'].max()}. "
        "Use an earthquake catalog covering the eruption study period."
    )

# Label 1 if event timestamp lies within any documented eruption interval;
# otherwise label 0. This reproduces contemporaneous state classification,
# not forecasting of a future eruption.
eq["erupting"] = 0
for row in er.itertuples(index=False):
    mask = (eq["time"] >= row.start_time) & (eq["time"] <= row.end_time)
    eq.loc[mask, "erupting"] = 1

# Counts use only earthquakes strictly earlier than the current timestamp.
# Simultaneous events do not count as prior events for one another.
time_indexed = eq.set_index("time").sort_index()
ones = pd.Series(1.0, index=time_indexed.index)
for window, name in [
    ("1D", "earthquakes_1d"),
    ("7D", "earthquakes_7d"),
    ("30D", "earthquakes_30d"),
]:
    time_indexed[name] = ones.rolling(window, closed="left").sum().to_numpy()

features = [
    "longitude", "latitude", "depth", "magnitude",
    "earthquakes_1d", "earthquakes_7d", "earthquakes_30d",
]
out = time_indexed.reset_index()
# Early records do not have a complete 30-day history; omit them.
out = out.dropna(subset=features + ["erupting"]).copy()
out["erupting"] = out["erupting"].astype(int)
out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
out.to_csv(PROCESSED / "model_data.csv", index=False)

print("Saved:", PROCESSED / "model_data.csv")
print("Rows after cleaning/features:", len(out))
print("Time range:", out["time"].min(), "to", out["time"].max())
print("Label counts (0=repose, 1=erupting):")
print(out["erupting"].value_counts().sort_index())
if out["erupting"].nunique() < 2:
    print("WARNING: only one target class remains; check the eruption timeline and data period.")
