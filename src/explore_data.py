from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

path = RAW / "earthquakes.csv"
if not path.exists():
    raise FileNotFoundError(f"Place earthquake catalog at {path}")

df = pd.read_csv(path)
required = ["time", "latitude", "longitude", "depth", "magnitude"]
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f"Earthquake CSV is missing columns: {missing}")

df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
for col in required[1:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
before = len(df)
df = df.dropna(subset=required).sort_values("time")
print(f"Rows read: {before}; rows with required fields: {len(df)}")
print(f"Time range: {df.time.min()} to {df.time.max()}")
print("\nSummary:")
print(df[required[1:]].describe())
print("\nMissing values in original data:")
print(pd.read_csv(path)[required].isna().sum())
print("\nPotential exact duplicate rows:", df.duplicated(subset=required).sum())

plt.figure(figsize=(8, 5))
df["magnitude"].hist(bins=30)
plt.xlabel("Magnitude")
plt.ylabel("Earthquake count")
plt.title("Earthquake magnitude distribution")
plt.tight_layout()
plt.savefig(FIGURES / "magnitude_distribution.png", dpi=150)
plt.close()

daily = df.set_index("time").resample("D").size()
plt.figure(figsize=(11, 4))
daily.plot()
plt.xlabel("Date (UTC)")
plt.ylabel("Earthquake records per day")
plt.title("Daily earthquake activity")
plt.tight_layout()
plt.savefig(FIGURES / "daily_earthquake_counts.png", dpi=150)
plt.close()
print(f"\nSaved plots to {FIGURES}")
