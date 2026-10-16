from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"

def main():
    parser = argparse.ArgumentParser(
        description="Classify eruption state for rows with engineered features."
    )
    parser.add_argument("csv", help="CSV containing the seven model feature columns")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train model first with `python src/train.py`.")
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]
    df = pd.read_csv(args.csv)
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing feature columns: {missing}")
    df["predicted_erupting"] = model.predict(df[features])
    df["eruption_probability"] = model.predict_proba(df[features])[:, 1]
    output = Path(args.csv).with_name("predictions.csv")
    df.to_csv(output, index=False)
    print("Saved predictions to", output)
    print("Note: predictions classify the contemporaneous target used in training; "
          "they are not validated future eruption warnings.")

if __name__ == "__main__":
    main()
