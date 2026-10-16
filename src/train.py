from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, cohen_kappa_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
)
from sklearn.model_selection import ParameterGrid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_data.csv"
MODELS = ROOT / "models"
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"
for folder in [MODELS, FIGURES, METRICS]:
    folder.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Run `python src/build_dataset.py` first.")

df = pd.read_csv(DATA)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)
features = [
    "longitude", "latitude", "depth", "magnitude",
    "earthquakes_1d", "earthquakes_7d", "earthquakes_30d",
]
target = "erupting"
if df[features + [target]].isna().any().any():
    raise ValueError("Missing feature/target values remain; rebuild/inspect data.")
if df[target].nunique() != 2:
    raise ValueError("Need both eruption classes to train/evaluate this classifier.")

n = len(df)
train_end, dev_end = int(n * .70), int(n * .90)
train, dev, test = df.iloc[:train_end], df.iloc[train_end:dev_end], df.iloc[dev_end:]
if min(len(train), len(dev), len(test)) == 0:
    raise ValueError("Not enough rows for 70/20/10 chronological split.")
X_train, y_train = train[features], train[target].astype(int)
X_dev, y_dev = dev[features], dev[target].astype(int)
X_test, y_test = test[features], test[target].astype(int)

split_info = {
    "strategy": "chronological 70/20/10; rows sorted by event time",
    "train_rows": len(train), "development_rows": len(dev), "test_rows": len(test),
    "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
    "development_class_counts": {
        str(k): int(v) for k, v in y_dev.value_counts().sort_index().items()
    },
    "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
    "train_start": str(train.time.min()), "train_end": str(train.time.max()),
    "development_start": str(dev.time.min()), "development_end": str(dev.time.max()),
    "test_start": str(test.time.min()), "test_end": str(test.time.max()),
}
(METRICS / "split_info.json").write_text(json.dumps(split_info, indent=2))

if y_train.nunique() < 2:
    raise ValueError("Training period has only one class. Review date coverage/labels.")

# Use development data for model selection. Test data is not used here.
selection_metric = "development_auroc" if y_dev.nunique() == 2 else "development_log_loss"
if y_dev.nunique() == 1:
    print(
        "WARNING: The chronological development period contains only one class "
        f"({int(y_dev.iloc[0])}). AUROC is undefined; selecting by log loss instead. "
        "This does not measure the model's ability to distinguish eruption from repose."
    )
grid = {
    "n_estimators": [100, 250],
    "max_depth": [8, 15, None],
    "min_samples_leaf": [1, 3],
}
best_model, best_params = None, None
best_score = -np.inf if selection_metric == "development_auroc" else np.inf
search_rows = []
for params in ParameterGrid(grid):
    model = RandomForestClassifier(
        **params, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_dev)[:, 1]
    score = (
        roc_auc_score(y_dev, scores)
        if selection_metric == "development_auroc"
        else log_loss(y_dev, scores, labels=[0, 1])
    )
    search_rows.append({**params, selection_metric: score})
    print(f"{params} -> {selection_metric}={score:.4f}")
    improved = (
        score > best_score
        if selection_metric == "development_auroc"
        else score < best_score
    )
    if improved:
        best_model, best_params, best_score = model, params, score

pd.DataFrame(search_rows).to_csv(METRICS / "model_search.csv", index=False)
joblib.dump(
    {"model": best_model, "features": features, "target": target, "params": best_params},
    MODELS / "random_forest.joblib",
)
(METRICS / "selected_model.json").write_text(json.dumps(
    {"best_params": best_params, selection_metric: best_score}, indent=2
))

# Final evaluation on the chronological test period.
pred = best_model.predict(X_test)
prob = best_model.predict_proba(X_test)[:, 1]
test_has_two_classes = y_test.nunique() == 2
kappa_is_defined = len(set(y_test).union(set(pred))) == 2
results = {
    "test_accuracy": accuracy_score(y_test, pred),
    "test_precision": precision_score(y_test, pred, zero_division=0),
    "test_recall": recall_score(y_test, pred, zero_division=0),
    "test_f1": f1_score(y_test, pred, zero_division=0),
    "test_cohen_kappa": cohen_kappa_score(y_test, pred) if kappa_is_defined else None,
    "test_auroc": roc_auc_score(y_test, prob) if test_has_two_classes else None,
    "test_auroc_note": (
        None if test_has_two_classes
        else "Undefined because the chronological test period contains only one class."
    ),
    "test_interpretation_note": (
        None if test_has_two_classes
        else "Test scores do not measure discrimination because the test period has one class."
    ),
    "test_rows": len(test),
    "test_positive_count": int(y_test.sum()),
    "test_negative_count": int((y_test == 0).sum()),
    "selected_params": best_params,
}
(METRICS / "test_metrics.json").write_text(json.dumps(results, indent=2))
pd.DataFrame([results]).to_csv(METRICS / "test_metrics.csv", index=False)
(METRICS / "classification_report.txt").write_text(
    classification_report(y_test, pred, labels=[0, 1], zero_division=0)
)

# Majority-class baseline.
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
baseline = {
    "baseline_accuracy": accuracy_score(y_test, dummy_pred),
    "baseline_precision": precision_score(y_test, dummy_pred, zero_division=0),
    "baseline_recall": recall_score(y_test, dummy_pred, zero_division=0),
    "baseline_f1": f1_score(y_test, dummy_pred, zero_division=0),
    "baseline_cohen_kappa": (
        cohen_kappa_score(y_test, dummy_pred)
        if len(set(y_test).union(set(dummy_pred))) == 2 else None
    ),
}
(METRICS / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2))

cm = confusion_matrix(y_test, pred, labels=[0, 1])
ConfusionMatrixDisplay(cm, display_labels=["Repose", "Erupting"]).plot()
plt.title("Random Forest — Chronological Test Set")
plt.tight_layout()
plt.savefig(FIGURES / "confusion_matrix.png", dpi=160)
plt.close()

if y_test.nunique() == 2:
    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("Random Forest — Chronological Test ROC")
else:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(
        0.5, 0.55,
        "ROC curve unavailable",
        ha="center", va="center", fontsize=16, fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.42,
        "The chronological test period contains only one class.",
        ha="center", va="center", fontsize=11, transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.32,
        "Both eruption and repose examples are required to calculate AUROC.",
        ha="center", va="center", fontsize=10, transform=ax.transAxes,
    )
    plt.title("Random Forest — Chronological Test ROC")
plt.tight_layout()
plt.savefig(FIGURES / "roc_curve.png", dpi=160)
plt.close()

importance = pd.Series(best_model.feature_importances_, index=features).sort_values()
importance.plot(kind="barh", figsize=(8, 5))
plt.xlabel("Feature importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(FIGURES / "feature_importance.png", dpi=160)
plt.close()

print("\nSelected parameters:", best_params)
print(f"{selection_metric}:", round(best_score, 4))
print("\nTEST METRICS (report these actual values):")
for key, value in results.items():
    print(f"{key}: {value}")
print("\nBaseline:", baseline)
print("\nSaved model to", MODELS / "random_forest.joblib")
print("Saved metrics to", METRICS)
