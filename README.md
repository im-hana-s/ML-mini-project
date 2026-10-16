# Predicting Eruptive Events at Volcanoes from Earthquake Data

A student mini-project inspired by the Stanford CS229 paper by Ben Mullet, Ankush Singh and Ying Qi Wong (Fall 2019). This starter implementation studies **contemporaneous eruption-state classification** at Kīlauea: `1 = erupting`, `0 = repose`. It does not establish operational or future eruption prediction.

## 1. Requirements
- Python 3.10, 3.11, or 3.12
- Git
- VS Code or another Python editor

## 2. Setup

### Windows PowerShell
```powershell
git clone <YOUR-PRIVATE-REPOSITORY-URL>
cd volcano-eruption-prediction
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`.

### macOS/Linux
```bash
git clone <YOUR-PRIVATE-REPOSITORY-URL>
cd volcano-eruption-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add the datasets
Put the earthquake catalog at `data/raw/earthquakes.csv` and eruption intervals at `data/raw/eruptions.csv`. Required schemas and data notes are in [`data/README.md`](data/README.md). Raw data is git-ignored. Do not commit datasets unless you have permission and the file size/license is suitable.

## 4. Run the pipeline
Run commands from the repository root, with the virtual environment activated:

```bash
python src/explore_data.py
python src/build_dataset.py
python src/train.py
```

Outputs:
- `outputs/figures/magnitude_distribution.png`
- `outputs/figures/daily_earthquake_counts.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curve.png` (an ROC plot when test data contains both classes; otherwise an explanation that ROC is undefined)
- `outputs/figures/feature_importance.png`
- `outputs/metrics/test_metrics.csv` and `test_metrics.json`
- `outputs/metrics/baseline_metrics.json`
- `outputs/metrics/split_info.json`
- `outputs/metrics/selected_model.json`
- `outputs/metrics/model_search.csv`
- `outputs/metrics/classification_report.txt`
- `models/random_forest.joblib`

Optional prediction for a CSV that already contains the seven engineered feature columns:
```bash
python src/predict.py data/processed/model_data.csv
```
This predicts the contemporaneous label defined for the project, not an eruption in a future time window.

## 5. Features
- `longitude`
- `latitude`
- `depth`
- `magnitude`
- earthquake record count in the preceding 1 day
- earthquake record count in the preceding 7 days
- earthquake record count in the preceding 30 days

Counts use prior timestamps only. The earliest records without a full rolling history are omitted.

## 6. Model and evaluation
A Random Forest classifier is selected by development-period AUROC when the chronological development period contains both classes. If it contains only one class, the script selects by log loss and prints a warning; this fallback does not measure the model's ability to distinguish eruption from repose. The chronological split is 70% train, 20% development, 10% test. Final metrics are reported on the last 10% only. Test AUROC is reported as undefined when that period lacks either class. Review split class counts and investigate time coverage and eruption labels before drawing conclusions.

Metrics include AUROC, Cohen's kappa, precision, recall, F1 and accuracy. Compare with the majority-class baseline. Do not insert the reference paper's results as your own results.

## 7. Tests
```bash
pytest -q
```
GitHub Actions runs these tests on pushes and pull requests to `main`.

## 8. Data provenance and limitations
The reference paper reports 3,764 earthquakes and uses location, depth, magnitude and earthquake counts over 1, 7 and 30 days. Your downloaded data may not match the paper's exact catalog or filtering. Cite the source actually used, report your real row counts/date range, and explain the difference. Catalog completeness and eruption-interval definitions affect results. Contemporaneous classification is not the same as forecasting an eruption in advance.

## 9. Research reference
Mullet, B., Singh, A., & Wong, Y. Q. (2019). *Predicting Eruptive Events at Volcanoes from Earthquake Data*. Stanford CS229, Fall 2019. See the original paper and the authors' repository linked in the project submission materials.

## 10. Team contributions
Document actual work and commits:
- Member 1: data acquisition, cleaning, eruption labels, rolling features and data tests.
- Member 2: split strategy, Random Forest, tuning, evaluation and model plots.
- Both: integration, README, report, presentation and demonstration.
