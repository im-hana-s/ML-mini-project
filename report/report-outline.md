# Report outline (replace prompts with your actual findings)

## Title
Predicting Eruptive Events at Kīlauea Volcano Using Earthquake Catalog Features

## Problem statement
Define the target carefully: contemporaneous eruption-state classification (`erupting` versus `repose`). Do not claim future forecasting unless you build and evaluate a future-window target.

## Dataset
Name and cite the earthquake catalog and eruption timeline actually used. State download date, filters, date range, number of raw and retained rows, class counts, units and data limitations.

## Methodology
Explain data cleaning, timestamp standardization, labels, previous 1/7/30-day counts, chronological 70/20/10 split, Random Forest, development-set selection and test metrics.

## Results
Insert actual test metrics from `outputs/metrics/test_metrics.json`. Compare with the baseline. Add confusion matrix and ROC curve. Do not copy the reference paper's scores as your own.

## Discussion and limitations
Discuss false positives/negatives, imbalance, temporal distribution, catalog completeness, eruption timeline accuracy and why the task does not establish operational eruption forecasting.

## Contributions
Record each member's actual tasks and meaningful Git commits.

## References
Cite the research paper and all dataset sources.
