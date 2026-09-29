# OpsGuard Demo Script

This script is designed for a two-to-three-minute recruiter or technical
screening walkthrough.

## 1. Opening Problem Statement

"OpsGuard is a predictive-maintenance portfolio project. The goal is to predict
binary machine-failure risk from machine-condition fields while avoiding common
ML pitfalls like target leakage, misleading accuracy, and unverified production
claims."

## 2. Project Scope

"This is a local baseline workflow, not a production maintenance platform. It
fetches the UCI AI4I synthetic predictive-maintenance dataset, validates the
schema, trains three scikit-learn baselines, evaluates them with
imbalanced-classification metrics, saves a model artifact, and exposes a small
Streamlit demo."

## 3. Application Input

Open the Streamlit app:

```powershell
.\scripts\run_streamlit_app.ps1
```

In the "Risk check" tab:

"The app starts with example input presets. I can choose a nominal low-wear
case, a high-tool-wear case, or a thermal-and-torque stress case, then edit the
six model inputs: product type, air temperature, process temperature,
rotational speed, torque, and tool wear."

## 4. Model Output

Click "Run risk check."

"The result panel shows the estimated failure risk, the selected decision
threshold, and whether the record is flagged for review. The language is
careful: this is a model signal for analysis, not a maintenance instruction."

## 5. Threshold and Human Review

Move the sidebar threshold slider.

"The threshold is intentionally exposed as an exploratory control. Lower
thresholds generally catch more failures but flag more non-failures. Higher
thresholds reduce review volume but can miss failures. In a real workflow, this
threshold would need operational review, inspection-capacity constraints, and
calibration evidence."

## 6. Evaluation Evidence

Open the "Evaluation" tab.

"The latest verified report compares three baselines. On the held-out test set,
the random forest has average precision 0.737, ROC-AUC 0.962, balanced accuracy
0.880, failure precision 0.589, failure recall 0.779, and failure F1 0.671.
At the fixed 0.5 threshold, it catches 53 of 68 failure rows, misses 15, and
flags 37 non-failure rows."

## 7. Architecture Summary

"The architecture is deliberately small and inspectable. The data script fetches
AI4I into `data/raw`. Validation checks schema and target fields. Feature code
removes identifiers, target columns, and failure-mode labels to control leakage.
Training builds scikit-learn pipelines and writes metrics, plots, and saved
model artifacts. Inference validates one input record and scores it through the
saved pipeline. Streamlit reads the local report and artifact for the demo."

## 8. Limitation

Open the "Limitations" tab.

"The biggest limitation is that AI4I is synthetic. The model is useful for
showing ML engineering discipline, but it should not be treated as evidence that
the same numbers would hold on real equipment. The project also does not include
production monitoring, alert routing, calibration governance, or a human-review
system."

## 9. Closing Statement

"The main thing I want this project to show is not that a random forest is a
finished maintenance system, but that I can build a reproducible ML baseline
with leakage controls, appropriate metrics for imbalance, saved artifacts, a
demo interface, and clear documentation about what is verified versus what is
planned."

## Optional Follow-Up Questions

If asked why accuracy is not the headline metric:

"The failure class is only 3.39% of the verified full report. A model can get
high accuracy by mostly predicting no failure. That is why the README leads with
average precision, recall, precision, F1, balanced accuracy, and the confusion
matrix."

If asked what should come next:

"I would add calibration analysis, richer error analysis, checked-in screenshots,
and then evaluate data with a validated time axis before making time-series or
monitoring claims."
