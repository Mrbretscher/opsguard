# Evaluation Plan

## Task

Predict the binary `Machine failure` target from operational AI4I features.

## Split Strategy

Use a stratified train/test split with fixed random seed `42`. Stratification preserves the minority failure class across the train and test sets.

Preprocessing is fit only on the training split inside scikit-learn pipelines.

The final test set is isolated from model selection. Milestone 1B uses fixed baseline model configurations and does not tune hyperparameters, choose thresholds, or iterate model settings based on final test-set performance.

## Leakage Controls

Excluded feature columns:

- `UDI` / `UID`
- `Product ID`
- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

The failure-mode columns are treated as target information, not model inputs.

## Models

- Dummy most-frequent classifier
- Logistic regression with balanced class weights
- Random forest with balanced class weights

The random forest is the first tree-based baseline because it can capture nonlinear tabular feature interactions and provides a common scikit-learn comparison point against logistic regression. It is not a tuned final model.

Class imbalance is handled through estimator class weights for the logistic-regression and random-forest baselines. No resampling is fit on the full dataset, and no test-set labels are used to compute training transformations.

## Metrics

Primary metrics:

- Average precision / PR-AUC
- Failure-class recall
- Failure-class precision
- Failure-class F1

Secondary metrics:

- Accuracy for context only
- ROC-AUC
- Balanced accuracy
- Confusion matrix
- Classification report
- Class distribution

Accuracy must not be used by itself to select a model because the failure class is rare.

## Threshold

Evaluate the default `0.5` threshold and report a precision/recall threshold table. No threshold is considered operationally approved in Milestone 1.

## Reporting Principles

Results should be reported only after commands have been run. Documentation should distinguish implemented behavior from planned later work.

The baseline workflow writes machine-readable metrics to `reports/baseline_metrics.json` and diagnostic plots to `reports/plots/`. The `reports/` directory is ignored by Git so local experiment outputs are reproducible without being committed.

## Known Limitations

- The AI4I dataset is synthetic and may not represent a real plant's sensor behavior, maintenance policy, or failure mechanisms.
- The rare failure class can make high accuracy misleading.
- Failure-mode labels are excluded from inputs because they are target-derived information.
- Row order is not treated as a validated time axis for Milestone 1B, so the initial split is stratified rather than chronological.
- The default `0.5` threshold is an evaluation convention, not an operational decision threshold.
