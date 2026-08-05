# Evaluation Plan

## Task

Predict the binary `Machine failure` target from operational AI4I features.

## Split Strategy

Use a stratified train/test split with a fixed random seed. Stratification preserves the minority failure class across the train and test sets.

Preprocessing is fit only on the training split inside scikit-learn pipelines.

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

The random forest is the first tree-based baseline, not a tuned final model.

## Metrics

Primary metrics:

- Average precision / PR-AUC
- Failure-class recall
- Failure-class precision
- Failure-class F1

Secondary metrics:

- ROC-AUC
- Balanced accuracy
- Confusion matrix
- Classification report

## Threshold

Evaluate the default `0.5` threshold and report a precision/recall threshold table. No threshold is considered operationally approved in Milestone 1.

## Reporting Principles

Results should be reported only after commands have been run. Documentation should distinguish implemented behavior from planned later work.
