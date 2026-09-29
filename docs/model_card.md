# Model Card: OpsGuard AI4I Baseline

## Model Summary

OpsGuard trains local scikit-learn baselines for binary machine-failure
classification on the UCI AI4I 2020 Predictive Maintenance Dataset. The current
verified local report compares a dummy classifier, balanced logistic regression,
and balanced random forest. The random-forest baseline has the strongest
average precision in the verified report.

This model card describes a portfolio baseline. It does not certify the model
for production maintenance decisions.

## Intended Use

Suitable uses:

- Demonstrating an end-to-end AI-engineering workflow.
- Comparing simple baselines on an imbalanced classification task.
- Exploring how threshold changes affect failure precision and recall.
- Running local single-record inference demos with saved artifacts.

Unsuitable uses:

- Autonomous maintenance decisions.
- Production alerting.
- Safety-critical scheduling.
- Claims about real industrial equipment performance.
- Threshold approval without operational validation.

## Task

Predict the binary target `Machine failure` from operational machine-condition
features available before the target label.

## Dataset

- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Source: UCI Machine Learning Repository, dataset ID `601`
- URL: <https://archive.ics.uci.edu/dataset/601/ai4i>
- DOI: `10.24432/C5HS5C`
- License: Creative Commons Attribution 4.0 International, CC BY 4.0
- Current local report size: 10,000 rows
- Positive class in the full verified report: 339 failure rows, 3.39%

The dataset is synthetic. See [DATA_CARD.md](DATA_CARD.md) for attribution,
schema, validation checks, and known limitations.

## Features

Included model inputs:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Excluded from model inputs:

- `UDI` / `UID`
- `Product ID`
- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

The failure-mode columns are excluded because they are target-derived
information for this binary failure task.

## Model Family and Preprocessing

Compared baselines:

- `DummyClassifier(strategy="most_frequent")`
- `LogisticRegression(class_weight="balanced")`
- `RandomForestClassifier(class_weight="balanced")`

Preprocessing is implemented inside scikit-learn pipelines:

- `Type` is one-hot encoded with `handle_unknown="ignore"`.
- Numeric columns are scaled for logistic regression.
- Numeric columns are passed through for dummy and random-forest baselines.

## Validation and Evaluation

The verified local report uses:

- stratified train/test split;
- test size `0.2`;
- random seed `42`;
- preprocessing fit only on the training split;
- fixed reporting threshold `0.5`.

The held-out test split contains 2,000 rows: 1,932 non-failure rows and 68
failure rows.

## Verified Metrics

Metrics from `reports/baseline_metrics.json`:

| Model | Average precision | ROC-AUC | Balanced accuracy | Failure precision | Failure recall | Failure F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy most frequent | 0.034 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.382 | 0.907 | 0.824 | 0.142 | 0.824 | 0.242 |
| Random forest | 0.737 | 0.962 | 0.880 | 0.589 | 0.779 | 0.671 |

Random-forest confusion matrix at threshold `0.5`:

| Actual class | Predicted no failure | Predicted failure |
| --- | ---: | ---: |
| No failure | 1,895 | 37 |
| Failure | 15 | 53 |

## Threshold

The threshold `0.5` is used as a fixed reporting convention. It is not tuned on
the final test set and is not operationally approved. The Streamlit app allows
interactive threshold exploration, but changing the slider does not change the
saved model artifact.

Threshold selection for real operations would require inspection-capacity
constraints, missed-failure cost analysis, calibration review, and validation
data separate from final test reporting.

## Error Analysis

At the random-forest reporting threshold:

- 53 failure rows were correctly flagged.
- 15 failure rows were missed.
- 37 non-failure rows were flagged.
- 1,895 non-failure rows were left unflagged.

False positives could create unnecessary inspection work. False negatives could
represent missed maintenance opportunities. These consequences are described as
modeling implications only; no production workflow is implemented.

## Ethical, Privacy, and Operational Considerations

- The dataset is synthetic and does not contain real customer, worker, or plant
  data in this project.
- A model trained on synthetic data should not be assumed to transfer to real
  equipment.
- Any operational use would need human review, monitoring, calibration checks,
  access control, logging, and incident-response procedures.
- The app should be presented as a decision-support demo, not as autonomous
  maintenance software.

## Known Limitations

- Synthetic dataset.
- Rare positive class.
- Stratified holdout split rather than validated chronological evaluation.
- No calibration analysis.
- No hyperparameter tuning.
- No live monitoring or drift detection.
- No production API or authentication layer.

## Recommended Next Steps

- Add calibration diagnostics.
- Add richer false-positive and false-negative examples.
- Capture and commit Streamlit screenshots for the demo README.
- Evaluate on a dataset with a validated time component before attempting
  time-series claims.
- Add deployment and monitoring only after the baseline evidence is stable.
