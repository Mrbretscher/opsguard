# OpsGuard

OpsGuard is a lean AI-engineering portfolio project for reproducible predictive-maintenance baselines on the UCI AI4I 2020 Predictive Maintenance Dataset.

## Current Status

Milestone 0 and Milestone 1 are implemented as a local scikit-learn baseline workflow. The project can fetch the AI4I dataset, validate schema and quality expectations, build leakage-controlled tabular features, train three baseline classifiers, and evaluate the result with imbalanced-classification metrics.

This is not a production maintenance system. It is a portfolio baseline intended to demonstrate sound ML engineering habits before adding heavier MLOps, deployment, monitoring, or time-series components.

## What It Demonstrates

- Python 3.11 package using a `src/` layout.
- Dataset acquisition without committing the raw dataset.
- Dataset attribution and licensing documentation.
- Schema and quality validation.
- Reproducible preprocessing with scikit-learn `Pipeline` and `ColumnTransformer`.
- Dummy, logistic-regression, and random-forest baselines.
- Evaluation for an imbalanced binary failure-classification problem.
- Unit tests, Ruff formatting/linting, mypy, and PowerShell verification scripts.

## Dataset

OpsGuard uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

- Source: <https://archive.ics.uci.edu/dataset/601/ai4i>
- DOI: `10.24432/C5HS5C`
- License: Creative Commons Attribution 4.0 International, CC BY 4.0
- File: `ai4i2020.csv`
- Size: approximately 10,000 rows
- Nature: synthetic predictive-maintenance data

The full dataset is not committed to this repository. Fetch it locally with the command below.

## Local Setup

Install Python 3.11 and make sure it is available through the Windows launcher as `py -3.11`.

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

## Fetch Data

```powershell
.\scripts\fetch_data.ps1
```

This writes the raw CSV to `data/raw/ai4i2020.csv`, which is ignored by Git.

## Run Baselines

```powershell
opsguard evaluate-baselines --data data/raw/ai4i2020.csv
```

The baseline workflow trains:

- `DummyClassifier(strategy="most_frequent")`
- `LogisticRegression(class_weight="balanced")`
- `RandomForestClassifier(class_weight="balanced")`

Features are limited to operational columns available before the label:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Identifier columns, `Product ID`, `Machine failure`, and failure-mode target columns are excluded from the feature matrix.

## Evaluation

The evaluation emphasizes minority-class failure detection rather than plain accuracy:

- Average precision / PR-AUC
- Failure-class recall
- Failure-class precision
- Failure-class F1
- ROC-AUC
- Balanced accuracy
- Confusion matrix
- Classification report
- Precision/recall threshold table

The default threshold is `0.5`. No operational threshold is presented as production-ready.

## Verify

```powershell
.\scripts\verify.ps1
```

The verification script runs:

- `ruff check .`
- `ruff format --check .`
- `mypy src/opsguard`
- `pytest --cov=opsguard`
- package import smoke test

## Documentation

- [Project brief](docs/project_brief.md)
- [Data card](docs/data_card.md)
- [Evaluation plan](docs/evaluation_plan.md)
- [Roadmap](docs/roadmap.md)

## Explicit Non-Goals For This Release

This release intentionally excludes TensorFlow, MLflow, FastAPI, Streamlit, Docker, cloud deployment, model monitoring, MetroPT-3, and a production database.

## License

Project code is released under the MIT License. Dataset rights remain with the dataset publisher and are governed by CC BY 4.0.
