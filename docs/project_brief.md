# OpsGuard Project Brief

## Summary

OpsGuard is an AI-engineering portfolio project that begins with a reproducible scikit-learn baseline for machine-failure classification on the UCI AI4I 2020 Predictive Maintenance Dataset.

The first release focuses on clean repository structure, data acquisition, validation, leakage-aware preprocessing, baseline modeling, and credible evaluation for an imbalanced binary classification task.

## Problem

Predictive maintenance systems estimate whether equipment is at risk of failure based on operational sensor and product-process signals. This project uses a synthetic public dataset to demonstrate the engineering workflow around that problem, not to make live maintenance decisions.

## Current Implementation

- Fetches AI4I dataset ID `601` from UCI using `ucimlrepo`.
- Stores raw data locally under `data/raw/`, which is ignored by Git.
- Validates expected schema, missing values, binary targets, and target class presence.
- Builds a leakage-controlled feature matrix.
- Trains Dummy, logistic-regression, and random-forest baselines.
- Evaluates with metrics appropriate for class imbalance.
- Persists the selected baseline artifact for local inference experiments.
- Provides a Streamlit GUI for single-record risk checks and report review.
- Provides Docker packaging for the local GUI demo.
- Provides unit tests and local verification scripts.

## Technology Stack

- Python 3.11
- pandas
- numpy
- scikit-learn
- matplotlib
- ucimlrepo
- pytest
- pytest-cov
- Ruff
- mypy
- JupyterLab
- Streamlit
- Docker

## Scope Boundaries

OpsGuard is currently a local baseline workflow with a demo GUI and local Docker packaging. It does not yet include API serving, experiment tracking, production monitoring, cloud deployment, alert routing, authentication, or production data storage.
