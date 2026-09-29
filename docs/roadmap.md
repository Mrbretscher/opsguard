# Roadmap

## Milestone 0: Repository Foundation

Status: implemented.

- Python 3.11 package with `src/` layout
- Project metadata and dependency declarations
- Ruff, mypy, pytest, and coverage configuration
- PowerShell setup and verification scripts
- Initial README and project documentation

## Milestone 1: AI4I Scikit-Learn Baseline

Status: implemented.

- UCI AI4I dataset acquisition through `ucimlrepo`
- Local raw data storage ignored by Git
- Schema and quality validation
- Leakage-controlled feature/target split
- Dummy, logistic-regression, and random-forest baselines
- Imbalanced binary classification evaluation
- Saved model artifact for local inference experiments
- Streamlit demo interface
- Docker packaging for the local GUI demo
- Unit tests for import, validation, features, modeling, inference, app support, and Docker setup

## Later Milestones

Planned, not yet implemented:

- Richer exploratory analysis and error analysis
- More complete reproducible experiment records
- Anomaly-detection framing
- Time-series modeling on an appropriate dataset
- MLflow or equivalent experiment tracking
- API serving
- Cloud deployment
- Model monitoring
- Additional predictive-maintenance datasets such as MetroPT-3
