"""Baseline model definitions and training helpers."""

from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from opsguard.config import RANDOM_SEED, TEST_SIZE
from opsguard.features import build_preprocessor


@dataclass(frozen=True)
class DataSplit:
    """Train/test split container for baseline training and evaluation."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> DataSplit:
    """Create a stratified holdout split for imbalanced binary classification."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return DataSplit(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def build_dummy_pipeline(*, random_state: int = RANDOM_SEED) -> Pipeline:
    """Build the trivial reference classifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                DummyClassifier(strategy="most_frequent", random_state=random_state),
            ),
        ]
    )


def build_logistic_regression_pipeline(
    *,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build a leakage-controlled logistic-regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(
    *,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build the first tree-based baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_baseline_pipelines(*, random_state: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Return all Milestone 1 baseline pipelines."""
    return {
        "dummy_most_frequent": build_dummy_pipeline(random_state=random_state),
        "logistic_regression": build_logistic_regression_pipeline(
            random_state=random_state
        ),
        "random_forest": build_random_forest_pipeline(random_state=random_state),
    }
