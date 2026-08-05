"""Evaluation utilities for imbalanced binary failure classification."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a binary classifier with imbalanced-classification metrics."""
    scores = _positive_class_scores(estimator, x_test)
    y_pred = (scores >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "average_precision": float(average_precision_score(y_test, scores)),
        "recall_failure": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision_failure": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1_failure": float(f1_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "threshold_table": threshold_table(y_test, scores).to_dict(orient="records"),
    }

    if len(set(y_test.tolist())) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    else:
        metrics["roc_auc"] = None

    return metrics


def threshold_table(
    y_true: pd.Series,
    scores: np.ndarray,
    *,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Summarize the precision/recall tradeoff across candidate thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return pd.DataFrame(
            [
                {
                    "threshold": 0.5,
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                }
            ]
        )

    indices = np.linspace(0, thresholds.size - 1, num=min(max_rows, thresholds.size))
    rows = []
    for index in sorted({int(round(value)) for value in indices}):
        rows.append(
            {
                "threshold": float(thresholds[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
            }
        )
    return pd.DataFrame(rows)


def _positive_class_scores(estimator: Any, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        return np.asarray(probabilities)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        return np.asarray(scores, dtype=float)
    return np.asarray(estimator.predict(x_test), dtype=float)
