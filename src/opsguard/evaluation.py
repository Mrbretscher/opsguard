"""Evaluation utilities for imbalanced binary failure classification."""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
    scores = positive_class_scores(estimator, x_test)
    y_pred = (scores >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_test, scores))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    matrix = cast(
        list[list[int]],
        confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    )

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "average_precision": pr_auc,
        "recall_failure": recall,
        "precision_failure": precision,
        "f1_failure": f1,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "confusion_matrix": matrix,
        "confusion_matrix_interpretation": confusion_matrix_interpretation(
            matrix,
            threshold=threshold,
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "threshold_table": threshold_table(
            y_test,
            scores,
            selected_threshold=threshold,
        ).to_dict(orient="records"),
        "operating_threshold": operating_threshold_metadata(
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
        ),
        "error_analysis_notes": error_analysis_notes(
            matrix,
            threshold=threshold,
        ),
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
    selected_threshold: float = 0.5,
) -> pd.DataFrame:
    """Summarize threshold tradeoffs with confusion counts."""
    _, _, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return pd.DataFrame([threshold_row(y_true, scores, selected_threshold)])

    indices = np.linspace(0, thresholds.size - 1, num=min(max_rows, thresholds.size))
    candidate_thresholds = {
        float(thresholds[int(round(value))]) for value in indices
    } | {float(selected_threshold)}
    rows = [
        threshold_row(y_true, scores, candidate_threshold)
        for candidate_threshold in sorted(candidate_thresholds)
    ]
    return pd.DataFrame(rows)


def threshold_row(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Return precision, recall, and error counts for one threshold."""
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = (
        int(value) for value in matrix.ravel()
    )
    return {
        "threshold": float(threshold),
        "precision_failure": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_failure": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_failure": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_failures": int(true_positives + false_positives),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positive_rate": _safe_rate(
            false_positives,
            false_positives + true_negatives,
        ),
        "false_negative_rate": _safe_rate(
            false_negatives,
            false_negatives + true_positives,
        ),
    }


def operating_threshold_metadata(
    *,
    threshold: float,
    precision: float,
    recall: float,
    f1: float,
) -> dict[str, Any]:
    """Describe the threshold used for baseline metric reporting."""
    return {
        "threshold": float(threshold),
        "source": "fixed_default_threshold",
        "selection_status": "not_operationally_approved",
        "selection_note": (
            "This threshold is used to report baseline classification metrics. "
            "It is not tuned on the final test set and is not an approved "
            "production operating point."
        ),
        "metrics_at_threshold": {
            "precision_failure": float(precision),
            "recall_failure": float(recall),
            "f1_failure": float(f1),
        },
    }


def confusion_matrix_interpretation(
    matrix: list[list[int]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Translate a binary confusion matrix into labeled counts."""
    true_negatives, false_positives, false_negatives, true_positives = (
        _confusion_counts(matrix)
    )
    return {
        "label_order": (
            "Rows are actual classes [0=no failure, 1=failure]; columns are "
            "predicted classes [0=no failure, 1=failure]."
        ),
        "threshold": float(threshold),
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "summary": (
            f"At threshold {threshold:.3f}, {true_positives} failures were "
            f"flagged correctly, {false_negatives} failures were missed, "
            f"{false_positives} non-failure rows were flagged, and "
            f"{true_negatives} non-failure rows were left unflagged."
        ),
    }


def error_analysis_notes(
    matrix: list[list[int]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Describe false-positive and false-negative implications conservatively."""
    true_negatives, false_positives, false_negatives, true_positives = (
        _confusion_counts(matrix)
    )
    return {
        "threshold": float(threshold),
        "false_positives": {
            "count": false_positives,
            "meaning": (
                "Non-failure records predicted as failures at the reporting threshold."
            ),
            "possible_consequence": (
                "These cases could create unnecessary inspection or maintenance "
                "work if alerts were acted on directly."
            ),
        },
        "false_negatives": {
            "count": false_negatives,
            "meaning": (
                "Failure records predicted as non-failures at the reporting threshold."
            ),
            "possible_consequence": (
                "These cases could represent missed maintenance opportunities "
                "if the model were used for alerting."
            ),
        },
        "limitations": [
            "Counts come from the held-out baseline test split only.",
            "The AI4I dataset is synthetic and may not match real equipment behavior.",
            (
                "No threshold is approved for production or autonomous "
                "maintenance decisions."
            ),
        ],
        "review_focus": (
            "Compare false positives and false negatives before changing the "
            "threshold; the preferred tradeoff depends on inspection capacity "
            "and the cost of missed failures."
        ),
        "supporting_counts": {
            "true_positives": true_positives,
            "true_negatives": true_negatives,
        },
    }


def positive_class_scores(estimator: Any, x_test: pd.DataFrame) -> np.ndarray:
    """Return model scores where larger values indicate class-1 failure risk."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        return np.asarray(probabilities)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        return np.asarray(scores, dtype=float)
    return np.asarray(estimator.predict(x_test), dtype=float)


def _confusion_counts(matrix: list[list[int]]) -> tuple[int, int, int, int]:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("Expected a 2x2 binary confusion matrix.")
    true_negatives = int(matrix[0][0])
    false_positives = int(matrix[0][1])
    false_negatives = int(matrix[1][0])
    true_positives = int(matrix[1][1])
    return true_negatives, false_positives, false_negatives, true_positives


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
