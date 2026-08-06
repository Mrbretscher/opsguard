"""Plotting helpers for local baseline reports."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_confusion_matrix_plot(
    matrix: Sequence[Sequence[int]],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Save a small confusion-matrix plot for a binary classifier."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(4, 4))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], labels=["No failure", "Failure"])
    axis.set_yticks([0, 1], labels=["No failure", "Failure"])

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(values[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_precision_recall_plot(
    y_true: pd.Series,
    scores: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Save a precision-recall curve for the positive failure class."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision, linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
