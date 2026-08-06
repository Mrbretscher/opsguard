"""End-to-end baseline training and reporting workflow."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from opsguard.config import (
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    RANDOM_SEED,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from opsguard.data import load_ai4i_csv
from opsguard.evaluation import evaluate_binary_classifier, positive_class_scores
from opsguard.features import split_features_target
from opsguard.modeling import build_baseline_pipelines, make_train_test_split
from opsguard.plots import save_confusion_matrix_plot, save_precision_recall_plot
from opsguard.validation import validate_ai4i_frame


@dataclass(frozen=True)
class BaselineRun:
    """Paths and metrics produced by a baseline training run."""

    report: dict[str, Any]
    metrics_path: Path
    plot_paths: dict[str, dict[str, Path]]


def run_baseline_training(
    *,
    data_path: Path = RAW_DATA_PATH,
    report_dir: Path = REPORTS_DIR,
    random_state: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> BaselineRun:
    """Load the AI4I data, train baselines, and persist local report artifacts."""
    frame = load_ai4i_csv(data_path)
    report_dir = Path(report_dir)
    report, plot_paths = train_and_evaluate_baselines(
        frame,
        report_dir=report_dir,
        random_state=random_state,
        test_size=test_size,
        save_plots=True,
    )
    metrics_path = save_baseline_metrics(report, report_dir / "baseline_metrics.json")
    return BaselineRun(report=report, metrics_path=metrics_path, plot_paths=plot_paths)


def train_and_evaluate_baselines(
    frame: pd.DataFrame,
    *,
    report_dir: Path,
    random_state: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    threshold: float = 0.5,
    save_plots: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    """Train fixed baseline pipelines and evaluate them on an isolated test set."""
    validate_ai4i_frame(frame)
    features, target = split_features_target(frame)
    split = make_train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    report: dict[str, Any] = {
        "task": "binary_machine_failure_prediction",
        "target_column": TARGET_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "excluded_columns": [
            *ID_COLUMNS,
            TARGET_COLUMN,
            *FAILURE_MODE_COLUMNS,
        ],
        "random_seed": random_state,
        "test_size": test_size,
        "split_strategy": "stratified_train_test_split",
        "model_selection_note": (
            "Milestone 1B uses fixed baseline configurations. The final test set "
            "is used once for reporting, not for hyperparameter or threshold tuning."
        ),
        "class_distribution": {
            "full": class_distribution(target),
            "train": class_distribution(split.y_train),
            "test": class_distribution(split.y_test),
        },
        "metrics": {},
    }

    plot_paths: dict[str, dict[str, Path]] = {}
    for model_name, pipeline in build_baseline_pipelines(
        random_state=random_state
    ).items():
        pipeline.fit(split.x_train, split.y_train)
        metrics = evaluate_binary_classifier(
            pipeline,
            split.x_test,
            split.y_test,
            threshold=threshold,
        )
        report["metrics"][model_name] = metrics

        if save_plots:
            model_plot_paths = save_model_plots(
                model_name=model_name,
                metrics=metrics,
                estimator=pipeline,
                x_test=split.x_test,
                y_test=split.y_test,
                report_dir=report_dir,
            )
            plot_paths[model_name] = model_plot_paths
            metrics["plots"] = {
                plot_name: str(path) for plot_name, path in model_plot_paths.items()
            }

    return report, plot_paths


def class_distribution(target: pd.Series) -> dict[str, dict[str, float | int]]:
    """Return class counts and rates as JSON-safe values."""
    counts = target.value_counts().sort_index()
    total = int(len(target))
    distribution: dict[str, dict[str, float | int]] = {}
    for label, count in counts.items():
        distribution[str(int(cast(Any, label)))] = {
            "count": int(count),
            "rate": float(count / total) if total else 0.0,
        }
    return distribution


def save_model_plots(
    *,
    model_name: str,
    metrics: dict[str, Any],
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    report_dir: Path,
) -> dict[str, Path]:
    """Save model-specific diagnostic plots under the local report directory."""
    plots_dir = Path(report_dir) / "plots"
    scores = positive_class_scores(estimator, x_test)
    confusion_matrix_path = save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        plots_dir / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} confusion matrix",
    )
    precision_recall_path = save_precision_recall_plot(
        y_test,
        scores,
        plots_dir / f"{model_name}_precision_recall.png",
        title=f"{model_name} precision-recall curve",
    )
    return {
        "confusion_matrix": confusion_matrix_path,
        "precision_recall": precision_recall_path,
    }


def save_baseline_metrics(report: dict[str, Any], output_path: Path) -> Path:
    """Write the baseline report as deterministic, machine-readable JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path
