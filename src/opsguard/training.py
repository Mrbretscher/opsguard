"""End-to-end baseline training and reporting workflow."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from opsguard.artifacts import (
    SavedModelArtifact,
    build_feature_schema,
    save_model_artifact,
)
from opsguard.config import (
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    MODELS_DIR,
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
    model_artifact: SavedModelArtifact


def run_baseline_training(
    *,
    data_path: Path = RAW_DATA_PATH,
    report_dir: Path = REPORTS_DIR,
    model_dir: Path = MODELS_DIR,
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
        artifact_dir=model_dir,
        dataset_source=data_path,
        save_plots=True,
    )
    selected_model = report["selected_model"]
    model_artifact = SavedModelArtifact(
        artifact_dir=Path(selected_model["artifact_dir"]),
        pipeline_path=Path(selected_model["artifact_dir"]) / "pipeline.pkl",
        metadata_path=Path(selected_model["artifact_metadata"]),
    )
    metrics_path = save_baseline_metrics(report, report_dir / "baseline_metrics.json")
    return BaselineRun(
        report=report,
        metrics_path=metrics_path,
        plot_paths=plot_paths,
        model_artifact=model_artifact,
    )


def train_and_evaluate_baselines(
    frame: pd.DataFrame,
    *,
    report_dir: Path,
    random_state: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    threshold: float = 0.5,
    artifact_dir: Path | None = None,
    dataset_source: str | Path | None = None,
    selection_metric: str = "average_precision",
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
        "model_selection_metric": selection_metric,
        "model_selection_note": (
            "Milestone 1B uses fixed baseline configurations. The saved artifact "
            "is the baseline with the highest report average precision and is "
            "for reproducible inference experiments, not production-ready model "
            "selection or threshold tuning."
        ),
        "class_distribution": {
            "full": class_distribution(target),
            "train": class_distribution(split.y_train),
            "test": class_distribution(split.y_test),
        },
        "metrics": {},
    }

    plot_paths: dict[str, dict[str, Path]] = {}
    fitted_pipelines: dict[str, Any] = {}
    for model_name, pipeline in build_baseline_pipelines(
        random_state=random_state
    ).items():
        pipeline.fit(split.x_train, split.y_train)
        fitted_pipelines[model_name] = pipeline
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

    selected_model_name = select_model_name(report["metrics"], metric=selection_metric)
    selected_metrics = report["metrics"][selected_model_name]
    report["selected_model"] = {
        "name": selected_model_name,
        "selection_metric": selection_metric,
        "selection_metric_value": selected_metrics[selection_metric],
        "threshold": threshold,
        "operating_threshold": selected_metrics["operating_threshold"],
        "confusion_matrix_interpretation": selected_metrics[
            "confusion_matrix_interpretation"
        ],
        "error_analysis_notes": selected_metrics["error_analysis_notes"],
    }

    if artifact_dir is not None:
        artifact_dataset_source = (
            dataset_source if dataset_source is not None else "in_memory"
        )
        artifact = save_model_artifact(
            pipeline=fitted_pipelines[selected_model_name],
            model_name=selected_model_name,
            metrics=selected_metrics,
            feature_schema=build_feature_schema(frame),
            threshold=threshold,
            random_seed=random_state,
            dataset_source=artifact_dataset_source,
            output_dir=artifact_dir,
            selection_metric=selection_metric,
        )
        report["selected_model"]["artifact_dir"] = str(artifact.artifact_dir)
        report["selected_model"]["artifact_metadata"] = str(artifact.metadata_path)

    return report, plot_paths


def select_model_name(
    metrics_by_model: dict[str, dict[str, Any]],
    *,
    metric: str,
) -> str:
    """Return the model with the largest value for the requested metric."""
    if not metrics_by_model:
        raise ValueError("Cannot select a model without metrics.")

    missing = [
        name for name, metrics in metrics_by_model.items() if metric not in metrics
    ]
    if missing:
        raise KeyError(f"Selection metric {metric!r} missing for models: {missing}")

    return max(metrics_by_model, key=lambda name: metrics_by_model[name][metric])


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
