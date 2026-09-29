"""Model artifact persistence helpers for trained OpsGuard pipelines."""

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from opsguard.config import (
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

ARTIFACT_VERSION = 1
PIPELINE_FILENAME = "pipeline.pkl"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class SavedModelArtifact:
    """Filesystem paths for a persisted trained model artifact."""

    artifact_dir: Path
    pipeline_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class LoadedModelArtifact:
    """Loaded model pipeline and metadata needed for inference."""

    pipeline: Pipeline
    metadata: dict[str, Any]
    artifact_dir: Path

    @property
    def threshold(self) -> float:
        """Return the classification threshold stored with the artifact."""
        return float(self.metadata["threshold"])

    @property
    def feature_columns(self) -> list[str]:
        """Return the model input columns in training order."""
        return [
            str(column["name"])
            for column in self.metadata["feature_schema"]["input_columns"]
        ]


def build_feature_schema(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe the model inputs and leakage-excluded columns from training data."""
    return {
        "target_column": TARGET_COLUMN,
        "input_columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "role": _feature_role(column),
            }
            for column in FEATURE_COLUMNS
        ],
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "excluded_columns": [
            *ID_COLUMNS,
            TARGET_COLUMN,
            *FAILURE_MODE_COLUMNS,
        ],
    }


def save_model_artifact(
    *,
    pipeline: Pipeline,
    model_name: str,
    metrics: Mapping[str, Any],
    feature_schema: Mapping[str, Any],
    threshold: float,
    random_seed: int,
    dataset_source: str | Path,
    output_dir: Path,
    selection_metric: str,
    trained_at: datetime | None = None,
) -> SavedModelArtifact:
    """Persist a fitted pipeline and JSON metadata under an artifact directory."""
    timestamp = _utc_timestamp(trained_at)
    artifact_dir = Path(output_dir) / _artifact_dir_name(model_name, timestamp)
    artifact_dir.mkdir(parents=True, exist_ok=False)

    pipeline_path = artifact_dir / PIPELINE_FILENAME
    with pipeline_path.open("wb") as file:
        pickle.dump(pipeline, file)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": model_name,
        "pipeline_filename": PIPELINE_FILENAME,
        "training_timestamp_utc": timestamp,
        "dataset_source": str(dataset_source),
        "feature_schema": dict(feature_schema),
        "threshold": float(threshold),
        "random_seed": int(random_seed),
        "selection_metric": selection_metric,
        "key_metrics": _key_metrics(metrics),
    }
    metadata_path = artifact_dir / METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    return SavedModelArtifact(
        artifact_dir=artifact_dir,
        pipeline_path=pipeline_path,
        metadata_path=metadata_path,
    )


def load_model_artifact(artifact_dir: Path) -> LoadedModelArtifact:
    """Load a persisted model pipeline and its metadata from an artifact directory."""
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pipeline_path = artifact_dir / str(
        metadata.get("pipeline_filename", PIPELINE_FILENAME)
    )
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model pipeline not found: {pipeline_path}")

    with pipeline_path.open("rb") as file:
        pipeline = pickle.load(file)

    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline in {pipeline_path}")

    return LoadedModelArtifact(
        pipeline=pipeline,
        metadata=metadata,
        artifact_dir=artifact_dir,
    )


def load_model_pipeline(artifact_dir: Path) -> Pipeline:
    """Load only the fitted pipeline from a persisted model artifact."""
    return load_model_artifact(artifact_dir).pipeline


def _feature_role(column: str) -> str:
    if column in CATEGORICAL_FEATURES:
        return "categorical"
    if column in NUMERIC_FEATURES:
        return "numeric"
    return "unknown"


def _utc_timestamp(value: datetime | None) -> str:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _artifact_dir_name(model_name: str, timestamp: str) -> str:
    safe_timestamp = timestamp.replace(":", "").replace("-", "")
    safe_timestamp = safe_timestamp.replace(".", "")
    safe_model_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in model_name
    )
    return f"{safe_timestamp}_{safe_model_name}"


def _key_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "threshold",
        "accuracy",
        "precision_failure",
        "recall_failure",
        "f1_failure",
        "average_precision",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "confusion_matrix",
        "operating_threshold",
        "confusion_matrix_interpretation",
        "error_analysis_notes",
    )
    return {name: metrics[name] for name in names if name in metrics}
