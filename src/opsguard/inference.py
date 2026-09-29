"""Single-record inference helpers for saved OpsGuard model artifacts."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from opsguard.artifacts import LoadedModelArtifact, load_model_artifact
from opsguard.evaluation import positive_class_scores


class InferenceInputError(ValueError):
    """Raised when a machine-condition record cannot be scored safely."""


@dataclass(frozen=True)
class InferenceResult:
    """Prediction response for one machine-condition input."""

    failure_probability: float
    threshold: float
    prediction: int
    explanation: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable representation of the inference result."""
        return {
            "failure_probability": self.failure_probability,
            "threshold": self.threshold,
            "prediction": self.prediction,
            "explanation": self.explanation,
        }


def predict_failure(
    machine_condition: Mapping[str, object],
    artifact_dir: str | Path,
    *,
    threshold: float | None = None,
) -> InferenceResult:
    """Load a saved artifact and score one machine-condition record."""
    artifact = load_model_artifact(Path(artifact_dir))
    features = validate_machine_condition(machine_condition, artifact)
    scores = positive_class_scores(artifact.pipeline, features)
    if len(scores) != 1:
        raise RuntimeError(f"Expected one inference score, received {len(scores)}.")

    failure_probability = float(scores[0])
    decision_threshold = (
        artifact.threshold if threshold is None else _validate_threshold(threshold)
    )
    prediction = int(failure_probability >= decision_threshold)
    return InferenceResult(
        failure_probability=failure_probability,
        threshold=decision_threshold,
        prediction=prediction,
        explanation=_explain_prediction(
            failure_probability,
            decision_threshold,
            prediction,
        ),
    )


def validate_machine_condition(
    machine_condition: Mapping[str, object],
    artifact: LoadedModelArtifact,
) -> pd.DataFrame:
    """Validate one input record and return features in artifact training order."""
    feature_columns = artifact.feature_columns
    missing_columns = [
        column
        for column in feature_columns
        if column not in machine_condition or _is_missing(machine_condition[column])
    ]
    if missing_columns:
        raise InferenceInputError(
            "Machine-condition input is missing required fields: "
            + ", ".join(missing_columns)
        )

    roles = _feature_roles_by_column(artifact)
    validated: dict[str, object] = {}
    for column in feature_columns:
        value = machine_condition[column]
        role = roles.get(column)
        if role == "numeric":
            validated[column] = _validate_numeric(column, value)
        elif role == "categorical":
            validated[column] = _validate_categorical(column, value)
        else:
            validated[column] = value

    return pd.DataFrame([validated], columns=feature_columns)


def _validate_numeric(column: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InferenceInputError(
            f"Field {column!r} must be a finite numeric value; "
            f"received {type(value).__name__}."
        )

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise InferenceInputError(f"Field {column!r} must be a finite numeric value.")
    return numeric_value


def _validate_categorical(column: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceInputError(
            f"Field {column!r} must be a non-empty string category."
        )
    return value


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InferenceInputError(
            f"Decision threshold must be a finite numeric value; "
            f"received {type(value).__name__}."
        )

    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise InferenceInputError(
            "Decision threshold must be a finite value between 0.0 and 1.0."
        )
    return threshold


def _feature_roles_by_column(artifact: LoadedModelArtifact) -> dict[str, str]:
    input_columns = artifact.metadata.get("feature_schema", {}).get(
        "input_columns",
        [],
    )
    roles: dict[str, str] = {}
    for column in input_columns:
        if isinstance(column, Mapping):
            name = column.get("name")
            role = column.get("role")
            if isinstance(name, str) and isinstance(role, str):
                roles[name] = role
    return roles


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return bool(np.isnan(float(value)))
    return False


def _explain_prediction(
    failure_probability: float,
    threshold: float,
    prediction: int,
) -> str:
    if prediction == 1:
        return (
            "Predicted failure because the failure probability "
            f"{failure_probability:.3f} is at or above the threshold {threshold:.3f}."
        )
    return (
        "Predicted no failure because the failure probability "
        f"{failure_probability:.3f} is below the threshold {threshold:.3f}."
    )
