from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from opsguard.artifacts import build_feature_schema, save_model_artifact
from opsguard.config import FEATURE_COLUMNS, RANDOM_SEED
from opsguard.inference import InferenceInputError, predict_failure


class FixedProbabilityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> "FixedProbabilityClassifier":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive_probabilities = np.full(len(features), self.probability)
        negative_probabilities = 1.0 - positive_probabilities
        return np.column_stack([negative_probabilities, positive_probabilities])


def test_predict_failure_returns_single_record_response(
    baseline_ai4i_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    artifact_dir = _save_fixed_probability_artifact(
        baseline_ai4i_frame,
        tmp_path,
        probability=0.7,
        threshold=0.5,
    )

    result = predict_failure(_machine_condition(), artifact_dir)

    assert result.failure_probability == 0.7
    assert result.threshold == 0.5
    assert result.prediction == 1
    assert "at or above the threshold" in result.explanation
    assert result.to_dict() == {
        "failure_probability": 0.7,
        "threshold": 0.5,
        "prediction": 1,
        "explanation": result.explanation,
    }


def test_predict_failure_rejects_missing_fields(
    baseline_ai4i_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    artifact_dir = _save_fixed_probability_artifact(
        baseline_ai4i_frame,
        tmp_path,
        probability=0.7,
        threshold=0.5,
    )
    condition = _machine_condition()
    condition.pop("Torque [Nm]")

    with pytest.raises(InferenceInputError, match="Torque \\[Nm\\]"):
        predict_failure(condition, artifact_dir)


def test_predict_failure_rejects_invalid_numeric_types(
    baseline_ai4i_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    artifact_dir = _save_fixed_probability_artifact(
        baseline_ai4i_frame,
        tmp_path,
        probability=0.7,
        threshold=0.5,
    )
    condition = _machine_condition()
    condition["Rotational speed [rpm]"] = "fast"

    with pytest.raises(InferenceInputError, match="Rotational speed"):
        predict_failure(condition, artifact_dir)


@pytest.mark.parametrize(
    ("probability", "threshold", "expected_prediction"),
    [
        (0.49, 0.5, 0),
        (0.5, 0.5, 1),
    ],
)
def test_predict_failure_uses_artifact_threshold(
    baseline_ai4i_frame: pd.DataFrame,
    tmp_path: Path,
    probability: float,
    threshold: float,
    expected_prediction: int,
) -> None:
    artifact_dir = _save_fixed_probability_artifact(
        baseline_ai4i_frame,
        tmp_path / f"probability-{probability}",
        probability=probability,
        threshold=threshold,
    )

    result = predict_failure(_machine_condition(), artifact_dir)

    assert result.threshold == threshold
    assert result.prediction == expected_prediction


def test_predict_failure_accepts_threshold_override(
    baseline_ai4i_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    artifact_dir = _save_fixed_probability_artifact(
        baseline_ai4i_frame,
        tmp_path,
        probability=0.7,
        threshold=0.8,
    )

    result = predict_failure(
        _machine_condition(),
        artifact_dir,
        threshold=0.65,
    )

    assert result.threshold == 0.65
    assert result.prediction == 1


def test_predict_failure_rejects_invalid_threshold_override(
    baseline_ai4i_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    artifact_dir = _save_fixed_probability_artifact(
        baseline_ai4i_frame,
        tmp_path,
        probability=0.7,
        threshold=0.5,
    )

    with pytest.raises(InferenceInputError, match="between 0.0 and 1.0"):
        predict_failure(_machine_condition(), artifact_dir, threshold=1.5)


def _machine_condition() -> dict[str, object]:
    return {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }


def _save_fixed_probability_artifact(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    probability: float,
    threshold: float,
) -> Path:
    features = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame["Machine failure"]
    pipeline = Pipeline(
        [
            ("model", FixedProbabilityClassifier(probability=probability)),
        ]
    )
    pipeline.fit(features, target)
    saved = save_model_artifact(
        pipeline=pipeline,
        model_name="fixed_probability",
        metrics={"average_precision": probability},
        feature_schema=build_feature_schema(frame),
        threshold=threshold,
        random_seed=RANDOM_SEED,
        dataset_source="memory://inference-fixture",
        output_dir=output_dir,
        selection_metric="average_precision",
        trained_at=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
    )
    return saved.artifact_dir
