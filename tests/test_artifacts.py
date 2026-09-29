import json
from datetime import UTC, datetime

import numpy as np

from opsguard.artifacts import (
    METADATA_FILENAME,
    PIPELINE_FILENAME,
    build_feature_schema,
    load_model_artifact,
    save_model_artifact,
)
from opsguard.config import FEATURE_COLUMNS, RANDOM_SEED
from opsguard.evaluation import evaluate_binary_classifier, positive_class_scores
from opsguard.features import split_features_target
from opsguard.modeling import build_baseline_pipelines
from opsguard.training import run_baseline_training, train_and_evaluate_baselines


def test_model_artifact_round_trips_pipeline_preprocessing(
    baseline_ai4i_frame,
    tmp_path,
) -> None:
    features, target = split_features_target(baseline_ai4i_frame)
    pipeline = build_baseline_pipelines(random_state=RANDOM_SEED)["logistic_regression"]
    pipeline.fit(features, target)
    metrics = evaluate_binary_classifier(pipeline, features, target)

    saved = save_model_artifact(
        pipeline=pipeline,
        model_name="logistic_regression",
        metrics=metrics,
        feature_schema=build_feature_schema(baseline_ai4i_frame),
        threshold=0.5,
        random_seed=RANDOM_SEED,
        dataset_source="memory://baseline-fixture",
        output_dir=tmp_path,
        selection_metric="average_precision",
        trained_at=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
    )
    loaded = load_model_artifact(saved.artifact_dir)

    original_scores = positive_class_scores(pipeline, features)
    loaded_scores = positive_class_scores(loaded.pipeline, features)

    assert saved.pipeline_path.name == PIPELINE_FILENAME
    assert saved.metadata_path.name == METADATA_FILENAME
    np.testing.assert_allclose(loaded_scores, original_scores)
    assert loaded.threshold == 0.5
    assert loaded.feature_columns == list(FEATURE_COLUMNS)
    assert loaded.metadata["dataset_source"] == "memory://baseline-fixture"
    assert loaded.metadata["random_seed"] == RANDOM_SEED
    assert loaded.metadata["training_timestamp_utc"] == "2026-09-29T12:00:00Z"
    assert loaded.metadata["feature_schema"]["target_column"] == "Machine failure"
    assert "average_precision" in loaded.metadata["key_metrics"]
    assert (
        loaded.metadata["key_metrics"]["operating_threshold"]["selection_status"]
        == "not_operationally_approved"
    )


def test_training_persists_selected_model_artifact(
    baseline_ai4i_frame,
    tmp_path,
) -> None:
    report, _ = train_and_evaluate_baselines(
        baseline_ai4i_frame,
        report_dir=tmp_path / "reports",
        artifact_dir=tmp_path / "models",
        dataset_source="memory://baseline-fixture",
        test_size=0.25,
        save_plots=False,
    )

    selected = report["selected_model"]
    loaded = load_model_artifact(selected["artifact_dir"])
    metadata = json.loads(loaded.artifact_dir.joinpath(METADATA_FILENAME).read_text())

    assert loaded.artifact_dir.is_relative_to(tmp_path / "models")
    assert loaded.artifact_dir.joinpath(PIPELINE_FILENAME).exists()
    assert metadata["model_name"] == selected["name"]
    assert metadata["threshold"] == selected["threshold"]
    assert metadata["dataset_source"] == "memory://baseline-fixture"
    assert metadata["selection_metric"] == "average_precision"
    assert (
        selected["selection_metric_value"]
        == metadata["key_metrics"]["average_precision"]
    )
    assert (
        metadata["key_metrics"]["operating_threshold"]
        == selected["operating_threshold"]
    )


def test_run_baseline_training_returns_saved_model_artifact(
    baseline_ai4i_frame,
    tmp_path,
) -> None:
    data_path = tmp_path / "ai4i2020.csv"
    baseline_ai4i_frame.to_csv(data_path, index=False)

    run = run_baseline_training(
        data_path=data_path,
        report_dir=tmp_path / "reports",
        model_dir=tmp_path / "models",
        test_size=0.25,
    )
    loaded = load_model_artifact(run.model_artifact.artifact_dir)

    assert run.metrics_path.exists()
    assert run.model_artifact.pipeline_path.exists()
    assert run.model_artifact.metadata_path.exists()
    assert loaded.metadata["dataset_source"] == str(data_path)
