import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from opsguard.config import RANDOM_SEED
from opsguard.evaluation import evaluate_binary_classifier
from opsguard.features import split_features_target
from opsguard.modeling import build_baseline_pipelines, make_train_test_split
from opsguard.training import (
    save_baseline_metrics,
    train_and_evaluate_baselines,
)


def test_baseline_builders_return_pipelines() -> None:
    pipelines = build_baseline_pipelines()

    assert set(pipelines) == {
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
    }
    assert all(isinstance(pipeline, Pipeline) for pipeline in pipelines.values())


def test_baseline_pipelines_start_with_column_transformer() -> None:
    pipelines = build_baseline_pipelines()

    for pipeline in pipelines.values():
        preprocessor = pipeline.named_steps["preprocessor"]
        assert isinstance(preprocessor, ColumnTransformer)
        assert pipeline.steps[-1][0] == "model"


def test_evaluation_handles_zero_positive_predictions(valid_ai4i_frame) -> None:
    features, target = split_features_target(valid_ai4i_frame)
    pipeline = build_baseline_pipelines()["dummy_most_frequent"]
    pipeline.fit(features, target)

    metrics = evaluate_binary_classifier(pipeline, features, target)

    assert metrics["precision_failure"] == 0.0
    assert {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    }.issubset(metrics)


def test_train_test_split_is_stratified() -> None:
    features = pd.DataFrame(
        {
            "Type": ["L"] * 20,
            "Air temperature [K]": range(20),
            "Process temperature [K]": range(20),
            "Rotational speed [rpm]": range(20),
            "Torque [Nm]": range(20),
            "Tool wear [min]": range(20),
        }
    )
    target = pd.Series([0] * 10 + [1] * 10, name="Machine failure")

    split = make_train_test_split(features, target, test_size=0.2)

    assert set(split.y_train.unique()) == {0, 1}
    assert set(split.y_test.unique()) == {0, 1}


def test_train_test_split_is_deterministic_and_disjoint(baseline_ai4i_frame) -> None:
    features, target = split_features_target(baseline_ai4i_frame)

    first = make_train_test_split(features, target, test_size=0.25)
    second = make_train_test_split(features, target, test_size=0.25)

    assert first.x_train.index.equals(second.x_train.index)
    assert first.x_test.index.equals(second.x_test.index)
    assert set(first.x_train.index).isdisjoint(first.x_test.index)


def test_preprocessing_is_fit_after_train_test_split(baseline_ai4i_frame) -> None:
    features, target = split_features_target(baseline_ai4i_frame)
    split = make_train_test_split(features, target, test_size=0.25)
    split.x_test.loc[split.x_test.index[0], "Type"] = "TEST_ONLY_TYPE"

    pipeline = build_baseline_pipelines(random_state=RANDOM_SEED)["logistic_regression"]
    pipeline.fit(split.x_train, split.y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"]
    learned_categories = set(encoder.categories_[0].tolist())

    assert "TEST_ONLY_TYPE" not in learned_categories


def test_training_report_is_deterministic(baseline_ai4i_frame, tmp_path) -> None:
    first_report, _ = train_and_evaluate_baselines(
        baseline_ai4i_frame,
        report_dir=tmp_path,
        test_size=0.25,
        save_plots=False,
    )
    second_report, _ = train_and_evaluate_baselines(
        baseline_ai4i_frame,
        report_dir=tmp_path,
        test_size=0.25,
        save_plots=False,
    )

    assert first_report == second_report


def test_training_saves_metrics_and_plots(baseline_ai4i_frame, tmp_path) -> None:
    report, plot_paths = train_and_evaluate_baselines(
        baseline_ai4i_frame,
        report_dir=tmp_path,
        test_size=0.25,
    )
    metrics_path = save_baseline_metrics(report, tmp_path / "baseline_metrics.json")

    saved_report = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert saved_report["random_seed"] == RANDOM_SEED
    assert saved_report["class_distribution"]["full"]["1"]["count"] == 4
    assert set(plot_paths) == set(build_baseline_pipelines())
    for model_plot_paths in plot_paths.values():
        assert model_plot_paths["confusion_matrix"].exists()
        assert model_plot_paths["precision_recall"].exists()
