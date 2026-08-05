import pandas as pd
from sklearn.pipeline import Pipeline

from opsguard.evaluation import evaluate_binary_classifier
from opsguard.features import split_features_target
from opsguard.modeling import build_baseline_pipelines, make_train_test_split


def test_baseline_builders_return_pipelines() -> None:
    pipelines = build_baseline_pipelines()

    assert set(pipelines) == {
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
    }
    assert all(isinstance(pipeline, Pipeline) for pipeline in pipelines.values())


def test_evaluation_handles_zero_positive_predictions(valid_ai4i_frame) -> None:
    features, target = split_features_target(valid_ai4i_frame)
    pipeline = build_baseline_pipelines()["dummy_most_frequent"]
    pipeline.fit(features, target)

    metrics = evaluate_binary_classifier(pipeline, features, target)

    assert metrics["precision_failure"] == 0.0
    assert "average_precision" in metrics
    assert "confusion_matrix" in metrics


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
