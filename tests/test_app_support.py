import json
import os
from pathlib import Path

from opsguard.app_support import (
    default_model_artifact_dir,
    example_input_presets,
    find_latest_model_artifact_dir,
    load_latest_report,
    selected_model_metrics,
    selected_model_name,
    selected_model_plot_paths,
    threshold_tradeoff_rows,
)
from opsguard.artifacts import METADATA_FILENAME, PIPELINE_FILENAME
from opsguard.config import FEATURE_COLUMNS


def test_load_latest_report_uses_newest_json(tmp_path: Path) -> None:
    old_report = tmp_path / "old_metrics.json"
    new_report = tmp_path / "baseline_metrics.json"
    old_report.write_text(json.dumps({"task": "old"}), encoding="utf-8")
    new_report.write_text(json.dumps({"task": "new"}), encoding="utf-8")
    os.utime(old_report, (1, 1))
    os.utime(new_report, (2, 2))

    latest = load_latest_report(tmp_path)

    assert latest.path == new_report
    assert latest.report == {"task": "new"}


def test_default_model_artifact_dir_uses_report_selected_artifact(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifact_dir(tmp_path / "models" / "selected")
    report = {"selected_model": {"artifact_dir": str(artifact_dir)}}

    assert (
        default_model_artifact_dir(report, model_dir=tmp_path / "models")
        == artifact_dir
    )


def test_default_model_artifact_dir_falls_back_to_latest_valid_artifact(
    tmp_path: Path,
) -> None:
    old_artifact = _artifact_dir(tmp_path / "models" / "old")
    new_artifact = _artifact_dir(tmp_path / "models" / "new")
    os.utime(old_artifact / METADATA_FILENAME, (1, 1))
    os.utime(new_artifact / METADATA_FILENAME, (2, 2))

    assert find_latest_model_artifact_dir(tmp_path / "models") == new_artifact
    assert (
        default_model_artifact_dir(
            {"selected_model": {"artifact_dir": str(tmp_path / "missing")}},
            model_dir=tmp_path / "models",
        )
        == new_artifact
    )


def test_selected_model_report_helpers_return_metrics_and_existing_plots(
    tmp_path: Path,
) -> None:
    plot_path = tmp_path / "plots" / "selected_precision_recall.png"
    plot_path.parent.mkdir(parents=True)
    plot_path.write_bytes(b"plot")
    report = {
        "selected_model": {"name": "selected"},
        "metrics": {
            "selected": {
                "average_precision": 0.75,
                "plots": {
                    "precision_recall": str(plot_path),
                    "missing": str(tmp_path / "missing.png"),
                },
            }
        },
    }

    assert selected_model_name(report) == "selected"
    assert selected_model_metrics(report)["average_precision"] == 0.75
    assert selected_model_plot_paths(report) == {"precision_recall": plot_path}


def test_example_input_presets_cover_model_features() -> None:
    presets = example_input_presets()

    assert [preset.name for preset in presets] == [
        "Nominal low-wear operation",
        "High tool wear",
        "Thermal and torque stress",
    ]
    for preset in presets:
        assert set(preset.values) == set(FEATURE_COLUMNS)


def test_threshold_tradeoff_rows_support_current_evaluator_keys() -> None:
    metrics = {
        "threshold_table": [
            {
                "threshold": 0.6,
                "precision_failure": 0.5,
                "recall_failure": 0.25,
            },
            {
                "threshold": "bad",
                "precision_failure": 0.5,
                "recall_failure": 0.5,
            },
            {
                "threshold": 0.2,
                "precision_failure": 0.25,
                "recall_failure": 0.75,
                "f1_failure": 0.4,
            },
        ]
    }

    rows = threshold_tradeoff_rows(metrics)

    assert rows == [
        {"threshold": 0.2, "precision": 0.25, "recall": 0.75, "f1": 0.4},
        {"threshold": 0.6, "precision": 0.5, "recall": 0.25, "f1": 1 / 3},
    ]


def test_threshold_tradeoff_rows_support_legacy_metric_keys() -> None:
    metrics = {
        "threshold_table": [
            {"threshold": 0.6, "precision": 0.5, "recall": 0.25},
            {"threshold": "bad", "precision": 0.5, "recall": 0.5},
            {"threshold": 0.2, "precision": 0.25, "recall": 0.75, "f1": 0.4},
        ]
    }

    rows = threshold_tradeoff_rows(metrics)

    assert rows == [
        {"threshold": 0.2, "precision": 0.25, "recall": 0.75, "f1": 0.4},
        {"threshold": 0.6, "precision": 0.5, "recall": 0.25, "f1": 1 / 3},
    ]


def _artifact_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    path.joinpath(METADATA_FILENAME).write_text("{}", encoding="utf-8")
    path.joinpath(PIPELINE_FILENAME).write_bytes(b"pipeline")
    return path
