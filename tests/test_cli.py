import json
import sys
from pathlib import Path
from typing import Any

from opsguard import cli
from opsguard.artifacts import SavedModelArtifact
from opsguard.training import BaselineRun


def test_fetch_data_cli_forwards_output_and_overwrite(
    monkeypatch: Any,
    capsys: Any,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "raw" / "ai4i2020.csv"
    calls: list[dict[str, Any]] = []

    def fake_fetch_ai4i_dataset(*, output_path: Path, overwrite: bool) -> Path:
        calls.append({"output_path": output_path, "overwrite": overwrite})
        return output_path

    monkeypatch.setattr(cli, "fetch_ai4i_dataset", fake_fetch_ai4i_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opsguard",
            "fetch-data",
            "--output",
            str(output_path),
            "--overwrite",
        ],
    )

    cli.main()

    assert calls == [{"output_path": output_path, "overwrite": True}]
    assert capsys.readouterr().out == f"Fetched AI4I dataset to {output_path}\n"


def test_fetch_data_cli_uses_default_output_without_overwrite(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fetch_ai4i_dataset(*, output_path: Path, overwrite: bool) -> Path:
        calls.append({"output_path": output_path, "overwrite": overwrite})
        return output_path

    monkeypatch.setattr(cli, "fetch_ai4i_dataset", fake_fetch_ai4i_dataset)
    monkeypatch.setattr(sys, "argv", ["opsguard", "fetch-data"])

    cli.main()

    assert calls == [{"output_path": Path("data/raw/ai4i2020.csv"), "overwrite": False}]


def test_evaluate_baselines_cli_forwards_paths_and_prints_report(
    monkeypatch: Any,
    capsys: Any,
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "raw" / "ai4i2020.csv"
    report_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    metrics_path = report_dir / "baseline_metrics.json"
    artifact = SavedModelArtifact(
        artifact_dir=model_dir / "artifact",
        pipeline_path=model_dir / "artifact" / "pipeline.pkl",
        metadata_path=model_dir / "artifact" / "metadata.json",
    )
    report = {
        "task": "binary_machine_failure_prediction",
        "metrics": {"dummy_most_frequent": {"accuracy": 0.75}},
    }
    calls: list[dict[str, Path]] = []

    def fake_run_baseline_training(
        *,
        data_path: Path,
        report_dir: Path,
        model_dir: Path,
    ) -> BaselineRun:
        calls.append(
            {"data_path": data_path, "report_dir": report_dir, "model_dir": model_dir}
        )
        return BaselineRun(
            report=report,
            metrics_path=metrics_path,
            plot_paths={},
            model_artifact=artifact,
        )

    monkeypatch.setattr(cli, "run_baseline_training", fake_run_baseline_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opsguard",
            "evaluate-baselines",
            "--data",
            str(data_path),
            "--report-dir",
            str(report_dir),
            "--model-dir",
            str(model_dir),
        ],
    )

    cli.main()

    captured = capsys.readouterr().out
    json_output, saved_message, artifact_message = captured.rsplit("\n", maxsplit=3)[:3]
    assert calls == [
        {"data_path": data_path, "report_dir": report_dir, "model_dir": model_dir}
    ]
    assert json.loads(json_output) == report
    assert saved_message == f"Saved metrics to {metrics_path}"
    assert (
        artifact_message == f"Saved selected model artifact to {artifact.artifact_dir}"
    )


def test_evaluate_baselines_cli_uses_default_paths(monkeypatch: Any) -> None:
    calls: list[dict[str, Path]] = []

    def fake_run_baseline_training(
        *,
        data_path: Path,
        report_dir: Path,
        model_dir: Path,
    ) -> BaselineRun:
        calls.append(
            {"data_path": data_path, "report_dir": report_dir, "model_dir": model_dir}
        )
        return BaselineRun(
            report={},
            metrics_path=report_dir / "baseline_metrics.json",
            plot_paths={},
            model_artifact=SavedModelArtifact(
                artifact_dir=model_dir / "artifact",
                pipeline_path=model_dir / "artifact" / "pipeline.pkl",
                metadata_path=model_dir / "artifact" / "metadata.json",
            ),
        )

    monkeypatch.setattr(cli, "run_baseline_training", fake_run_baseline_training)
    monkeypatch.setattr(sys, "argv", ["opsguard", "evaluate-baselines"])

    cli.main()

    assert calls == [
        {
            "data_path": Path("data/raw/ai4i2020.csv"),
            "report_dir": Path("reports"),
            "model_dir": Path("models"),
        }
    ]
