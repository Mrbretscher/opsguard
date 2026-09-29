"""Support utilities for the optional OpsGuard Streamlit app."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from opsguard.artifacts import METADATA_FILENAME, PIPELINE_FILENAME
from opsguard.config import MODELS_DIR, PROJECT_ROOT, REPORTS_DIR


@dataclass(frozen=True)
class LatestReport:
    """Loaded metrics report and the file it came from."""

    path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class MachineInputPreset:
    """Named machine-condition example for the Streamlit demo."""

    name: str
    description: str
    values: dict[str, object]


def load_latest_report(report_dir: Path = REPORTS_DIR) -> LatestReport:
    """Load the newest JSON metrics report from the report directory."""
    path = find_latest_report_path(report_dir)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in metrics report: {path}")
    return LatestReport(path=path, report=cast(dict[str, Any], loaded))


def find_latest_report_path(report_dir: Path = REPORTS_DIR) -> Path:
    """Return the newest JSON report path in a local report directory."""
    report_dir = Path(report_dir)
    if not report_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {report_dir}")

    candidates = [path for path in report_dir.glob("*.json") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No JSON metrics reports found in: {report_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def default_model_artifact_dir(
    report: Mapping[str, Any] | None,
    *,
    model_dir: Path = MODELS_DIR,
) -> Path:
    """Return the selected report artifact when present, otherwise latest artifact."""
    selected_artifact = _selected_report_artifact_dir(report)
    if selected_artifact is not None and is_model_artifact_dir(selected_artifact):
        return selected_artifact
    return find_latest_model_artifact_dir(model_dir)


def find_latest_model_artifact_dir(model_dir: Path = MODELS_DIR) -> Path:
    """Return the newest valid saved-model artifact directory."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifact directory not found: {model_dir}")

    candidates = [
        metadata_path.parent
        for metadata_path in model_dir.glob(f"*/{METADATA_FILENAME}")
        if is_model_artifact_dir(metadata_path.parent)
    ]
    if not candidates:
        raise FileNotFoundError(f"No saved model artifacts found in: {model_dir}")
    return max(
        candidates,
        key=lambda path: (path.joinpath(METADATA_FILENAME).stat().st_mtime, path.name),
    )


def is_model_artifact_dir(path: Path) -> bool:
    """Return whether a directory contains the required saved-model files."""
    path = Path(path)
    return (
        path.joinpath(METADATA_FILENAME).is_file()
        and path.joinpath(PIPELINE_FILENAME).is_file()
    )


def selected_model_name(report: Mapping[str, Any]) -> str | None:
    """Return the selected model name from a metrics report, when available."""
    selected_model = report.get("selected_model")
    if not isinstance(selected_model, Mapping):
        return None
    name = selected_model.get("name")
    return name if isinstance(name, str) else None


def selected_model_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return metrics for the selected model in a metrics report."""
    model_name = selected_model_name(report)
    metrics_by_model = report.get("metrics")
    if model_name is None or not isinstance(metrics_by_model, Mapping):
        return {}

    metrics = metrics_by_model.get(model_name)
    if not isinstance(metrics, Mapping):
        return {}
    return dict(metrics)


def selected_model_plot_paths(report: Mapping[str, Any]) -> dict[str, Path]:
    """Return existing plot paths for the selected model in a metrics report."""
    metrics = selected_model_metrics(report)
    plots = metrics.get("plots")
    if not isinstance(plots, Mapping):
        return {}

    existing_paths: dict[str, Path] = {}
    for name, raw_path in plots.items():
        if not isinstance(name, str) or not isinstance(raw_path, str):
            continue
        path = resolve_project_path(raw_path)
        if path.is_file():
            existing_paths[name] = path
    return existing_paths


def example_input_presets() -> list[MachineInputPreset]:
    """Return representative single-record inputs for the demo app."""
    return [
        MachineInputPreset(
            name="Nominal low-wear operation",
            description="Typical low-load example with little accumulated tool wear.",
            values={
                "Type": "L",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551.0,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0.0,
            },
        ),
        MachineInputPreset(
            name="High tool wear",
            description="Same product family, with substantially higher tool wear.",
            values={
                "Type": "L",
                "Air temperature [K]": 300.5,
                "Process temperature [K]": 310.1,
                "Rotational speed [rpm]": 1398.0,
                "Torque [Nm]": 46.3,
                "Tool wear [min]": 190.0,
            },
        ),
        MachineInputPreset(
            name="Thermal and torque stress",
            description="Higher process temperature and torque under slower rotation.",
            values={
                "Type": "H",
                "Air temperature [K]": 303.2,
                "Process temperature [K]": 312.8,
                "Rotational speed [rpm]": 1280.0,
                "Torque [Nm]": 62.0,
                "Tool wear [min]": 145.0,
            },
        ),
    ]


def threshold_tradeoff_rows(metrics: Mapping[str, Any]) -> list[dict[str, float]]:
    """Return numeric threshold tradeoff rows safe for display and charting."""
    threshold_table = metrics.get("threshold_table")
    if not isinstance(threshold_table, list):
        return []

    rows: list[dict[str, float]] = []
    for item in threshold_table:
        if not isinstance(item, Mapping):
            continue
        row = _numeric_threshold_row(item)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda row: row["threshold"])


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a possibly relative path from the repository root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _selected_report_artifact_dir(report: Mapping[str, Any] | None) -> Path | None:
    if report is None:
        return None

    selected_model = report.get("selected_model")
    if not isinstance(selected_model, Mapping):
        return None

    artifact_dir = selected_model.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    return resolve_project_path(artifact_dir)


def _numeric_threshold_row(item: Mapping[str, Any]) -> dict[str, float] | None:
    threshold = _finite_float(item.get("threshold"))
    precision = _finite_float(item.get("precision"))
    recall = _finite_float(item.get("recall"))
    if threshold is None or precision is None or recall is None:
        return None

    row = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
    }
    f1 = _finite_float(item.get("f1"))
    if f1 is None and precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    if f1 is not None:
        row["f1"] = f1
    return row


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value
