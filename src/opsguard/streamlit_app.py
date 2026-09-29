"""Optional Streamlit interface for local OpsGuard inference and reports."""

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

from opsguard.app_support import (
    default_model_artifact_dir,
    example_input_presets,
    load_latest_report,
    resolve_project_path,
    selected_model_metrics,
    selected_model_name,
    selected_model_plot_paths,
    threshold_tradeoff_rows,
)
from opsguard.artifacts import load_model_artifact
from opsguard.config import PROJECT_ROOT, REPORTS_DIR
from opsguard.inference import InferenceInputError, predict_failure


def main() -> None:
    """Render the Streamlit application."""
    st = _load_streamlit()
    st.set_page_config(page_title="OpsGuard", layout="wide")

    report_dir = resolve_project_path(
        st.sidebar.text_input("Report directory", value=str(REPORTS_DIR))
    )
    report_bundle = _load_report_for_ui(st, report_dir)
    report = report_bundle.report if report_bundle is not None else None

    artifact_dir = _artifact_path_input(st, report)
    artifact = _load_artifact_for_ui(st, artifact_dir)
    default_threshold = artifact.threshold if artifact is not None else 0.5

    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
        help=(
            "Exploratory classification threshold for this single prediction. "
            "It does not change the saved model artifact."
        ),
    )

    _render_first_screen(st, report_bundle, artifact, artifact_dir)

    prediction_tab, report_tab, limitations_tab = st.tabs(
        ["Risk check", "Evaluation", "Limitations"]
    )
    with prediction_tab:
        _render_prediction_form(st, artifact_dir, threshold)
    with report_tab:
        _render_report(st, report_bundle)
    with limitations_tab:
        _render_limitations(st)


def _load_streamlit() -> Any:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Streamlit is not installed. Install it with: pip install -e ".[app]"'
        ) from error
    return st


def _load_report_for_ui(st: Any, report_dir: Path) -> Any | None:
    try:
        return load_latest_report(report_dir)
    except (FileNotFoundError, ValueError) as error:
        st.sidebar.warning(str(error))
        return None


def _artifact_path_input(st: Any, report: Mapping[str, Any] | None) -> Path:
    try:
        default_artifact_dir = default_model_artifact_dir(report)
        default_value = str(default_artifact_dir)
    except FileNotFoundError:
        default_value = "models/<artifact-directory>"

    return resolve_project_path(
        st.sidebar.text_input("Model artifact directory", value=default_value)
    )


def _load_artifact_for_ui(st: Any, artifact_dir: Path) -> Any | None:
    try:
        artifact = load_model_artifact(artifact_dir)
    except (FileNotFoundError, TypeError, KeyError) as error:
        st.sidebar.error(str(error))
        return None

    metadata = artifact.metadata
    st.sidebar.metric("Artifact threshold", f"{artifact.threshold:.2f}")
    st.sidebar.caption(f"Model: {metadata.get('model_name', 'unknown')}")
    return artifact


def _render_first_screen(
    st: Any,
    report_bundle: Any | None,
    artifact: Any | None,
    artifact_dir: Path,
) -> None:
    st.title("OpsGuard")
    st.subheader("Predictive-maintenance baseline demo")
    st.caption(
        "A local portfolio app for exploring a saved scikit-learn baseline on "
        "synthetic AI4I machine-condition records."
    )

    status_columns = st.columns(3)
    status_columns[0].metric(
        "Selected model",
        str(artifact.metadata.get("model_name", "Unavailable"))
        if artifact is not None
        else "Unavailable",
    )
    status_columns[1].metric(
        "Saved threshold",
        f"{artifact.threshold:.0%}" if artifact is not None else "n/a",
    )
    status_columns[2].metric(
        "Latest report",
        report_bundle.path.name if report_bundle is not None else "Not found",
    )

    if artifact is None:
        st.warning(
            "No saved model artifact is loaded yet. Generate one with the baseline "
            "training workflow before running predictions."
        )
    else:
        st.info(f"Loaded artifact: `{artifact_dir}`")

    st.markdown(
        "This demo is intentionally scoped to baseline model behavior, evaluation "
        "evidence, and threshold tradeoffs. It is not an operational monitoring "
        "system and does not approve maintenance actions."
    )
    _render_doc_links(st)


def _render_prediction_form(st: Any, artifact_dir: Path, threshold: float) -> None:
    st.subheader("Machine risk check")
    presets = example_input_presets()
    preset_names = [preset.name for preset in presets]
    selected_preset = presets[
        preset_names.index(
            st.selectbox("Example input preset", options=preset_names, index=0)
        )
    ]
    st.caption(selected_preset.description)

    with st.form("machine-condition-form"):
        values = selected_preset.values
        type_options = ["L", "M", "H"]
        type_default = str(values["Type"])
        type_value = st.selectbox(
            "Type",
            options=type_options,
            index=type_options.index(type_default)
            if type_default in type_options
            else 0,
        )
        left, right = st.columns(2)
        with left:
            air_temperature = st.number_input(
                "Air temperature [K]",
                value=_preset_number(values, "Air temperature [K]"),
                step=0.1,
                format="%.1f",
            )
            rotational_speed = st.number_input(
                "Rotational speed [rpm]",
                value=int(_preset_number(values, "Rotational speed [rpm]")),
                step=1,
            )
            tool_wear = st.number_input(
                "Tool wear [min]",
                value=int(_preset_number(values, "Tool wear [min]")),
                step=1,
            )
        with right:
            process_temperature = st.number_input(
                "Process temperature [K]",
                value=_preset_number(values, "Process temperature [K]"),
                step=0.1,
                format="%.1f",
            )
            torque = st.number_input(
                "Torque [Nm]",
                value=_preset_number(values, "Torque [Nm]"),
                step=0.1,
                format="%.1f",
            )

        submitted = st.form_submit_button("Run risk check")

    if not submitted:
        return

    machine_condition = {
        "Type": str(type_value),
        "Air temperature [K]": float(air_temperature),
        "Process temperature [K]": float(process_temperature),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
    }
    try:
        result = predict_failure(
            machine_condition,
            artifact_dir,
            threshold=float(threshold),
        )
    except (FileNotFoundError, TypeError, KeyError, InferenceInputError) as error:
        st.error(str(error))
        return

    _render_risk_result_panel(st, result)


def _render_report(st: Any, report_bundle: Any | None) -> None:
    if report_bundle is None:
        st.info("Run the baseline training workflow to generate local report files.")
        return

    report = report_bundle.report
    model_name = selected_model_name(report)
    metrics = selected_model_metrics(report)

    st.subheader("Latest metrics")
    st.caption(str(report_bundle.path))
    if model_name is not None:
        st.write(f"Selected model: `{model_name}`")

    _render_metric_summary(st, metrics)
    _render_threshold_tradeoffs(st, metrics)
    _render_plots(st, report)


def _render_metric_summary(st: Any, metrics: Mapping[str, Any]) -> None:
    metric_specs = [
        ("Average precision", "average_precision"),
        ("Failure recall", "recall_failure"),
        ("Failure precision", "precision_failure"),
        ("Balanced accuracy", "balanced_accuracy"),
    ]
    columns = st.columns(len(metric_specs))
    for column, (label, key) in zip(columns, metric_specs, strict=True):
        column.metric(label, _format_metric(metrics.get(key)))


def _render_threshold_tradeoffs(st: Any, metrics: Mapping[str, Any]) -> None:
    threshold_rows = threshold_tradeoff_rows(metrics)
    if not threshold_rows:
        st.info("No threshold table was found for this report.")
        return

    st.subheader("Threshold tradeoffs")
    st.caption(
        "Lower thresholds generally catch more failures and flag more non-failures; "
        "higher thresholds do the opposite. These are report diagnostics, not "
        "approved operating points."
    )
    chart_frame = pd.DataFrame(threshold_rows)
    st.line_chart(
        chart_frame,
        x="threshold",
        y=[column for column in ("precision", "recall", "f1") if column in chart_frame],
        use_container_width=True,
    )
    st.dataframe(
        chart_frame.style.format("{:.3f}"),
        use_container_width=True,
        hide_index=True,
    )


def _render_plots(st: Any, report: Mapping[str, Any]) -> None:
    plot_paths = selected_model_plot_paths(report)
    if not plot_paths:
        st.info("No selected-model plots were found for this report.")
        return

    st.subheader("Diagnostic plots")
    columns = st.columns(len(plot_paths))
    for column, (name, path) in zip(columns, plot_paths.items(), strict=False):
        column.image(str(path), caption=name.replace("_", " ").title())


def _render_risk_result_panel(st: Any, result: Any) -> None:
    st.subheader("Risk result")
    decision = "Flagged for review" if result.prediction == 1 else "Not flagged"
    margin = result.failure_probability - result.threshold

    columns = st.columns(3)
    columns[0].metric("Estimated failure risk", f"{result.failure_probability:.1%}")
    columns[1].metric("Decision threshold", f"{result.threshold:.0%}")
    columns[2].metric("Decision", decision)

    message = (
        f"{result.explanation} The score is {abs(margin):.1%} "
        f"{'above' if margin >= 0 else 'below'} the selected threshold."
    )
    if result.prediction == 1:
        st.error(message)
    else:
        st.success(message)
    st.caption(
        "Interpret this as a model signal for demo analysis, not as a maintenance "
        "instruction."
    )


def _render_limitations(st: Any) -> None:
    st.subheader("Model limitations")
    st.markdown(
        """
- AI4I is synthetic, so performance may not transfer to real equipment, sensor
  drift, maintenance policies, or site-specific failure mechanisms.
- The failure class is rare; accuracy can look strong even when failure recall
  or precision is weak.
- The app scores one record at a time and does not include monitoring, alert
  routing, calibration review, or human approval workflows.
- The adjustable threshold is exploratory. It should be chosen only with
  inspection capacity, missed-failure cost, and validation evidence in view.
- Failure-mode target columns are excluded from the model inputs to avoid
  leakage into the binary failure prediction task.
"""
    )
    _render_doc_links(st)


def _render_doc_links(st: Any) -> None:
    docs = [
        ("Dataset documentation", PROJECT_ROOT / "docs" / "DATA_CARD.md"),
        ("Evaluation plan", PROJECT_ROOT / "docs" / "evaluation_plan.md"),
    ]
    links = " | ".join(
        f"[{label}]({path.as_uri()})" for label, path in docs if path.exists()
    )
    if links:
        st.markdown(links)


def _preset_number(values: Mapping[str, object], key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Preset field {key!r} must be numeric.")
    return float(value)


def _format_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "n/a"


if __name__ == "__main__":
    main()
