"""Acquisition and validation helpers for the UCI AI4I dataset."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas.api.types import CategoricalDtype
from ucimlrepo import fetch_ucirepo

from opsguard.config import (
    AI4I_DATASET_ID,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)
from opsguard.validation import DataValidationError, validate_ai4i_frame

AI4I_DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"
AI4I_DATASET_URL = "https://archive.ics.uci.edu/dataset/601/ai4i"
AI4I_DATASET_DOI = "10.24432/C5HS5C"
AI4I_DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
AI4I_COLUMN_ALIASES = {
    "UID": "UDI",
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}

FetchUciRepo = Callable[..., Any]


@dataclass(frozen=True)
class Ai4iValidationSummary:
    """Concise summary of local AI4I validation results."""

    row_count: int
    column_count: int
    target_column: str
    identifier_columns: tuple[str, ...]
    predictive_feature_columns: tuple[str, ...]
    target_class_counts: dict[int, int]
    target_class_proportions: dict[int, float]

    def to_text(self) -> str:
        """Return a human-readable validation summary."""
        balance = ", ".join(
            f"{label}: {self.target_class_counts[label]} "
            f"({self.target_class_proportions[label]:.2%})"
            for label in sorted(self.target_class_counts)
        )
        identifiers = ", ".join(self.identifier_columns)
        features = ", ".join(self.predictive_feature_columns)
        return "\n".join(
            [
                "AI4I validation summary",
                f"- Rows: {self.row_count}",
                f"- Columns: {self.column_count}",
                f"- Target: {self.target_column}",
                f"- Target class balance: {balance}",
                f"- Identifier columns excluded from features: {identifiers}",
                f"- Predictive feature columns: {features}",
            ]
        )


@dataclass(frozen=True)
class Ai4iAcquisitionResult:
    """Result of acquiring and validating a local AI4I CSV."""

    output_path: Path
    summary: Ai4iValidationSummary


def acquire_ai4i_dataset(
    output_path: Path = RAW_DATA_PATH,
    *,
    overwrite: bool = False,
    fetcher: FetchUciRepo = fetch_ucirepo,
) -> Ai4iAcquisitionResult:
    """Fetch, validate, and save the UCI AI4I dataset.

    If ``output_path`` already exists and ``overwrite`` is false, the local file
    is loaded and validated without making a network call.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        frame = load_ai4i_csv(output_path)
        return Ai4iAcquisitionResult(
            output_path=output_path,
            summary=build_ai4i_validation_summary(frame),
        )

    dataset = fetcher(id=AI4I_DATASET_ID)
    validate_ai4i_dataset_identity(dataset)
    frame = dataset_to_frame(dataset)
    summary = build_ai4i_validation_summary(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return Ai4iAcquisitionResult(output_path=output_path, summary=summary)


def fetch_ai4i_dataset(
    output_path: Path = RAW_DATA_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Fetch the UCI AI4I 2020 dataset and save it as a local CSV file."""
    return acquire_ai4i_dataset(
        output_path=output_path,
        overwrite=overwrite,
    ).output_path


def load_ai4i_csv(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a previously fetched AI4I CSV file."""
    path = Path(path)
    if not path.exists():
        message = (
            f"AI4I dataset not found at {path}. "
            "Run `python scripts/fetch_ai4i.py` first."
        )
        raise FileNotFoundError(message)
    return pd.read_csv(path)


def validate_ai4i_dataset_identity(dataset: Any) -> None:
    """Ensure the returned UCI metadata matches the expected AI4I dataset."""
    metadata = getattr(dataset, "metadata", None)
    dataset_name = _metadata_value(metadata, "name")
    if dataset_name is None or AI4I_DATASET_NAME.lower() not in dataset_name.lower():
        raise DataValidationError(
            "Expected UCI dataset "
            f"{AI4I_DATASET_ID} ({AI4I_DATASET_NAME!r}); "
            f"received {dataset_name!r}."
        )

    uci_id = _metadata_value(metadata, "uci_id") or _metadata_value(metadata, "id")
    if uci_id is not None and int(uci_id) != AI4I_DATASET_ID:
        raise DataValidationError(
            f"Expected UCI dataset ID {AI4I_DATASET_ID}; received {uci_id}."
        )


def dataset_to_frame(dataset: Any) -> pd.DataFrame:
    """Convert a ``ucimlrepo`` dataset object into one raw dataframe."""
    data = dataset.data
    original = getattr(data, "original", None)
    if original is not None:
        return normalize_ai4i_schema(pd.DataFrame(original))

    parts: list[pd.DataFrame] = []
    for attr_name in ("ids", "features", "targets"):
        value = getattr(data, attr_name, None)
        if value is not None:
            parts.append(pd.DataFrame(value))

    if not parts:
        raise DataValidationError("ucimlrepo returned no tabular data for AI4I.")
    return normalize_ai4i_schema(pd.concat(parts, axis=1))


def normalize_ai4i_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI API AI4I column aliases to OpsGuard's canonical schema."""
    normalized = frame.copy()
    aliases_to_drop: list[str] = []
    rename_map: dict[str, str] = {}

    for alias, canonical in AI4I_COLUMN_ALIASES.items():
        has_alias = alias in normalized.columns
        has_canonical = canonical in normalized.columns
        if not has_alias:
            continue
        if has_canonical:
            if not _series_values_match(normalized[alias], normalized[canonical]):
                raise DataValidationError(
                    "AI4I dataframe contains conflicting alias columns: "
                    f"{alias!r} and {canonical!r}."
                )
            aliases_to_drop.append(alias)
            continue
        rename_map[alias] = canonical

    if aliases_to_drop:
        normalized = normalized.drop(columns=aliases_to_drop)
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    ordered_columns = [
        column for column in REQUIRED_COLUMNS if column in normalized.columns
    ]
    extra_columns = [
        column for column in normalized.columns if column not in ordered_columns
    ]
    return normalized.loc[:, [*ordered_columns, *extra_columns]]


def build_ai4i_validation_summary(frame: pd.DataFrame) -> Ai4iValidationSummary:
    """Validate AI4I data and return a concise local validation summary."""
    validate_ai4i_frame(frame)
    _validate_identifier_feature_separation()
    _validate_expected_dtypes(frame)

    target_counts_series = frame[TARGET_COLUMN].value_counts().sort_index()
    target_class_counts: dict[int, int] = {}
    for label, count in target_counts_series.items():
        target_class_counts[int(cast(Any, label))] = int(count)
    total = int(target_counts_series.sum())
    target_class_proportions = {
        label: count / total for label, count in target_class_counts.items()
    }

    present_identifier_columns = tuple(
        column for column in ID_COLUMNS if column in frame
    )
    return Ai4iValidationSummary(
        row_count=len(frame),
        column_count=len(frame.columns),
        target_column=TARGET_COLUMN,
        identifier_columns=present_identifier_columns,
        predictive_feature_columns=FEATURE_COLUMNS,
        target_class_counts=target_class_counts,
        target_class_proportions=target_class_proportions,
    )


def _validate_identifier_feature_separation() -> None:
    identifier_features = sorted(set(ID_COLUMNS).intersection(FEATURE_COLUMNS))
    if identifier_features:
        raise DataValidationError(
            "Identifier columns must not be predictive features: "
            + ", ".join(identifier_features)
        )


def _validate_expected_dtypes(frame: pd.DataFrame) -> None:
    if not pd.api.types.is_string_dtype(frame["Product ID"]):
        raise DataValidationError(
            "Column 'Product ID' must contain product identifiers."
        )
    if not pd.api.types.is_integer_dtype(frame["UDI"]):
        raise DataValidationError("Column 'UDI' must contain integer identifiers.")
    if not (
        pd.api.types.is_string_dtype(frame["Type"])
        or isinstance(frame["Type"].dtype, CategoricalDtype)
    ):
        raise DataValidationError("Column 'Type' must contain categorical labels.")

    non_numeric_columns = [
        column
        for column in (*NUMERIC_FEATURES, TARGET_COLUMN)
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise DataValidationError(
            "Expected numeric AI4I columns are not numeric: "
            + ", ".join(non_numeric_columns)
        )


def _series_values_match(left: pd.Series, right: pd.Series) -> bool:
    left_values = left.reset_index(drop=True)
    right_values = right.reset_index(drop=True)
    equal_values = left_values.eq(right_values)
    matching_missing_values = left_values.isna() & right_values.isna()
    return bool((equal_values | matching_missing_values).all())


def _metadata_value(metadata: Any, key: str) -> str | None:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
    else:
        value = getattr(metadata, key, None)
    if value is None:
        return None
    return str(value)
