"""Schema and quality validation for the AI4I dataset."""

from collections.abc import Iterable

import pandas as pd

from opsguard.config import (
    BINARY_TARGET_COLUMNS,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


class DataValidationError(ValueError):
    """Raised when a dataset or feature matrix violates the AI4I contract."""


def validate_ai4i_frame(frame: pd.DataFrame) -> None:
    """Validate the raw AI4I dataframe schema and basic data quality."""
    if frame.empty:
        raise DataValidationError("AI4I dataframe is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise DataValidationError(
            "AI4I dataframe is missing required columns: " + ", ".join(missing_columns)
        )

    required_subset = frame.loc[:, list(REQUIRED_COLUMNS)]
    missing_counts = required_subset.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    if not columns_with_missing.empty:
        formatted = ", ".join(
            f"{column}={count}" for column, count in columns_with_missing.items()
        )
        raise DataValidationError(
            f"AI4I dataframe contains missing values: {formatted}"
        )

    for column in BINARY_TARGET_COLUMNS:
        values = set(frame[column].dropna().unique().tolist())
        if not values.issubset({0, 1}):
            formatted_values = ", ".join(str(value) for value in sorted(values))
            raise DataValidationError(
                f"Column {column!r} must be binary with values 0/1; "
                f"found {formatted_values}."
            )

    target_values = set(frame[TARGET_COLUMN].unique().tolist())
    if target_values != {0, 1}:
        raise DataValidationError(
            f"Target column {TARGET_COLUMN!r} must contain both classes 0 and 1."
        )

    non_numeric_columns = [
        column
        for column in FEATURE_COLUMNS
        if column != "Type" and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise DataValidationError(
            "Expected numeric feature columns are not numeric: "
            + ", ".join(non_numeric_columns)
        )


def validate_feature_columns(
    columns: Iterable[str],
    *,
    forbidden_columns: Iterable[str],
) -> None:
    """Ensure feature columns do not include target or leakage-prone columns."""
    column_set = set(columns)
    leakage_columns = sorted(column_set.intersection(forbidden_columns))
    if leakage_columns:
        raise DataValidationError(
            "Feature matrix includes forbidden leakage columns: "
            + ", ".join(leakage_columns)
        )
