import pytest

from opsguard.validation import (
    DataValidationError,
    validate_ai4i_frame,
    validate_feature_columns,
)


def test_validate_ai4i_frame_accepts_valid_frame(valid_ai4i_frame) -> None:
    validate_ai4i_frame(valid_ai4i_frame)


def test_validate_ai4i_frame_rejects_missing_columns(valid_ai4i_frame) -> None:
    invalid = valid_ai4i_frame.drop(columns=["Torque [Nm]"])

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_ai4i_frame(invalid)


def test_validate_ai4i_frame_rejects_non_binary_target(valid_ai4i_frame) -> None:
    invalid = valid_ai4i_frame.copy()
    invalid.loc[0, "Machine failure"] = 2

    with pytest.raises(DataValidationError, match="must be binary"):
        validate_ai4i_frame(invalid)


def test_validate_feature_columns_rejects_leakage_columns() -> None:
    with pytest.raises(DataValidationError, match="forbidden leakage columns"):
        validate_feature_columns(
            ["Type", "Machine failure"],
            forbidden_columns=["Machine failure"],
        )
