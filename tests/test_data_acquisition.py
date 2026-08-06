from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from opsguard.config import REQUIRED_COLUMNS
from opsguard.data import (
    AI4I_DATASET_NAME,
    acquire_ai4i_dataset,
    load_ai4i_csv,
)
from opsguard.data.ai4i import (
    build_ai4i_validation_summary,
    dataset_to_frame,
    normalize_ai4i_schema,
    validate_ai4i_dataset_identity,
)
from opsguard.validation import DataValidationError

UCI_API_COLUMN_NAMES = {
    "UDI": "UID",
    "Air temperature [K]": "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool wear",
}


def _dataset(frame: pd.DataFrame, *, name: str = AI4I_DATASET_NAME) -> Any:
    return SimpleNamespace(
        metadata={"name": name, "uci_id": "601"},
        data=SimpleNamespace(original=frame),
    )


def test_acquire_ai4i_dataset_fetches_validates_and_saves(
    tmp_path: Path,
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    output_path = tmp_path / "ai4i2020.csv"

    def fake_fetcher(*, id: int) -> Any:
        assert id == 601
        return _dataset(valid_ai4i_frame)

    result = acquire_ai4i_dataset(output_path, fetcher=fake_fetcher)

    assert result.output_path == output_path
    assert output_path.exists()
    assert result.summary.row_count == len(valid_ai4i_frame)
    assert result.summary.target_class_counts == {0: 4, 1: 2}


def test_acquire_ai4i_dataset_normalizes_mocked_ucimlrepo_response(
    tmp_path: Path,
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    output_path = tmp_path / "ai4i2020.csv"
    uci_api_frame = valid_ai4i_frame.rename(columns=UCI_API_COLUMN_NAMES)

    def fake_fetcher(*, id: int) -> Any:
        assert id == 601
        return _dataset(uci_api_frame)

    result = acquire_ai4i_dataset(output_path, fetcher=fake_fetcher)
    saved = pd.read_csv(output_path)

    assert result.summary.row_count == len(valid_ai4i_frame)
    assert list(saved.columns) == list(REQUIRED_COLUMNS)
    pd.testing.assert_frame_equal(saved, valid_ai4i_frame)


def test_acquire_ai4i_dataset_reuses_existing_local_file_without_fetch(
    tmp_path: Path,
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    output_path = tmp_path / "ai4i2020.csv"
    valid_ai4i_frame.to_csv(output_path, index=False)

    def fail_fetcher(*, id: int) -> Any:
        raise AssertionError("unit test should not fetch from UCI")

    result = acquire_ai4i_dataset(output_path, fetcher=fail_fetcher)

    assert result.summary.row_count == len(valid_ai4i_frame)


def test_validate_ai4i_dataset_identity_rejects_wrong_dataset(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    wrong_dataset = _dataset(valid_ai4i_frame, name="Iris")

    with pytest.raises(DataValidationError, match="Expected UCI dataset"):
        validate_ai4i_dataset_identity(wrong_dataset)


def test_dataset_to_frame_combines_ucimlrepo_parts(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            ids=valid_ai4i_frame.loc[:, ["UDI", "Product ID"]],
            features=valid_ai4i_frame.drop(
                columns=["UDI", "Product ID", "Machine failure"]
            ),
            targets=valid_ai4i_frame.loc[:, ["Machine failure"]],
        )
    )

    frame = dataset_to_frame(dataset)

    assert set(valid_ai4i_frame.columns).issubset(frame.columns)
    assert len(frame) == len(valid_ai4i_frame)


def test_normalize_ai4i_schema_maps_uci_api_column_names(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    uci_api_frame = valid_ai4i_frame.rename(columns=UCI_API_COLUMN_NAMES)

    normalized = normalize_ai4i_schema(uci_api_frame)

    assert list(normalized.columns) == list(REQUIRED_COLUMNS)
    pd.testing.assert_frame_equal(normalized, valid_ai4i_frame)


def test_normalize_ai4i_schema_preserves_canonical_column_names(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    normalized = normalize_ai4i_schema(valid_ai4i_frame)

    assert list(normalized.columns) == list(REQUIRED_COLUMNS)
    pd.testing.assert_frame_equal(normalized, valid_ai4i_frame)


def test_normalize_ai4i_schema_maps_uid_to_udi(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    uci_api_frame = valid_ai4i_frame.rename(columns={"UDI": "UID"})

    normalized = normalize_ai4i_schema(uci_api_frame)

    assert "UDI" in normalized.columns
    assert "UID" not in normalized.columns
    pd.testing.assert_series_equal(normalized["UDI"], valid_ai4i_frame["UDI"])


def test_normalize_ai4i_schema_outputs_expected_columns(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    normalized = normalize_ai4i_schema(
        valid_ai4i_frame.rename(columns=UCI_API_COLUMN_NAMES)
    )

    assert list(normalized.columns) == list(REQUIRED_COLUMNS)
    assert len(normalized.columns) == 14


def test_normalize_ai4i_schema_rejects_conflicting_alias_columns(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    invalid = valid_ai4i_frame.copy()
    invalid.insert(0, "UID", invalid["UDI"] + 100)

    with pytest.raises(DataValidationError, match="conflicting alias columns"):
        normalize_ai4i_schema(invalid)


def test_normalize_ai4i_schema_does_not_invent_missing_required_columns(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    invalid = valid_ai4i_frame.drop(columns=["Torque [Nm]"])
    normalized = normalize_ai4i_schema(invalid)

    assert "Torque [Nm]" not in normalized.columns
    with pytest.raises(DataValidationError, match="missing required columns"):
        build_ai4i_validation_summary(normalized)


def test_build_ai4i_validation_summary_reports_class_balance(
    valid_ai4i_frame: pd.DataFrame,
) -> None:
    summary = build_ai4i_validation_summary(valid_ai4i_frame)

    assert summary.target_class_counts == {0: 4, 1: 2}
    assert summary.target_class_proportions[1] == pytest.approx(2 / 6)
    assert "UDI" in summary.identifier_columns
    assert "UDI" not in summary.predictive_feature_columns


def test_load_ai4i_csv_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AI4I dataset not found"):
        load_ai4i_csv(tmp_path / "missing.csv")
