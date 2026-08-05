from opsguard.config import FAILURE_MODE_COLUMNS, ID_COLUMNS, TARGET_COLUMN
from opsguard.features import (
    FORBIDDEN_FEATURE_COLUMNS,
    build_preprocessor,
    split_features_target,
)


def test_split_features_target_excludes_leakage_columns(valid_ai4i_frame) -> None:
    features, target = split_features_target(valid_ai4i_frame)

    assert target.name == TARGET_COLUMN
    assert set(ID_COLUMNS).isdisjoint(features.columns)
    assert set(FAILURE_MODE_COLUMNS).isdisjoint(features.columns)
    assert TARGET_COLUMN not in features.columns
    assert set(FORBIDDEN_FEATURE_COLUMNS).isdisjoint(features.columns)


def test_build_preprocessor_has_expected_transformers() -> None:
    preprocessor = build_preprocessor()

    assert {name for name, _, _ in preprocessor.transformers} == {
        "categorical",
        "numeric",
    }
