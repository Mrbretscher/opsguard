"""Feature and target construction for the AI4I binary baseline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from opsguard.config import (
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from opsguard.validation import validate_feature_columns

FORBIDDEN_FEATURE_COLUMNS = (*ID_COLUMNS, TARGET_COLUMN, *FAILURE_MODE_COLUMNS)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return leakage-controlled features and the binary failure target."""
    validate_feature_columns(
        FEATURE_COLUMNS,
        forbidden_columns=FORBIDDEN_FEATURE_COLUMNS,
    )
    features = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    target = frame.loc[:, TARGET_COLUMN].copy()
    return features, target


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a reusable preprocessing transformer for AI4I tabular features."""
    numeric_transformer: str | StandardScaler
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
