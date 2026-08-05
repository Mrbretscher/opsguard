"""Project constants for the AI4I baseline workflow."""

from pathlib import Path

AI4I_DATASET_ID = 601
RANDOM_SEED = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "ai4i2020.csv"

ID_COLUMNS = ("UDI", "UID", "Product ID")
CATEGORICAL_FEATURES = ("Type",)
NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")

REQUIRED_COLUMNS = (
    "UDI",
    "Product ID",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
)
BINARY_TARGET_COLUMNS = (TARGET_COLUMN, *FAILURE_MODE_COLUMNS)
