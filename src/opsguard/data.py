"""Dataset acquisition and loading helpers."""

from pathlib import Path
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

from opsguard.config import AI4I_DATASET_ID, RAW_DATA_PATH


def fetch_ai4i_dataset(
    output_path: Path = RAW_DATA_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Fetch the UCI AI4I 2020 dataset and save it as a local CSV file.

    The raw CSV is intentionally written under ``data/raw`` and ignored by Git.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    dataset = fetch_ucirepo(id=AI4I_DATASET_ID)
    frame = _dataset_to_frame(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def load_ai4i_csv(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a previously fetched AI4I CSV file."""
    path = Path(path)
    if not path.exists():
        message = (
            f"AI4I dataset not found at {path}. "
            "Run scripts/fetch_data.ps1 or `opsguard fetch-data` first."
        )
        raise FileNotFoundError(message)
    return pd.read_csv(path)


def _dataset_to_frame(dataset: Any) -> pd.DataFrame:
    data = dataset.data
    original = getattr(data, "original", None)
    if original is not None:
        return pd.DataFrame(original).copy()

    parts: list[pd.DataFrame] = []
    for attr_name in ("ids", "features", "targets"):
        value = getattr(data, attr_name, None)
        if value is not None:
            parts.append(pd.DataFrame(value))

    if not parts:
        raise ValueError("ucimlrepo returned no tabular data for AI4I dataset.")
    return pd.concat(parts, axis=1)
