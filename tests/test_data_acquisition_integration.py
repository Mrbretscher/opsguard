import os
from pathlib import Path

import pytest

from opsguard.data import acquire_ai4i_dataset


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("OPSGUARD_RUN_INTEGRATION") != "1",
    reason="set OPSGUARD_RUN_INTEGRATION=1 to fetch the real UCI dataset",
)
def test_fetch_ai4i_dataset_from_uci(tmp_path: Path) -> None:
    result = acquire_ai4i_dataset(tmp_path / "ai4i2020.csv", overwrite=True)

    assert result.output_path.exists()
    assert result.summary.row_count > 0
    assert result.summary.target_class_counts.keys() == {0, 1}
