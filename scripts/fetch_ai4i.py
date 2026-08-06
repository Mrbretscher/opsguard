#!/usr/bin/env python
"""Fetch and validate the UCI AI4I dataset."""

import argparse
from pathlib import Path

from opsguard.data import acquire_ai4i_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and validate the UCI AI4I 2020 dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/ai4i2020.csv"),
        help="Local CSV path for the downloaded raw dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if the local CSV already exists.",
    )
    args = parser.parse_args()

    result = acquire_ai4i_dataset(output_path=args.output, overwrite=args.overwrite)
    print(f"Saved AI4I dataset to {result.output_path}")
    print(result.summary.to_text())


if __name__ == "__main__":
    main()
