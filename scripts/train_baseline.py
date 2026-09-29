"""Thin command-line wrapper for the OpsGuard baseline workflow."""

import argparse
import json
from pathlib import Path

from opsguard.training import run_baseline_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OpsGuard Milestone 1B scikit-learn baselines."
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    run = run_baseline_training(
        data_path=args.data,
        report_dir=args.report_dir,
        model_dir=args.model_dir,
    )
    print(json.dumps(run.report, indent=2))
    print(f"Saved metrics to {run.metrics_path}")
    print(f"Saved selected model artifact to {run.model_artifact.artifact_dir}")


if __name__ == "__main__":
    main()
