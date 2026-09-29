"""Small command-line entrypoints for the Milestone 1 workflow."""

import argparse
import json
from pathlib import Path

from opsguard.data import fetch_ai4i_dataset
from opsguard.training import run_baseline_training


def main() -> None:
    parser = argparse.ArgumentParser(description="OpsGuard AI4I baseline utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch AI4I dataset.")
    fetch_parser.add_argument("--output", type=Path, default=None)
    fetch_parser.add_argument("--overwrite", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate-baselines",
        help="Train and evaluate all Milestone 1 baselines.",
    )
    evaluate_parser.add_argument("--data", type=Path, default=None)
    evaluate_parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    evaluate_parser.add_argument("--model-dir", type=Path, default=Path("models"))

    args = parser.parse_args()

    if args.command == "fetch-data":
        output_path = fetch_ai4i_dataset(
            output_path=args.output if args.output else Path("data/raw/ai4i2020.csv"),
            overwrite=args.overwrite,
        )
        print(f"Fetched AI4I dataset to {output_path}")
        return

    if args.command == "evaluate-baselines":
        run = run_baseline_training(
            data_path=args.data if args.data else Path("data/raw/ai4i2020.csv"),
            report_dir=args.report_dir,
            model_dir=args.model_dir,
        )
        print(json.dumps(run.report, indent=2))
        print(f"Saved metrics to {run.metrics_path}")
        print(f"Saved selected model artifact to {run.model_artifact.artifact_dir}")


if __name__ == "__main__":
    main()
