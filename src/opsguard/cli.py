"""Small command-line entrypoints for the Milestone 1 workflow."""

import argparse
import json
from pathlib import Path

from opsguard.data import fetch_ai4i_dataset, load_ai4i_csv
from opsguard.evaluation import evaluate_binary_classifier
from opsguard.features import split_features_target
from opsguard.modeling import build_baseline_pipelines, make_train_test_split
from opsguard.validation import validate_ai4i_frame


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

    args = parser.parse_args()

    if args.command == "fetch-data":
        output_path = fetch_ai4i_dataset(
            output_path=args.output if args.output else Path("data/raw/ai4i2020.csv"),
            overwrite=args.overwrite,
        )
        print(f"Fetched AI4I dataset to {output_path}")
        return

    if args.command == "evaluate-baselines":
        frame = load_ai4i_csv(args.data if args.data else Path("data/raw/ai4i2020.csv"))
        validate_ai4i_frame(frame)
        features, target = split_features_target(frame)
        split = make_train_test_split(features, target)

        results = {}
        for name, pipeline in build_baseline_pipelines().items():
            pipeline.fit(split.x_train, split.y_train)
            results[name] = evaluate_binary_classifier(
                pipeline,
                split.x_test,
                split.y_test,
            )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
