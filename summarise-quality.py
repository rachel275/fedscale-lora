#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise FedScale model-quality JSONL logs."
    )
    parser.add_argument(
        "experiment_directory",
        type=Path,
        help="Top-level model sweep directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def infer_model_method(
    run_name: str,
) -> tuple[str, str]:

    suffixes = [
        ("topk10", "topk10"),
        ("topk1", "topk1"),
        ("lora", "lora"),
        ("full", "full"),
    ]

    for suffix, method in suffixes:

        marker = f"_{suffix}"

        if run_name.endswith(marker):

            return (
                run_name[:-len(marker)],
                method,
            )

    return run_name, "unknown"


def load_quality_records(
    run_directory: Path,
) -> list[dict[str, Any]]:

    path = (
        run_directory
        / "quality.jsonl"
    )

    if not path.exists():
        return []

    records = []

    with path.open(
        encoding="utf-8"
    ) as source:

        for line_number, line in enumerate(
            source,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:

                raise RuntimeError(
                    f"Invalid JSON in "
                    f"{path}:{line_number}"
                ) from error

    return records


def main() -> None:

    args = parse_args()

    experiment_directory = (
        args.experiment_directory
        .expanduser()
        .resolve()
    )

    if not experiment_directory.is_dir():

        raise SystemExit(
            f"Not a directory: "
            f"{experiment_directory}"
        )

    rows = []

    for run_directory in sorted(
        experiment_directory.iterdir()
    ):

        if not run_directory.is_dir():
            continue

        records = load_quality_records(
            run_directory
        )

        if not records:

            print(
                "Warning: no quality "
                f"records in {run_directory}"
            )

            continue

        model_name, inferred_method = (
            infer_model_method(
                run_directory.name
            )
        )

        records.sort(
            key=lambda record: int(
                record.get(
                    "round",
                    0,
                )
            )
        )

        for record in records:

            rows.append(
                {
                    "run_name":
                        run_directory.name,

                    "model":
                        record.get(
                            "model",
                            model_name,
                        ),

                    "model_group":
                        model_name,

                    "method":
                        record.get(
                            "method",
                            inferred_method,
                        ),

                    "topk_ratio":
                        record.get(
                            "topk_ratio"
                        ),

                    "round":
                        record.get(
                            "round"
                        ),

                    "virtual_clock":
                        record.get(
                            "virtual_clock"
                        ),

                    "validation_loss":
                        record.get(
                            "validation_loss"
                        ),

                    "perplexity":
                        record.get(
                            "perplexity"
                        ),
                }
            )

    if not rows:

        raise SystemExit(
            "No quality records found."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Wrote {len(rows)} quality "
        f"records to {args.output}"
    )

    latest_by_run = {}

    for row in rows:

        run_name = row[
            "run_name"
        ]

        previous = (
            latest_by_run.get(
                run_name
            )
        )

        if (
            previous is None
            or int(
                row.get(
                    "round",
                    0,
                )
                or 0
            )
            > int(
                previous.get(
                    "round",
                    0,
                )
                or 0
            )
        ):

            latest_by_run[
                run_name
            ] = row

    for run_name, row in sorted(
        latest_by_run.items()
    ):

        print(
            f"{run_name:24s} "
            f"round={row['round']} "
            f"loss={row['validation_loss']} "
            f"perplexity={row['perplexity']}"
        )


if __name__ == "__main__":
    main()
