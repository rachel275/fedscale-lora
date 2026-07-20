#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise FedScale GEMM compute traces."
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


def infer_model_method(run_name: str) -> tuple[str, str]:
    suffixes = [
        ("topk10", "topk10"),
        ("topk1", "topk1"),
        ("lora", "lora"),
        ("full", "full"),
    ]

    for suffix, method in suffixes:
        marker = f"_{suffix}"
        if run_name.endswith(marker):
            return run_name[:-len(marker)], method

    return run_name, "unknown"


def load_gemm_records(run_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    pattern = str(
        run_directory / "gemm-executor-*.jsonl"
    )

    for filename in sorted(glob.glob(pattern)):
        with open(filename, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid JSON in {filename}:{line_number}"
                    ) from error

    return records


def load_communication_records(
    run_directory: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    pattern = str(
        run_directory / "communication-executor-*.jsonl"
    )

    for filename in sorted(glob.glob(pattern)):
        with open(filename, encoding="utf-8") as source:
            for line in source:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

    return records


def record_flops(record: dict[str, Any]) -> int:
    if "flops" in record:
        return int(record["flops"])

    m = int(record["gemm_M"])
    n = int(record["gemm_N"])
    k = int(record["gemm_K"])

    return 2 * m * n * k


def summarise_run(
    run_directory: Path,
) -> dict[str, Any] | None:

    gemm_records = load_gemm_records(
        run_directory
    )

    if not gemm_records:
        return None

    # Only actual FL client training records.
    gemm_records = [
        record
        for record in gemm_records
        if record.get("client_id") is not None
    ]

    if not gemm_records:
        return None

    communication_records = (
        load_communication_records(
            run_directory
        )
    )

    upload_messages = sum(
        1
        for record in communication_records
        if record.get("direction") == "upload"
    )

    # One upload corresponds to one completed client update.
    completed_updates = max(
        upload_messages,
        1,
    )

    total_flops = sum(
        record_flops(record)
        for record in gemm_records
    )

    phase_flops = {
        "forward": 0,
        "backward_input": 0,
        "backward_weight": 0,
    }

    phase_counts = {
        "forward": 0,
        "backward_input": 0,
        "backward_weight": 0,
    }

    clients = set()

    for record in gemm_records:

        client_id = record.get("client_id")

        if client_id is not None:
            clients.add(int(client_id))

        phase = str(
            record.get("phase", "")
        )

        if phase in phase_flops:
            phase_flops[phase] += (
                record_flops(record)
            )

            phase_counts[phase] += 1

    model, method = infer_model_method(
        run_directory.name
    )

    return {
        "run_name":
            run_directory.name,

        "model":
            model,

        "method":
            method,

        "observed_unique_clients":
            len(clients),

        "completed_client_updates":
            completed_updates,

        "gemm_records":
            len(gemm_records),

        "total_gemm_flops":
            total_flops,

        "gemm_flops_per_client_update":
            total_flops / completed_updates,

        "total_gemm_gflops":
            total_flops / 1e9,

        "gemm_gflops_per_client_update":
            total_flops
            / completed_updates
            / 1e9,

        "forward_records":
            phase_counts["forward"],

        "backward_input_records":
            phase_counts[
                "backward_input"
            ],

        "backward_weight_records":
            phase_counts[
                "backward_weight"
            ],

        "forward_flops":
            phase_flops["forward"],

        "backward_input_flops":
            phase_flops[
                "backward_input"
            ],

        "backward_weight_flops":
            phase_flops[
                "backward_weight"
            ],

        "forward_gflops":
            phase_flops["forward"]
            / 1e9,

        "backward_input_gflops":
            phase_flops[
                "backward_input"
            ]
            / 1e9,

        "backward_weight_gflops":
            phase_flops[
                "backward_weight"
            ]
            / 1e9,
    }


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

    summaries = []

    for run_directory in sorted(
        experiment_directory.iterdir()
    ):
        if not run_directory.is_dir():
            continue

        summary = summarise_run(
            run_directory
        )

        if summary is None:
            print(
                "Warning: no usable GEMM "
                f"records in {run_directory}"
            )
            continue

        summaries.append(summary)

    if not summaries:
        raise SystemExit(
            "No GEMM records found."
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
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print(
        f"Wrote {len(summaries)} runs "
        f"to {args.output}"
    )

    for summary in summaries:
        print(
            f"{summary['run_name']:24s} "
            f"{summary['gemm_gflops_per_client_update']:10.2f} "
            f"GFLOPs/update "
            f"({summary['gemm_records']} records)"
        )


if __name__ == "__main__":
    main()
