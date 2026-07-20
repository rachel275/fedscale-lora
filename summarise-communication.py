#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise FedScale communication JSONL logs."
    )
    parser.add_argument(
        "sweep_directory",
        type=Path,
        help="Directory containing one subdirectory per experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_records(run_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    pattern = str(
        run_directory / "communication-executor-*.jsonl"
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


def extract_participants(run_name: str) -> int | None:
    match = re.search(r"_p(\d+)$", run_name)
    if match is None:
        return None
    return int(match.group(1))

def extract_method(
    run_name: str,
    records: list[dict[str, Any]],
) -> str:
    methods = {
        str(record.get("method"))
        for record in records
        if record.get("method")
    }

    if len(methods) == 1:
        return methods.pop()

    if "_lora_" in run_name:
        return "lora"

    if "_full_" in run_name:
        return "full"

    return "unknown"

def summarise_run(
    run_directory: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    totals: dict[str, int] = defaultdict(int)
    message_counts: dict[str, int] = defaultdict(int)
    clients: set[int] = set()
    rounds: set[int] = set()

    raw_upload_bytes = 0
    raw_download_bytes = 0

    for record in records:
        direction = str(record["direction"])
        serialized_bytes = int(record.get("serialized_bytes", 0))

        totals[direction] += serialized_bytes
        message_counts[direction] += 1

        if "client_id" in record:
            clients.add(int(record["client_id"]))

        if "round" in record:
            rounds.add(int(record["round"]))

        if direction == "upload":
            raw_upload_bytes += int(
                record.get("raw_update_bytes", 0)
            )
        elif direction == "download":
            raw_download_bytes += int(
                record.get("raw_model_bytes", 0)
            )

    upload_bytes = totals["upload"]
    download_bytes = totals["download"]
    total_bytes = upload_bytes + download_bytes

    completed_client_updates = message_counts["upload"]

    average_per_update = (
        total_bytes / completed_client_updates
        if completed_client_updates
        else 0
    )

    return {
        "run_name": run_directory.name,
        "method": extract_method(
            run_directory.name,
            records,
        ),
        "configured_participants": extract_participants(
            run_directory.name
        ),
        "observed_unique_clients": len(clients),
        "observed_rounds": len(rounds),
        "upload_messages": message_counts["upload"],
        "download_messages": message_counts["download"],
        "raw_upload_bytes": raw_upload_bytes,
        "raw_download_bytes": raw_download_bytes,
        "serialized_upload_bytes": upload_bytes,
        "serialized_download_bytes": download_bytes,
        "serialized_total_bytes": total_bytes,
        "average_total_bytes_per_client_update": average_per_update,
        "serialized_upload_mib": upload_bytes / (1024**2),
        "serialized_download_mib": download_bytes / (1024**2),
        "serialized_total_mib": total_bytes / (1024**2),
    }


def main() -> None:
    args = parse_arguments()

    if not args.sweep_directory.is_dir():
        raise SystemExit(
            f"Not a directory: {args.sweep_directory}"
        )

    summaries: list[dict[str, Any]] = []

    for run_directory in sorted(args.sweep_directory.iterdir()):
        if not run_directory.is_dir():
            continue

        records = load_records(run_directory)

        if not records:
            print(
                f"Warning: no communication records in "
                f"{run_directory}"
            )
            continue

        summaries.append(
            summarise_run(run_directory, records)
        )

    if not summaries:
        raise SystemExit("No communication records found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} runs to {args.output}")

    for summary in summaries:
        print(
            f"{summary['run_name']:12s} "
            f"{summary['serialized_total_mib']:10.2f} MiB "
            f"across {summary['upload_messages']} updates"
        )


if __name__ == "__main__":
    main()
