#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path):
    """Read a JSONL file and return a list of dictionaries."""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Could not parse "
                    f"{path}:{line_number}: {exc}"
                )

    return records


def collect_files(run_dir, pattern, run_name):
    """Collect all matching executor JSONL files."""
    records = []

    for path in sorted(run_dir.glob(pattern)):
        print(f"Reading {path}")

        file_records = read_jsonl(path)

        for record in file_records:
            # Add the run name if it is not already present.
            record.setdefault("run_name", run_name)

            # Record the source file for traceability.
            record["source_file"] = path.name

        records.extend(file_records)

    return records


def write_csv(records, output_path):
    if not records:
        print(f"No records found for {output_path.name}")
        return

    df = pd.DataFrame(records)

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Wrote {len(df):,} rows -> {output_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing experiment run directories",
    )

    parser.add_argument(
        "--output-dir",
        default="csv_results",
        help="Directory for generated CSV files",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Combined records across every experiment.
    all_client_metrics = []
    all_client_updates = []
    all_gemm = []

    # Each directory is treated as one experiment/run.
    for run_dir in sorted(results_dir.iterdir()):

        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        print()
        print("=" * 70)
        print(f"Processing run: {run_name}")
        print("=" * 70)

        client_metrics = collect_files(
            run_dir,
            "client-metrics-executor-*.jsonl",
            run_name,
        )

        client_updates = collect_files(
            run_dir,
            "client-update-executor-*.jsonl",
            run_name,
        )

        gemm = collect_files(
            run_dir,
            "gemm-executor-*.jsonl",
            run_name,
        )

        # -------------------------------------------------
        # Save one set of CSVs for this individual run.
        # -------------------------------------------------

        run_output = output_dir / run_name

        run_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_csv(
            client_metrics,
            run_output / "client_metrics.csv",
        )

        write_csv(
            client_updates,
            run_output / "client_updates.csv",
        )

        write_csv(
            gemm,
            run_output / "gemm_operations.csv",
        )

        # Add to global datasets.
        all_client_metrics.extend(client_metrics)
        all_client_updates.extend(client_updates)
        all_gemm.extend(gemm)

    # -------------------------------------------------
    # Save combined CSVs across ALL models and methods.
    # -------------------------------------------------

    print()
    print("=" * 70)
    print("Writing combined datasets")
    print("=" * 70)

    write_csv(
        all_client_metrics,
        output_dir / "all_client_metrics.csv",
    )

    write_csv(
        all_client_updates,
        output_dir / "all_client_updates.csv",
    )

    write_csv(
        all_gemm,
        output_dir / "all_gemm_operations.csv",
    )


if __name__ == "__main__":
    main()