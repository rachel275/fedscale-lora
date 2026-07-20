#!/usr/bin/env python3
"""
Plot FedScale Full / LoRA / Top-K experiment results.

Expected experiment structure:

model-sweep-.../
├── albert_base_full/
│   ├── communication-executor-1.jsonl
│   ├── gemm-executor-1.jsonl
│   └── quality.jsonl
├── albert_base_lora/
├── albert_base_topk10/
├── albert_base_topk1/
├── bert_base_full/
└── ...

The script generates:

1. communication_per_update.png
2. gemm_compute_per_update.png
3. compute_vs_communication_retained.png
4. validation_loss_over_rounds.png
5. perplexity_over_rounds.png
6. quality_vs_cumulative_communication.png
7. gemm_shape_distribution.png

It also writes:
    plot-summary.csv

Usage:

    python3 plot-results.py \
        ~/fedscale-lora/results/model-sweep-YYYYMMDDTHHMMSSZ

Optional:

    python3 plot-results.py EXPERIMENT_DIR \
        --output-dir EXPERIMENT_DIR/plots

    python3 plot-results.py EXPERIMENT_DIR \
        --model albert_base

By default, plots include all discovered models/runs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots from FedScale communication, GEMM, and quality logs."
    )
    parser.add_argument(
        "experiment_directory",
        type=Path,
        help="Top-level model-sweep-* results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Default: <experiment_directory>/plots",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional model prefix to plot, e.g. albert_base, bert_base, "
            "distilbert_base, bert_large."
        ),
    )
    parser.add_argument(
        "--top-shapes",
        type=int,
        default=15,
        help="Number of most frequent GEMM shapes to include in shape distribution.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(
                    f"Warning: skipping invalid JSON in "
                    f"{path}:{line_number}: {error}"
                )

    return records


def infer_model_method(run_name: str) -> tuple[str, str]:
    """
    Convert names such as:

        albert_base_full
        albert_base_lora
        albert_base_topk10
        albert_base_topk1

    into model and method labels.
    """

    method_patterns = [
        ("topk10", "Top-K 10%"),
        ("topk1", "Top-K 1%"),
        ("lora", "LoRA"),
        ("full", "Full"),
    ]

    for suffix, label in method_patterns:
        marker = f"_{suffix}"
        if run_name.endswith(marker):
            return run_name[: -len(marker)], label

    return run_name, "Unknown"


def method_sort_key(method: str) -> int:
    order = {
        "Full": 0,
        "LoRA": 1,
        "Top-K 10%": 2,
        "Top-K 1%": 3,
    }
    return order.get(method, 99)


def display_model(model: str) -> str:
    labels = {
        "albert_base": "ALBERT Base",
        "bert_base": "BERT Base",
        "distilbert_base": "DistilBERT Base",
        "bert_large": "BERT Large",
    }
    return labels.get(model, model.replace("_", " ").title())


def load_communication_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for filename in sorted(
        glob.glob(str(run_dir / "communication-executor-*.jsonl"))
    ):
        records.extend(read_jsonl(Path(filename)))

    return records


def load_gemm_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for filename in sorted(
        glob.glob(str(run_dir / "gemm-executor-*.jsonl"))
    ):
        records.extend(read_jsonl(Path(filename)))

    return records


def load_quality_records(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir / "quality.jsonl")


def get_serialized_bytes(record: dict[str, Any]) -> int:
    return int(record.get("serialized_bytes", 0) or 0)


def get_gemm_flops(record: dict[str, Any]) -> int:
    if "flops" in record:
        return int(record["flops"])

    try:
        m = int(record["gemm_M"])
        n = int(record["gemm_N"])
        k = int(record["gemm_K"])
    except (KeyError, TypeError, ValueError):
        return 0

    return 2 * m * n * k


def collect_run_summaries(
    experiment_dir: Path,
    model_filter: str | None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summaries: list[dict[str, Any]] = []
    quality_by_run: dict[str, list[dict[str, Any]]] = {}

    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        model, method = infer_model_method(run_dir.name)

        if method == "Unknown":
            continue

        if model_filter and model != model_filter:
            continue

        comm = load_communication_records(run_dir)
        gemm = load_gemm_records(run_dir)
        quality = load_quality_records(run_dir)

        upload_records = [
            r for r in comm if r.get("direction") == "upload"
        ]
        download_records = [
            r for r in comm if r.get("direction") == "download"
        ]

        upload_count = len(upload_records)
        download_count = len(download_records)

        serialized_upload = sum(
            get_serialized_bytes(r) for r in upload_records
        )
        serialized_download = sum(
            get_serialized_bytes(r) for r in download_records
        )
        serialized_total = serialized_upload + serialized_download

        # A completed client update is naturally represented by an upload.
        updates = max(upload_count, 1)

        communication_per_update = serialized_total / updates

        training_gemm = [
            r
            for r in gemm
            if r.get("client_id") is not None
        ]

        total_gemm_flops = sum(
            get_gemm_flops(r) for r in training_gemm
        )

        # If one upload corresponds to one client update, normalize by uploads.
        gemm_flops_per_update = total_gemm_flops / updates

        forward_flops = sum(
            get_gemm_flops(r)
            for r in training_gemm
            if r.get("phase") == "forward"
        )
        backward_input_flops = sum(
            get_gemm_flops(r)
            for r in training_gemm
            if r.get("phase") == "backward_input"
        )
        backward_weight_flops = sum(
            get_gemm_flops(r)
            for r in training_gemm
            if r.get("phase") == "backward_weight"
        )

        latest_quality = (
            max(
                quality,
                key=lambda r: int(r.get("round", 0)),
            )
            if quality
            else {}
        )

        summaries.append(
            {
                "run_name": run_dir.name,
                "model": model,
                "model_label": display_model(model),
                "method": method,
                "upload_messages": upload_count,
                "download_messages": download_count,
                "serialized_upload_bytes": serialized_upload,
                "serialized_download_bytes": serialized_download,
                "serialized_total_bytes": serialized_total,
                "communication_bytes_per_update": communication_per_update,
                "communication_mib_per_update": communication_per_update
                / (1024**2),
                "gemm_records": len(training_gemm),
                "gemm_flops_total": total_gemm_flops,
                "gemm_flops_per_update": gemm_flops_per_update,
                "gemm_gflops_per_update": gemm_flops_per_update / 1e9,
                "forward_gflops": forward_flops / 1e9,
                "backward_input_gflops": backward_input_flops / 1e9,
                "backward_weight_gflops": backward_weight_flops / 1e9,
                "latest_round": latest_quality.get("round"),
                "validation_loss": latest_quality.get("validation_loss"),
                "perplexity": latest_quality.get("perplexity"),
                "run_dir": str(run_dir),
            }
        )

        quality_by_run[run_dir.name] = quality

    summaries.sort(
        key=lambda row: (
            row["model"],
            method_sort_key(row["method"]),
        )
    )

    return summaries, quality_by_run


def write_summary_csv(
    output_path: Path,
    summaries: list[dict[str, Any]],
) -> None:
    if not summaries:
        return

    excluded = {"run_dir"}

    fieldnames = [
        key
        for key in summaries[0].keys()
        if key not in excluded
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in summaries:
            writer.writerow(
                {
                    key: value
                    for key, value in row.items()
                    if key not in excluded
                }
            )


def group_summaries_by_model(
    summaries: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in summaries:
        grouped[row["model"]].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: method_sort_key(row["method"]))

    return dict(grouped)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def plot_communication_per_update(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    if not summaries:
        return

    labels = [
        f"{row['model_label']}\n{row['method']}"
        for row in summaries
    ]
    values = [
        row["communication_mib_per_update"]
        for row in summaries
    ]

    plt.figure(figsize=(max(9, len(labels) * 0.8), 6))
    plt.bar(labels, values)
    plt.ylabel("Serialized communication per update (MiB)")
    plt.title("Communication per client update")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)

    save_figure(
        output_dir / "1_communication_per_update.png"
    )


def plot_compute_per_update(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    valid = [
        row
        for row in summaries
        if row["gemm_records"] > 0
    ]

    if not valid:
        return

    labels = [
        f"{row['model_label']}\n{row['method']}"
        for row in valid
    ]
    values = [
        row["gemm_gflops_per_update"]
        for row in valid
    ]

    plt.figure(figsize=(max(9, len(labels) * 0.8), 6))
    plt.bar(labels, values)
    plt.ylabel("Traced GEMM work per update (GFLOPs)")
    plt.title("GEMM compute per client update")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)

    save_figure(
        output_dir / "2_gemm_compute_per_update.png"
    )


def plot_retained_compute_vs_communication(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    grouped = group_summaries_by_model(summaries)

    labels: list[str] = []
    communication_retained: list[float] = []
    compute_retained: list[float] = []

    for model, rows in grouped.items():
        baseline = next(
            (
                row
                for row in rows
                if row["method"] == "Full"
            ),
            None,
        )

        if baseline is None:
            continue

        base_comm = baseline[
            "communication_bytes_per_update"
        ]
        base_compute = baseline[
            "gemm_flops_per_update"
        ]

        if base_comm <= 0 or base_compute <= 0:
            continue

        for row in rows:
            labels.append(
                f"{row['model_label']}\n{row['method']}"
            )

            communication_retained.append(
                100.0
                * row["communication_bytes_per_update"]
                / base_comm
            )

            compute_retained.append(
                100.0
                * row["gemm_flops_per_update"]
                / base_compute
            )

    if not labels:
        return

    x = list(range(len(labels)))
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 0.9), 6))

    plt.bar(
        [i - width / 2 for i in x],
        communication_retained,
        width=width,
        label="Communication retained",
    )

    plt.bar(
        [i + width / 2 for i in x],
        compute_retained,
        width=width,
        label="GEMM compute retained",
    )

    plt.axhline(
        100,
        linewidth=1,
        linestyle="--",
    )

    plt.ylabel("Retained relative to Full (%)")
    plt.title("Compute retained versus communication retained")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    save_figure(
        output_dir
        / "3_compute_vs_communication_retained.png"
    )


def plot_quality_over_rounds(
    summaries: list[dict[str, Any]],
    quality_by_run: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    metric: str,
    filename: str,
    ylabel: str,
    title: str,
) -> None:
    plotted = False

    plt.figure(figsize=(10, 6))

    for row in summaries:
        quality = quality_by_run.get(
            row["run_name"],
            [],
        )

        points = [
            record
            for record in quality
            if record.get(metric) is not None
        ]

        if not points:
            continue

        points.sort(
            key=lambda record: int(
                record.get("round", 0)
            )
        )

        rounds = [
            int(record["round"])
            for record in points
        ]

        values = [
            float(record[metric])
            for record in points
        ]

        plt.plot(
            rounds,
            values,
            marker="o",
            label=(
                f"{row['model_label']} "
                f"{row['method']}"
            ),
        )

        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure(output_dir / filename)


def build_communication_timeline(
    run_dir: Path,
) -> list[tuple[int, float]]:
    """
    Estimate cumulative communication after each client update.

    We accumulate serialized communication in timestamp order and create
    a point whenever an upload record is observed. The returned list is:

        [(update_number, cumulative_MiB), ...]

    This is used to align quality checkpoints approximately with cumulative
    communication. For exact round-level alignment, communication records
    should contain reliable round identifiers for both upload and download.
    """

    records = load_communication_records(run_dir)

    records.sort(
        key=lambda record: float(
            record.get("timestamp", 0.0)
        )
    )

    cumulative = 0
    update = 0
    timeline: list[tuple[int, float]] = []

    for record in records:
        cumulative += get_serialized_bytes(record)

        if record.get("direction") == "upload":
            update += 1

            timeline.append(
                (
                    update,
                    cumulative / (1024**2),
                )
            )

    return timeline


def quality_communication_points(
    row: dict[str, Any],
    quality: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """
    Produce (cumulative MiB, validation loss) points.

    Prefer round-aware communication logs where available. If communication
    records do not have usable round information, approximate communication
    as:

        communication per update
        × observed quality round

    This is adequate for exploratory plots, but publication results should
    use round-tagged communication logs.
    """

    run_dir = Path(row["run_dir"])
    comm_records = load_communication_records(run_dir)

    by_round: dict[int, int] = defaultdict(int)
    have_rounds = False

    for record in comm_records:
        if "round" not in record:
            continue

        try:
            round_number = int(record["round"])
        except (TypeError, ValueError):
            continue

        have_rounds = True
        by_round[round_number] += get_serialized_bytes(
            record
        )

    points: list[tuple[float, float]] = []

    if have_rounds:
        cumulative = 0

        cumulative_by_round: dict[int, float] = {}

        for round_number in sorted(by_round):
            cumulative += by_round[round_number]

            cumulative_by_round[round_number] = (
                cumulative / (1024**2)
            )

        for record in sorted(
            quality,
            key=lambda r: int(r.get("round", 0)),
        ):
            if record.get("validation_loss") is None:
                continue

            round_number = int(record.get("round", 0))

            eligible = [
                r
                for r in cumulative_by_round
                if r <= round_number
            ]

            cumulative_mib = (
                cumulative_by_round[max(eligible)]
                if eligible
                else 0.0
            )

            points.append(
                (
                    cumulative_mib,
                    float(record["validation_loss"]),
                )
            )

        return points

    communication_per_update_mib = row[
        "communication_mib_per_update"
    ]

    for record in quality:
        if record.get("validation_loss") is None:
            continue

        round_number = int(record.get("round", 0))

        points.append(
            (
                communication_per_update_mib
                * max(round_number, 1),
                float(record["validation_loss"]),
            )
        )

    return points


def plot_quality_vs_communication(
    summaries: list[dict[str, Any]],
    quality_by_run: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    plotted = False

    plt.figure(figsize=(10, 6))

    for row in summaries:
        quality = quality_by_run.get(
            row["run_name"],
            [],
        )

        points = quality_communication_points(
            row,
            quality,
        )

        if not points:
            continue

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        plt.plot(
            x,
            y,
            marker="o",
            label=(
                f"{row['model_label']} "
                f"{row['method']}"
            ),
        )

        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Cumulative serialized communication (MiB)")
    plt.ylabel("Validation loss")
    plt.title(
        "Model quality versus cumulative communication"
    )
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure(
        output_dir
        / "6_quality_vs_cumulative_communication.png"
    )


def plot_gemm_shape_distribution(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    top_shapes: int,
) -> None:
    """
    Plot the top GEMM shapes, with one series per run.

    Shape is represented as M × K × N to mirror the mathematical
    multiplication:

        [M, K] @ [K, N]
    """

    run_shape_counts: dict[str, Counter[str]] = {}
    total_counts: Counter[str] = Counter()

    for row in summaries:
        run_dir = Path(row["run_dir"])

        records = [
            record
            for record in load_gemm_records(run_dir)
            if record.get("client_id") is not None
        ]

        counts: Counter[str] = Counter()

        for record in records:
            try:
                m = int(record["gemm_M"])
                n = int(record["gemm_N"])
                k = int(record["gemm_K"])
            except (KeyError, TypeError, ValueError):
                continue

            shape = f"{m}×{k}×{n}"

            counts[shape] += 1
            total_counts[shape] += 1

        if counts:
            run_shape_counts[
                f"{row['model_label']} {row['method']}"
            ] = counts

    if not run_shape_counts:
        return

    shapes = [
        shape
        for shape, _
        in total_counts.most_common(top_shapes)
    ]

    x = list(range(len(shapes)))

    plt.figure(
        figsize=(
            max(11, len(shapes) * 0.9),
            7,
        )
    )

    for run_label, counts in run_shape_counts.items():
        values = [
            counts.get(shape, 0)
            for shape in shapes
        ]

        plt.plot(
            x,
            values,
            marker="o",
            label=run_label,
        )

    plt.xticks(
        x,
        shapes,
        rotation=45,
        ha="right",
    )

    plt.xlabel("GEMM shape M×K×N")
    plt.ylabel("Observed operation count")
    plt.title(
        f"Top {len(shapes)} GEMM shape distributions"
    )
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure(
        output_dir
        / "7_gemm_shape_distribution.png"
    )


def main() -> None:
    args = parse_args()

    experiment_dir = (
        args.experiment_directory.expanduser().resolve()
    )

    if not experiment_dir.is_dir():
        raise SystemExit(
            f"Not a directory: {experiment_dir}"
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else experiment_dir / "plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries, quality_by_run = collect_run_summaries(
        experiment_dir,
        args.model,
    )

    if not summaries:
        raise SystemExit(
            "No recognizable experiment runs found."
        )

    summary_path = output_dir / "plot-summary.csv"

    write_summary_csv(
        summary_path,
        summaries,
    )

    print(f"Wrote {summary_path}")

    # 1. Communication per client update.
    plot_communication_per_update(
        summaries,
        output_dir,
    )

    # 2. GEMM compute per client update.
    plot_compute_per_update(
        summaries,
        output_dir,
    )

    # 3. Compute retained vs communication retained.
    plot_retained_compute_vs_communication(
        summaries,
        output_dir,
    )

    # 4. Validation loss over rounds.
    plot_quality_over_rounds(
        summaries,
        quality_by_run,
        output_dir,
        metric="validation_loss",
        filename="4_validation_loss_over_rounds.png",
        ylabel="Validation loss",
        title="Validation loss over federated rounds",
    )

    # 5. Perplexity over rounds.
    plot_quality_over_rounds(
        summaries,
        quality_by_run,
        output_dir,
        metric="perplexity",
        filename="5_perplexity_over_rounds.png",
        ylabel="Perplexity",
        title="Perplexity over federated rounds",
    )

    # 6. Quality vs cumulative communication.
    plot_quality_vs_communication(
        summaries,
        quality_by_run,
        output_dir,
    )

    # 7. GEMM shape distribution.
    plot_gemm_shape_distribution(
        summaries,
        output_dir,
        args.top_shapes,
    )

    print()
    print("Finished.")
    print(f"Plots: {output_dir}")
    print()
    print("Note:")
    print(
        "Quality-vs-communication is exact when communication "
        "records contain round IDs. Otherwise the script uses "
        "communication-per-update × quality round as an approximation."
    )


if __name__ == "__main__":
    main()
