#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    return pd.to_numeric(series, errors="coerce").mean()


def safe_max(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return np.nan
    return pd.to_numeric(series, errors="coerce").max()


def safe_sum(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return pd.to_numeric(series, errors="coerce").sum()


def infer_model_method_from_run_name(run_name: str) -> tuple[str, str]:
    suffixes = {
        "_topk10": "topk10",
        "_topk1": "topk1",
        "_lora": "lora",
        "_full": "full",
    }

    for suffix, method in suffixes.items():
        if run_name.endswith(suffix):
            return run_name[:-len(suffix)], method

    return run_name, "unknown"


def normalize_model_name(model: str) -> str:
    model = str(model)

    mapping = {
        "albert-base-v2": "albert_base",
        "albert_base": "albert_base",

        "distilbert-base-uncased": "distilbert_base",
        "distilbert_base": "distilbert_base",

        "bert-base-uncased": "bert_base",
        "bert_base": "bert_base",

        "bert-large-uncased": "bert_large",
        "bert_large": "bert_large",

        "meta-llama/Llama-3.2-1B": "llama_1b",
        "llama_1b": "llama_1b",

        "meta-llama/Llama-3.1-8B": "llama_8b",
        "llama_8b": "llama_8b",
    }

    return mapping.get(model, model)


def ensure_model_method_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "run_name" in df.columns:
        inferred = df["run_name"].astype(str).apply(
            infer_model_method_from_run_name
        )

        if "model" not in df.columns:
            df["model"] = inferred.apply(lambda x: x[0])

        if "method" not in df.columns:
            df["method"] = inferred.apply(lambda x: x[1])

    if "model" not in df.columns:
        raise ValueError(
            "Input must contain either 'model' or 'run_name'."
        )

    if "method" not in df.columns:
        raise ValueError(
            "Input must contain either 'method' or 'run_name'."
        )

    df["model"] = df["model"].apply(normalize_model_name)

    if "run_name" in df.columns:
        run_names = df["run_name"].astype(str)

        df.loc[
            run_names.str.contains("topk10", case=False, na=False),
            "method",
        ] = "topk10"

        df.loc[
            run_names.str.contains("topk1", case=False, na=False)
            & ~run_names.str.contains("topk10", case=False, na=False),
            "method",
        ] = "topk1"

    return df


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_client_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = ensure_model_method_columns(df)

    peak_rss_col = first_existing_column(
        df, ["peak_rss_bytes", "peak_rss", "max_rss_bytes"]
    )
    peak_extra_rss_col = first_existing_column(
        df, ["peak_extra_rss_bytes", "peak_extra_rss"]
    )
    model_param_col = first_existing_column(
        df,
        [
            "model_parameter_bytes",
            "model_param_bytes",
            "parameter_bytes",
            "model_bytes",
        ],
    )
    trainable_param_col = first_existing_column(
        df,
        [
            "trainable_parameter_bytes",
            "trainable_param_bytes",
            "trainable_bytes",
        ],
    )
    gradient_col = first_existing_column(
        df, ["gradient_bytes", "grad_bytes"]
    )
    optimizer_col = first_existing_column(
        df, ["optimizer_state_bytes", "optimizer_bytes"]
    )
    training_duration_col = first_existing_column(
        df,
        [
            "training_duration_s",
            "training_duration_sec",
            "training_time_s",
            "duration_s",
        ],
    )

    rows = []

    for (model, method), group in df.groupby(
        ["model", "method"], dropna=False
    ):
        row = {
            "model": model,
            "method": method,
            "client_metric_records": len(group),
        }

        if peak_rss_col:
            row["peak_rss_bytes"] = safe_max(group[peak_rss_col])
            row["peak_rss_gib"] = row["peak_rss_bytes"] / (1024 ** 3)

        if peak_extra_rss_col:
            row["peak_extra_rss_bytes"] = safe_max(group[peak_extra_rss_col])
            row["peak_extra_rss_gib"] = (
                row["peak_extra_rss_bytes"] / (1024 ** 3)
            )

        if model_param_col:
            row["model_parameter_bytes"] = safe_max(group[model_param_col])
            row["model_parameter_gib"] = (
                row["model_parameter_bytes"] / (1024 ** 3)
            )

        if trainable_param_col:
            row["trainable_parameter_bytes"] = safe_max(
                group[trainable_param_col]
            )
            row["trainable_parameter_mib"] = (
                row["trainable_parameter_bytes"] / (1024 ** 2)
            )

        if gradient_col:
            row["gradient_bytes"] = safe_max(group[gradient_col])
            row["gradient_mib"] = row["gradient_bytes"] / (1024 ** 2)

        if optimizer_col:
            row["optimizer_state_bytes"] = safe_max(group[optimizer_col])
            row["optimizer_state_mib"] = (
                row["optimizer_state_bytes"] / (1024 ** 2)
            )

        if training_duration_col:
            row["mean_training_duration_s"] = safe_mean(
                group[training_duration_col]
            )
            row["max_training_duration_s"] = safe_max(
                group[training_duration_col]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_client_updates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = ensure_model_method_columns(df)

    download_col = first_existing_column(
        df,
        [
            "download_time_s",
            "download_duration_s",
            "download_s",
        ],
    )
    training_col = first_existing_column(
        df,
        [
            "training_time_s",
            "train_time_s",
            "local_training_time_s",
            "training_duration_s",
        ],
    )
    upload_col = first_existing_column(
        df,
        [
            "upload_time_s",
            "upload_duration_s",
            "upload_s",
        ],
    )
    total_col = first_existing_column(
        df,
        [
            "total_time_s",
            "client_update_time_s",
            "end_to_end_time_s",
            "duration_s",
        ],
    )
    download_bytes_col = first_existing_column(
        df,
        [
            "download_bytes",
            "serialized_download_bytes",
        ],
    )
    upload_bytes_col = first_existing_column(
        df,
        [
            "upload_bytes",
            "serialized_upload_bytes",
        ],
    )

    rows = []

    for (model, method), group in df.groupby(
        ["model", "method"], dropna=False
    ):
        row = {
            "model": model,
            "method": method,
            "completed_client_updates": len(group),
        }

        if download_col:
            row["mean_download_time_s"] = safe_mean(group[download_col])

        if training_col:
            row["mean_client_training_time_s"] = safe_mean(group[training_col])

        if upload_col:
            row["mean_upload_time_s"] = safe_mean(group[upload_col])

        if total_col:
            row["mean_client_update_time_s"] = safe_mean(group[total_col])

        if download_bytes_col:
            row["mean_download_bytes_per_update"] = safe_mean(
                group[download_bytes_col]
            )

        if upload_bytes_col:
            row["mean_upload_bytes_per_update"] = safe_mean(
                group[upload_bytes_col]
            )

        if download_bytes_col and upload_bytes_col:
            total_bytes = (
                pd.to_numeric(group[download_bytes_col], errors="coerce")
                + pd.to_numeric(group[upload_bytes_col], errors="coerce")
            )

            row["mean_communication_bytes_per_update"] = total_bytes.mean()
            row["mean_communication_mib_per_update"] = (
                row["mean_communication_bytes_per_update"] / (1024 ** 2)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_gemm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = ensure_model_method_columns(df)

    flops_col = first_existing_column(df, ["flops", "gemm_flops"])
    duration_col = first_existing_column(
        df, ["duration_us", "gemm_duration_us"]
    )
    phase_col = first_existing_column(df, ["phase"])
    client_col = first_existing_column(df, ["client_id"])

    rows = []

    for (model, method), group in df.groupby(
        ["model", "method"], dropna=False
    ):
        row = {
            "model": model,
            "method": method,
            "gemm_records": len(group),
        }

        if flops_col:
            total_flops = safe_sum(group[flops_col])
            row["gemm_flops_total"] = total_flops
            row["gemm_gflops_total"] = total_flops / 1e9
            row["mean_gemm_flops"] = safe_mean(group[flops_col])

        if duration_col:
            total_duration_us = safe_sum(group[duration_col])
            row["gemm_time_total_s"] = total_duration_us / 1e6
            row["mean_gemm_duration_us"] = safe_mean(group[duration_col])

        if flops_col and duration_col:
            total_flops = safe_sum(group[flops_col])
            total_duration_s = safe_sum(group[duration_col]) / 1e6

            if total_duration_s > 0:
                row["achieved_gemm_gflops"] = (
                    total_flops / total_duration_s / 1e9
                )

        if client_col:
            row["unique_clients_in_gemm_trace"] = (
                group[client_col].dropna().nunique()
            )

        if phase_col and flops_col:
            phase_names = {
                "forward": "forward",
                "backward_input": "backward_input",
                "backward_weight": "backward_weight",
            }

            for raw_phase, output_name in phase_names.items():
                phase_group = group[group[phase_col] == raw_phase]

                row[f"{output_name}_gemm_gflops_total"] = (
                    safe_sum(phase_group[flops_col]) / 1e9
                )

                if duration_col:
                    row[f"{output_name}_gemm_time_s"] = (
                        safe_sum(phase_group[duration_col]) / 1e6
                    )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a model x method client-resource summary from combined "
            "client metrics, client updates, and GEMM CSV files."
        )
    )

    parser.add_argument(
        "--client-metrics",
        required=True,
        help="Path to all_client_metrics.csv",
    )
    parser.add_argument(
        "--client-updates",
        required=True,
        help="Path to all_client_updates.csv",
    )
    parser.add_argument(
        "--gemm",
        required=True,
        help="Path to all_gemm_operations.csv",
    )
    parser.add_argument(
        "--output",
        default="client-resource-summary.csv",
        help="Output summary CSV",
    )

    args = parser.parse_args()

    metrics_df = pd.read_csv(Path(args.client_metrics))
    updates_df = pd.read_csv(Path(args.client_updates))
    gemm_df = pd.read_csv(Path(args.gemm))

    metrics_summary = summarize_client_metrics(metrics_df)
    updates_summary = summarize_client_updates(updates_df)
    gemm_summary = summarize_gemm(gemm_df)

    summary = metrics_summary.merge(
        updates_summary,
        on=["model", "method"],
        how="outer",
    )

    summary = summary.merge(
        gemm_summary,
        on=["model", "method"],
        how="outer",
    )

    if (
        "gemm_flops_total" in summary.columns
        and "completed_client_updates" in summary.columns
    ):
        valid = summary["completed_client_updates"] > 0

        summary.loc[
            valid,
            "gemm_gflops_per_client_update"
        ] = (
            summary.loc[valid, "gemm_flops_total"]
            / summary.loc[valid, "completed_client_updates"]
            / 1e9
        )

    if (
        "gemm_time_total_s" in summary.columns
        and "completed_client_updates" in summary.columns
    ):
        valid = summary["completed_client_updates"] > 0

        summary.loc[
            valid,
            "gemm_time_s_per_client_update"
        ] = (
            summary.loc[valid, "gemm_time_total_s"]
            / summary.loc[valid, "completed_client_updates"]
        )

    if (
        "gemm_time_s_per_client_update" in summary.columns
        and "mean_client_training_time_s" in summary.columns
    ):
        valid = summary["mean_client_training_time_s"] > 0

        summary.loc[
            valid,
            "gemm_fraction_of_training_time"
        ] = (
            summary.loc[valid, "gemm_time_s_per_client_update"]
            / summary.loc[valid, "mean_client_training_time_s"]
        )

        summary.loc[
            valid,
            "gemm_percent_of_training_time"
        ] = (
            summary.loc[valid, "gemm_fraction_of_training_time"] * 100
        )

    preferred_model_order = {
        "albert_base": 0,
        "distilbert_base": 1,
        "bert_base": 2,
        "bert_large": 3,
        "llama_1b": 4,
        "llama_3b": 5,
        "llama_8b": 6,
    }

    preferred_method_order = {
        "full": 0,
        "lora": 1,
        "topk10": 2,
        "topk1": 3,
    }

    summary["_model_order"] = (
        summary["model"].map(preferred_model_order).fillna(999)
    )
    summary["_method_order"] = (
        summary["method"].map(preferred_method_order).fillna(999)
    )

    summary = (
        summary
        .sort_values(
            ["_model_order", "_method_order", "model", "method"]
        )
        .drop(columns=["_model_order", "_method_order"])
    )

    output_path = Path(args.output)

    summary.to_csv(output_path, index=False)

    print(f"Wrote summary to: {output_path}")
    print(f"Rows: {len(summary)}")

    display_columns = [
        col
        for col in [
            "model",
            "method",
            "completed_client_updates",
            "mean_client_update_time_s",
            "mean_client_training_time_s",
            "peak_rss_gib",
            "mean_communication_mib_per_update",
            "gemm_gflops_per_client_update",
            "gemm_time_s_per_client_update",
            "achieved_gemm_gflops",
            "gemm_percent_of_training_time",
        ]
        if col in summary.columns
    ]

    if display_columns:
        print()
        print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()