#!/usr/bin/env python
"""
Compare multiple non-uniform N1 loss-vs-step result directories.

Each input path can be either:
- a result directory containing ``loss_history.csv``
- the ``loss_history.csv`` file itself

The script saves comparison plots and a compact CSV summary under
``Dence_Alternating/non_uniform_n1_graph_version/loss_vs_step/comparisons``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Dence_Alternating.random_graph_version.loss_vs_step.compare_results import (
    RunData,
    extract_timestamp_prefix,
    load_config,
    load_loss_history,
    load_mean_runtime,
    plot_metric,
    resolve_result_dir,
    strip_timestamp_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare non-uniform N1 Dence_Alternating loss-vs-step runs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result directories or loss_history.csv files to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for each input path. Must match the number of paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "loss_vs_step/comparisons/<timestamp>."
        ),
    )
    return parser.parse_args()


def default_label(result_dir: Path, config: dict[str, object]) -> str:
    base = strip_timestamp_prefix(result_dir.name)
    timestamp_prefix = extract_timestamp_prefix(result_dir.name)
    alpha = config.get("alpha")
    p = config.get("p")
    r = config.get("r")
    noise_var = config.get("noise_var")

    summary_parts = [base]
    detail_parts = []
    if alpha is not None:
        detail_parts.append(f"alpha={alpha}")
    if p is not None:
        detail_parts.append(f"p={p}")
    if r is not None:
        detail_parts.append(f"r={r}")
    if noise_var is not None:
        detail_parts.append(f"noise={noise_var}")

    if detail_parts:
        summary_parts.append(f"({', '.join(detail_parts)})")

    summary = " ".join(summary_parts)
    if timestamp_prefix is None:
        return summary
    return f"{summary} [{timestamp_prefix}]"


def load_run(result_dir: Path, label: str | None) -> RunData:
    config = load_config(result_dir)
    history = load_loss_history(result_dir)
    resolved_label = label or default_label(result_dir, config)
    return RunData(
        result_dir=result_dir,
        label=resolved_label,
        config=config,
        steps=history["steps"],
        loss_mean=history["loss_mean"],
        loss_std=history["loss_std"],
        log10_loss_mean=history["log10_loss_mean"],
        log10_loss_std=history["log10_loss_std"],
        cosine_mean=history["cosine_mean"],
        cosine_std=history["cosine_std"],
        final_loss_mean=float(history["loss_mean"][-1]),
        final_cosine_mean=float(history["cosine_mean"][-1]),
        mean_runtime_sec=load_mean_runtime(result_dir),
    )


def build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        resolved = output_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved = (
        Path(__file__).resolve().parent / "comparisons" / f"{timestamp}_comparison"
    )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_summary(output_dir: Path, runs: list[RunData]) -> None:
    summary_path = output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "result_dir",
                "graph_model",
                "alpha",
                "p",
                "r",
                "N1",
                "N2",
                "M",
                "noise_var",
                "num_replicas",
                "max_recorded_step",
                "final_loss_mean",
                "final_cosine_similarity_mean",
                "mean_runtime_sec",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.label,
                    str(run.result_dir),
                    run.config.get("graph_model", ""),
                    run.config.get("alpha", ""),
                    run.config.get("p", ""),
                    run.config.get("r", ""),
                    run.config.get("N1", ""),
                    run.config.get("N2", ""),
                    run.config.get("M", ""),
                    run.config.get("noise_var", ""),
                    run.config.get("num_replicas", ""),
                    int(run.steps[-1]),
                    f"{run.final_loss_mean:.10e}",
                    f"{run.final_cosine_mean:.10e}",
                    ""
                    if run.mean_runtime_sec is None
                    else f"{run.mean_runtime_sec:.4f}",
                ]
            )


def save_manifest(output_dir: Path, runs: list[RunData]) -> None:
    manifest = {
        "algorithm_family": "Dence_Alternating_non_uniform_n1_loss_vs_step_comparison",
        "num_inputs": len(runs),
        "inputs": [
            {
                "label": run.label,
                "result_dir": str(run.result_dir),
                "graph_model": run.config.get("graph_model"),
                "alpha": run.config.get("alpha"),
                "p": run.config.get("p"),
                "r": run.config.get("r"),
                "N1": run.config.get("N1"),
                "N2": run.config.get("N2"),
                "M": run.config.get("M"),
                "noise_var": run.config.get("noise_var"),
                "num_replicas": run.config.get("num_replicas"),
            }
            for run in runs
        ],
    }
    with (output_dir / "comparison_manifest.yaml").open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.paths):
        raise ValueError("--labels must have the same length as the input paths.")

    result_dirs = [resolve_result_dir(path_str) for path_str in args.paths]
    labels = args.labels or [None] * len(result_dirs)
    runs = [load_run(result_dir, label) for result_dir, label in zip(result_dirs, labels)]
    output_dir = build_output_dir(args.output_dir)

    save_summary(output_dir, runs)
    save_manifest(output_dir, runs)
    plot_metric(
        output_dir / "loss_vs_step_linear_comparison.png",
        runs,
        metric_attr="loss_mean",
        std_attr="loss_std",
        ylabel="Observed MSE",
        title="Observed MSE vs Step Comparison",
        band_floor=0.0,
    )
    plot_metric(
        output_dir / "loss_vs_step_log10_comparison.png",
        runs,
        metric_attr="log10_loss_mean",
        std_attr="log10_loss_std",
        ylabel="log10(Observed MSE)",
        title="log10(Observed MSE) vs Step Comparison",
    )
    plot_metric(
        output_dir / "cosine_similarity_vs_step_comparison.png",
        runs,
        metric_attr="cosine_mean",
        std_attr="cosine_std",
        ylabel="Cosine Similarity",
        title="Cosine Similarity vs Step Comparison",
    )

    print("=" * 60)
    print("Dence_Alternating non-uniform N1 loss-vs-step comparison")
    print("=" * 60)
    for run in runs:
        runtime_text = (
            "n/a" if run.mean_runtime_sec is None else f"{run.mean_runtime_sec:.4f}s"
        )
        print(
            f"{run.label}: final_loss_mean={run.final_loss_mean:.10e}, "
            f"final_cosine_mean={run.final_cosine_mean:.10e}, "
            f"max_step={int(run.steps[-1])}, mean_runtime={runtime_text}"
        )
    print(f"Comparison saved to: {output_dir}")


if __name__ == "__main__":
    main()
