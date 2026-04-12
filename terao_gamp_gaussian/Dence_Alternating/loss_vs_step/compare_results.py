#!/usr/bin/env python
"""
Compare multiple Dence_Alternating loss-vs-step result directories.

Each input path can be either:
- a result directory containing ``loss_history.csv``
- the ``loss_history.csv`` file itself

The script saves comparison plots and a compact CSV summary under
``Dence_Alternating/loss_vs_step/comparisons``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


COLORS = [
    "#1565C0",
    "#D84315",
    "#2E7D32",
    "#6A1B9A",
    "#00838F",
    "#F9A825",
    "#5D4037",
    "#546E7A",
]


@dataclass
class RunData:
    result_dir: Path
    label: str
    config: dict[str, object]
    steps: np.ndarray
    loss_mean: np.ndarray
    loss_std: np.ndarray
    log10_loss_mean: np.ndarray
    log10_loss_std: np.ndarray
    cosine_mean: np.ndarray
    cosine_std: np.ndarray
    final_loss_mean: float
    final_cosine_mean: float
    mean_runtime_sec: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Dence_Alternating loss-vs-step runs."
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
        help="Optional output directory. Defaults to loss_vs_step/comparisons/<timestamp>.",
    )
    return parser.parse_args()


def resolve_result_dir(path_str: str) -> Path:
    candidate = Path(path_str).expanduser().resolve()
    if candidate.is_file():
        if candidate.name != "loss_history.csv":
            raise FileNotFoundError(
                f"{candidate} is a file, but not loss_history.csv."
            )
        result_dir = candidate.parent
    else:
        result_dir = candidate

    loss_history_path = result_dir / "loss_history.csv"
    if not loss_history_path.exists():
        raise FileNotFoundError(
            f"loss_history.csv was not found under {result_dir}."
        )
    return result_dir


def strip_timestamp_prefix(name: str) -> str:
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]
    return name


def extract_timestamp_prefix(name: str) -> str | None:
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return None


def default_label(result_dir: Path, config: dict[str, object]) -> str:
    base = strip_timestamp_prefix(result_dir.name)
    timestamp_prefix = extract_timestamp_prefix(result_dir.name)
    alpha = config.get("alpha")
    noise_var = config.get("noise_var")
    if alpha is None or noise_var is None:
        if timestamp_prefix is None:
            return base
        return f"{base} [{timestamp_prefix}]"
    summary = f"{base} (alpha={alpha}, noise={noise_var})"
    if timestamp_prefix is None:
        return summary
    return f"{summary} [{timestamp_prefix}]"


def load_config(result_dir: Path) -> dict[str, object]:
    config_path = result_dir / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        return {}
    return config


def load_loss_history(result_dir: Path) -> dict[str, np.ndarray]:
    history_path = result_dir / "loss_history.csv"
    with history_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "step",
            "loss_mean",
            "loss_std",
            "log10_loss_mean",
            "log10_loss_std",
            "cosine_similarity_mean",
            "cosine_similarity_std",
        ]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{history_path} is missing required columns: {', '.join(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"{history_path} is empty.")

    def col(name: str, *, dtype: type[float] | type[int] = float) -> np.ndarray:
        return np.asarray([dtype(row[name]) for row in rows])

    return {
        "steps": col("step", dtype=int),
        "loss_mean": col("loss_mean"),
        "loss_std": col("loss_std"),
        "log10_loss_mean": col("log10_loss_mean"),
        "log10_loss_std": col("log10_loss_std"),
        "cosine_mean": col("cosine_similarity_mean"),
        "cosine_std": col("cosine_similarity_std"),
    }


def load_mean_runtime(result_dir: Path) -> float | None:
    summary_path = result_dir / "replica_summary.csv"
    if not summary_path.exists():
        return None

    runtimes: list[float] = []
    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runtime_raw = row.get("runtime_sec", "")
            if runtime_raw == "":
                continue
            runtimes.append(float(runtime_raw))

    if not runtimes:
        return None
    return float(np.mean(runtimes))


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
                "alpha",
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
                    run.config.get("alpha", ""),
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
        "algorithm_family": "Dence_Alternating_loss_vs_step_comparison",
        "num_inputs": len(runs),
        "inputs": [
            {
                "label": run.label,
                "result_dir": str(run.result_dir),
                "alpha": run.config.get("alpha"),
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


def plot_metric(
    output_path: Path,
    runs: list[RunData],
    metric_attr: str,
    std_attr: str,
    ylabel: str,
    title: str,
    band_floor: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, run in enumerate(runs):
        color = COLORS[idx % len(COLORS)]
        metric = getattr(run, metric_attr)
        std = getattr(run, std_attr)
        lower = metric - std
        upper = metric + std
        if band_floor is not None:
            lower = np.maximum(lower, band_floor)

        ax.plot(
            run.steps,
            metric,
            linewidth=2.2,
            color=color,
            label=run.label,
        )
        ax.fill_between(
            run.steps,
            lower,
            upper,
            color=color,
            alpha=0.18,
        )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    print("Dence_Alternating loss-vs-step comparison")
    print("=" * 60)
    for run in runs:
        runtime_text = (
            "n/a"
            if run.mean_runtime_sec is None
            else f"{run.mean_runtime_sec:.4f}s"
        )
        print(
            f"{run.label}: final_loss_mean={run.final_loss_mean:.10e}, "
            f"final_cosine_mean={run.final_cosine_mean:.10e}, "
            f"max_step={int(run.steps[-1])}, mean_runtime={runtime_text}"
        )
    print(f"Comparison saved to: {output_dir}")


if __name__ == "__main__":
    main()
