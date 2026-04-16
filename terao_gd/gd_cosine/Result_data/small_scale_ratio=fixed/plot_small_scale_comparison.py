#!/usr/bin/env python
"""
Plot small-scale AGD cosine-similarity results with fixed ratio on one graph.

This directory currently contains metrics for some runs and only plot images for
others, so the script plots every series that has a metrics.csv and reports the
missing ones.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "cosine_similarity_comparison.png"


def load_config(result_dir: Path) -> dict[str, object]:
    config_path = result_dir / "config.yaml"
    if not config_path.exists():
        return {}

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}
    return config if isinstance(config, dict) else {}


def load_label(result_dir: Path) -> str:
    config = load_config(result_dir)
    n1 = config.get("N1")
    m = config.get("M")
    if n1 is not None and m is not None:
        return f"N={n1}, M={m}"
    return result_dir.name


def load_series(metrics_path: Path) -> tuple[list[float], list[float], list[float]]:
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        replica_cols = sorted(
            [name for name in fieldnames if name.startswith("cosine_similarity_replica_")],
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )

        alphas: list[float] = []
        means: list[float] = []
        sems: list[float] = []

        for row in reader:
            values = [float(row[col]) for col in replica_cols if row[col] != ""]
            if not values:
                continue

            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(var)
            sem = std / math.sqrt(len(values))

            alphas.append(float(row["alpha"]))
            means.append(mean)
            sems.append(sem)

    return alphas, means, sems


def main() -> None:
    result_dirs = sorted(
        path for path in BASE_DIR.iterdir() if path.is_dir() and not path.name.startswith("__")
    )
    metrics_files = [path / "metrics.csv" for path in result_dirs if (path / "metrics.csv").exists()]
    missing_metrics = [path.name for path in result_dirs if not (path / "metrics.csv").exists()]

    if len(metrics_files) < 1:
        raise FileNotFoundError("No metrics.csv files found under small_scale_ratio=fixed/.")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1E88E5", "#D81B60", "#43A047", "#FB8C00"]
    markers = ["o", "s", "^", "D"]

    for color, marker, metrics_path in zip(colors, markers, metrics_files):
        alphas, means, sems = load_series(metrics_path)
        label = load_label(metrics_path.parent)
        ax.errorbar(
            alphas,
            means,
            yerr=sems,
            fmt=f"{marker}-",
            linewidth=2,
            markersize=6,
            capsize=4,
            color=color,
            label=label,
        )

    ax.set_xlabel(r"$\alpha$", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title("AGD Cosine Similarity Comparison (ratio fixed)", fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if missing_metrics:
        note = "Missing metrics: " + ", ".join(missing_metrics)
        ax.text(
            0.99,
            0.02,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="dimgray",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_PATH}")
    if missing_metrics:
        print("Missing metrics for: " + ", ".join(missing_metrics))


if __name__ == "__main__":
    main()
