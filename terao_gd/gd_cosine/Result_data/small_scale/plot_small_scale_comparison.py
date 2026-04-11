#!/usr/bin/env python
"""
Plot small-scale AGD cosine-similarity results on a single graph.

This script reads per-replica values from each metrics.csv under this directory
and recomputes mean / SEM from the detailed CSV columns.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "cosine_similarity_comparison.png"


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
    metrics_files = sorted(BASE_DIR.glob("*/metrics.csv"))
    if len(metrics_files) < 2:
        raise FileNotFoundError("Expected at least two metrics.csv files under small_scale/.")

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#1E88E5", "#D81B60", "#43A047", "#FB8C00"]

    for color, metrics_path in zip(colors, metrics_files):
        alphas, means, sems = load_series(metrics_path)
        label = metrics_path.parent.name
        ax.errorbar(
            alphas,
            means,
            yerr=sems,
            fmt="o-",
            linewidth=2,
            markersize=6,
            capsize=4,
            color=color,
            label=label,
        )

    ax.set_xlabel(r"$\alpha$", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title("AGD Cosine Similarity Comparison", fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
