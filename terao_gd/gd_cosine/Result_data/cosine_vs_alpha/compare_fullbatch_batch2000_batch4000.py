#!/usr/bin/env python
"""
Plot full-batch and two minibatch cosine-vs-alpha results on one graph.

This script supports the current result layout where:
- the full-batch result may only have ``plot.csv``
- the minibatch results provide ``metrics.csv``
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


BASE_DIR = Path(__file__).resolve().parent
FULLBATCH_DIR = BASE_DIR / "Fullbatch_1000x100_alpha0-5"
MINIBATCH_2000_DIR = (
    BASE_DIR.parent.parent.parent
    / "gd_cosine_minibatch"
    / "Result_data"
    / "cosine_vs_alpha"
    / "batch2000_1000x100_alpha0-5_batchsize2000"
)
MINIBATCH_4000_DIR = (
    BASE_DIR.parent.parent.parent
    / "gd_cosine_minibatch"
    / "Result_data"
    / "cosine_vs_alpha"
    / "batch4000_1000x100_alpha0-5"
)

OUTPUT_DIR = BASE_DIR / "comparisons" / "fullbatch_batch2000_batch4000_1000x100_alpha0-5"
PLOT_PATH = OUTPUT_DIR / "cosine_similarity_comparison.png"
CSV_PATH = OUTPUT_DIR / "comparison_series.csv"
MANIFEST_PATH = OUTPUT_DIR / "comparison_manifest.yaml"

FULLBATCH_ROW_RE = re.compile(
    r"^\s*(?P<alpha>[0-9.]+)\s*\|\s*"
    r"(?P<mean>[0-9.eE+-]+)\s*±\s*(?P<std>[0-9.eE+-]+)\s*\|\s*"
    r"(?P<loss_mean>[0-9.eE+-]+)\s*±\s*(?P<loss_std>[0-9.eE+-]+)\s*\|\s*"
    r"(?P<steps>[0-9.eE+-]+)\s*$"
)


def load_config(result_dir: Path) -> dict[str, object]:
    with (result_dir / "config.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config under {result_dir}")
    return config


def load_fullbatch_series(result_dir: Path) -> tuple[list[float], list[float], list[float]]:
    config = load_config(result_dir)
    num_replicas = max(int(config.get("num_replicas", 1)), 1)
    plot_csv_path = result_dir / "plot.csv"
    if not plot_csv_path.exists():
        raise FileNotFoundError(f"plot.csv not found under {result_dir}")

    alphas: list[float] = []
    means: list[float] = []
    sems: list[float] = []

    for line in plot_csv_path.read_text().splitlines():
        match = FULLBATCH_ROW_RE.match(line)
        if match is None:
            continue

        alpha = float(match.group("alpha"))
        mean = float(match.group("mean"))
        std = float(match.group("std"))
        sem = std / math.sqrt(num_replicas)

        alphas.append(alpha)
        means.append(mean)
        sems.append(sem)

    if not alphas:
        raise ValueError(f"Could not parse any data rows from {plot_csv_path}")

    return alphas, means, sems


def load_metrics_series(result_dir: Path) -> tuple[list[float], list[float], list[float]]:
    config = load_config(result_dir)
    num_replicas = max(int(config.get("num_replicas", 1)), 1)
    metrics_path = result_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found under {result_dir}")

    alphas: list[float] = []
    means: list[float] = []
    sems: list[float] = []

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alphas.append(float(row["alpha"]))
            mean = float(row["cosine_similarity_mean"])
            std = float(row["cosine_similarity_std"])
            means.append(mean)
            sems.append(std / math.sqrt(num_replicas))

    return alphas, means, sems


def save_comparison_csv(
    series_map: dict[str, tuple[list[float], list[float], list[float]]],
) -> None:
    alpha_grid = series_map["Full-batch"][0]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "alpha",
                "fullbatch_mean",
                "fullbatch_sem",
                "batch2000_mean",
                "batch2000_sem",
                "batch4000_mean",
                "batch4000_sem",
            ]
        )
        for idx, alpha in enumerate(alpha_grid):
            writer.writerow(
                [
                    alpha,
                    series_map["Full-batch"][1][idx],
                    series_map["Full-batch"][2][idx],
                    series_map["Minibatch batch_size=2000"][1][idx],
                    series_map["Minibatch batch_size=2000"][2][idx],
                    series_map["Minibatch batch_size=4000"][1][idx],
                    series_map["Minibatch batch_size=4000"][2][idx],
                ]
            )


def save_manifest() -> None:
    manifest = {
        "comparison_name": "fullbatch_batch2000_batch4000_1000x100_alpha0-5",
        "sources": [
            {
                "label": "Full-batch",
                "result_dir": str(FULLBATCH_DIR),
                "data_file": str(FULLBATCH_DIR / "plot.csv"),
            },
            {
                "label": "Minibatch batch_size=2000",
                "result_dir": str(MINIBATCH_2000_DIR),
                "data_file": str(MINIBATCH_2000_DIR / "metrics.csv"),
            },
            {
                "label": "Minibatch batch_size=4000",
                "result_dir": str(MINIBATCH_4000_DIR),
                "data_file": str(MINIBATCH_4000_DIR / "metrics.csv"),
            },
        ],
        "outputs": {
            "plot_path": str(PLOT_PATH),
            "csv_path": str(CSV_PATH),
        },
    }
    with MANIFEST_PATH.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    series_map = {
        "Full-batch": load_fullbatch_series(FULLBATCH_DIR),
        "Minibatch batch_size=2000": load_metrics_series(MINIBATCH_2000_DIR),
        "Minibatch batch_size=4000": load_metrics_series(MINIBATCH_4000_DIR),
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    styles = [
        ("Full-batch", "#1E88E5", "o"),
        ("Minibatch batch_size=2000", "#D81B60", "s"),
        ("Minibatch batch_size=4000", "#43A047", "^"),
    ]

    for label, color, marker in styles:
        alphas, means, sems = series_map[label]
        ax.errorbar(
            alphas,
            means,
            yerr=sems,
            fmt=f"{marker}-",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
            capthick=1.2,
            elinewidth=1.2,
            label=label,
        )

    ax.set_xlabel(r"$\alpha$", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title("Full-batch vs Minibatch Cosine Similarity", fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_comparison_csv(series_map)
    save_manifest()

    print(f"Saved comparison plot: {PLOT_PATH}")
    print(f"Saved comparison csv: {CSV_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
