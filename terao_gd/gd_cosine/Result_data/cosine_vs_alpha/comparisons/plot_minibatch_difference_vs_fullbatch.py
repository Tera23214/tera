#!/usr/bin/env python
"""
Plot cosine-similarity differences relative to the full-batch baseline.

The script reads the combined comparison CSV created by
`compare_fullbatch_batch2000_batch4000.py` and saves:
- a difference plot with y = minibatch - fullbatch
- a CSV with the plotted difference series
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = BASE_DIR / "fullbatch_batch2000_batch4000_1000x100_alpha0-5"
INPUT_CSV_PATH = COMPARISON_DIR / "comparison_series.csv"
OUTPUT_PLOT_PATH = COMPARISON_DIR / "minibatch_minus_fullbatch_difference.png"
OUTPUT_CSV_PATH = COMPARISON_DIR / "minibatch_minus_fullbatch_difference.csv"


def load_difference_series() -> list[dict[str, float]]:
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {INPUT_CSV_PATH}")

    rows: list[dict[str, float]] = []
    with INPUT_CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = float(row["alpha"])

            fullbatch_mean = float(row["fullbatch_mean"])
            fullbatch_sem = float(row["fullbatch_sem"])

            batch2000_mean = float(row["batch2000_mean"])
            batch2000_sem = float(row["batch2000_sem"])

            batch4000_mean = float(row["batch4000_mean"])
            batch4000_sem = float(row["batch4000_sem"])

            diff_2000 = batch2000_mean - fullbatch_mean
            diff_4000 = batch4000_mean - fullbatch_mean

            # Treat the two estimates as independent for a simple visual error bar.
            diff_2000_sem = math.sqrt(batch2000_sem**2 + fullbatch_sem**2)
            diff_4000_sem = math.sqrt(batch4000_sem**2 + fullbatch_sem**2)

            rows.append(
                {
                    "alpha": alpha,
                    "fullbatch_mean": fullbatch_mean,
                    "batch2000_minus_fullbatch": diff_2000,
                    "batch2000_minus_fullbatch_sem": diff_2000_sem,
                    "batch4000_minus_fullbatch": diff_4000,
                    "batch4000_minus_fullbatch_sem": diff_4000_sem,
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {INPUT_CSV_PATH}")

    return rows


def save_difference_csv(rows: list[dict[str, float]]) -> None:
    with OUTPUT_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "fullbatch_mean",
                "batch2000_minus_fullbatch",
                "batch2000_minus_fullbatch_sem",
                "batch4000_minus_fullbatch",
                "batch4000_minus_fullbatch_sem",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_differences(rows: list[dict[str, float]]) -> None:
    alphas = [row["alpha"] for row in rows]
    diff_2000 = [row["batch2000_minus_fullbatch"] for row in rows]
    diff_2000_sem = [row["batch2000_minus_fullbatch_sem"] for row in rows]
    diff_4000 = [row["batch4000_minus_fullbatch"] for row in rows]
    diff_4000_sem = [row["batch4000_minus_fullbatch_sem"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.errorbar(
        alphas,
        diff_2000,
        yerr=diff_2000_sem,
        fmt="s-",
        color="#D81B60",
        linewidth=2,
        markersize=6,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        label="batch_size=2000 - fullbatch",
    )
    ax.errorbar(
        alphas,
        diff_4000,
        yerr=diff_4000_sem,
        fmt="^-",
        color="#43A047",
        linewidth=2,
        markersize=6,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        label="batch_size=4000 - fullbatch",
    )

    ax.set_xlabel(r"$\alpha$", fontsize=14)
    ax.set_ylabel("difference_cosine (minibatch - fullbatch)", fontsize=14)
    ax.set_title("Minibatch Minus Full-batch Cosine Difference", fontsize=16)
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_difference_series()
    save_difference_csv(rows)
    plot_differences(rows)

    print(f"Saved difference plot: {OUTPUT_PLOT_PATH}")
    print(f"Saved difference csv: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
