#!/usr/bin/env python
"""
Plot cosine similarity vs step for multiple informative-initialization runs.

By default this script scans the current directory for subdirectories containing
``loss_history.csv`` and overlays their ``cosine_similarity_mean`` curves on a
single figure. The alpha value is parsed from the directory name.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHA_PATTERN = re.compile(r"alpha([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay cosine_similarity_mean vs step for multiple alpha runs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing alpha-specific result subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input-dir>/cosine_vs_step_multi_alpha.png.",
    )
    return parser.parse_args()


def extract_alpha(path: Path) -> float:
    match = ALPHA_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not parse alpha from directory name: {path.name}")
    return float(match.group(1))


def load_curve(csv_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    cosine: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            cosine.append(float(row["cosine_similarity_mean"]))

    if not steps:
        raise ValueError(f"No rows found in {csv_path}")

    return steps, cosine


def discover_runs(input_dir: Path) -> list[tuple[float, Path]]:
    runs: list[tuple[float, Path]] = []

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "loss_history.csv"
        if not csv_path.exists():
            continue
        alpha = extract_alpha(subdir)
        runs.append((alpha, csv_path))

    runs.sort(key=lambda item: item[0])
    return runs


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "cosine_vs_step_multi_alpha.png"
    )

    runs = discover_runs(input_dir)
    if not runs:
        raise RuntimeError(f"No loss_history.csv files found under {input_dir}")

    fig, ax = plt.subplots(figsize=(10, 7))

    for alpha, csv_path in runs:
        steps, cosine = load_curve(csv_path)
        ax.plot(steps, cosine, linewidth=2.0, label=f"alpha={alpha:.1f}")

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title("Cosine Similarity vs Step (Informative Init)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
