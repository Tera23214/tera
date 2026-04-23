#!/usr/bin/env python3
"""Overlay cosine-vs-alpha summaries from multiple experiment folders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "Result_data"
    / "cosine_vs_alpha"
    / "M_fixed"
    / "M_100"
)


def _to_float(value: str) -> float:
    return float(value.strip())


def read_summary(exp_dir: Path) -> list[dict[str, object]]:
    """Read either metrics.csv or alpha_summary.csv and normalize columns."""
    candidates = [
        (
            exp_dir / "metrics.csv",
            "cosine_similarity_mean",
            "cosine_similarity_std",
        ),
        (
            exp_dir / "alpha_summary.csv",
            "mean_cosine_similarity",
            "std_cosine_similarity",
        ),
    ]

    for csv_path, mean_col, std_col in candidates:
        if not csv_path.exists():
            continue

        rows: list[dict[str, object]] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("alpha"):
                    continue
                rows.append(
                    {
                        "label": exp_dir.name,
                        "alpha": _to_float(row["alpha"]),
                        "mean_cosine_similarity": _to_float(row[mean_col]),
                        "std_cosine_similarity": _to_float(row.get(std_col) or "0"),
                        "source_file": str(csv_path),
                    }
                )

        return sorted(rows, key=lambda r: float(r["alpha"]))

    return []


def discover_experiments(root: Path) -> dict[str, list[dict[str, object]]]:
    experiments: dict[str, list[dict[str, object]]] = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = read_summary(exp_dir)
        if rows:
            experiments[exp_dir.name] = rows
    return experiments


def write_combined_csv(output_path: Path, experiments: dict[str, list[dict[str, object]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "alpha",
        "mean_cosine_similarity",
        "std_cosine_similarity",
        "source_file",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in sorted(experiments):
            for row in experiments[label]:
                writer.writerow(row)


def plot_overlay(output_path: Path, experiments: dict[str, list[dict[str, object]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, rows in experiments.items():
        alphas = [float(row["alpha"]) for row in rows]
        means = [float(row["mean_cosine_similarity"]) for row in rows]
        stds = [float(row["std_cosine_similarity"]) for row in rows]
        plt.errorbar(
            alphas,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            linewidth=1.8,
            markersize=4,
            label=label,
        )

    plt.xlabel("alpha")
    plt.ylabel("mean cosine similarity")
    plt.title("Cosine similarity vs alpha")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay cosine-vs-alpha curves from experiment subdirectories."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the overlay plot and combined CSV.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "comparison_overlay").resolve()
    experiments = discover_experiments(root)
    if not experiments:
        raise SystemExit(f"No readable summaries found under {root}")

    plot_path = output_dir / "cosine_vs_alpha_overlay.png"
    csv_path = output_dir / "cosine_vs_alpha_overlay_summary.csv"
    plot_overlay(plot_path, experiments)
    write_combined_csv(csv_path, experiments)

    print(f"Read {len(experiments)} experiment folders from {root}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
