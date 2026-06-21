#!/usr/bin/env python3
"""Plot SE_t time-series results as m_y vs step for each lambda."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "SE_t_results.dat"
    parser = argparse.ArgumentParser(
        description="Plot m_y vs step from tera/SE_t output."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=default_input,
        help="Input .dat file with columns: lambda t m_w m_x m_y.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to INPUT stem + '_my_vs_step.png'.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=(0.0, 1.0),
        help="Set y-axis limits. Defaults to 0 1.",
    )
    parser.add_argument(
        "--title",
        default="SE time evolution: step vs m_y",
        help="Plot title.",
    )
    return parser.parse_args()


def load_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = np.genfromtxt(path, names=True, dtype=float, encoding=None)
    data = np.atleast_1d(data)

    required = {"lambda", "t", "m_y"}
    names = set(data.dtype.names or [])
    missing = required - names
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Found: {', '.join(data.dtype.names or [])}"
        )
    return data


def make_plot(
    data: np.ndarray,
    output: Path,
    title: str,
    log_y: bool,
    ylim: tuple[float, float],
) -> None:
    lambda_values = np.unique(data["lambda"])
    lambda_values.sort()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(lambda_values)))

    for color, lam in zip(colors, lambda_values):
        subset = data[data["lambda"] == lam]
        order = np.argsort(subset["t"])
        subset = subset[order]
        ax.plot(
            subset["t"],
            subset["m_y"],
            linewidth=2,
            color=color,
            label=f"lambda={lam:g}",
        )

    ax.set_xlabel("step t")
    ax.set_ylabel("m_y")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_my_vs_step.png")

    data = load_table(input_path)
    make_plot(
        data=data,
        output=output_path,
        title=args.title,
        log_y=args.log_y,
        ylim=tuple(args.ylim),
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
