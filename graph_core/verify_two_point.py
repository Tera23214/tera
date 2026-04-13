#!/usr/bin/env python
"""
Verification utilities for the two-point heterogeneous row-degree graph model.

This script generates a graph from (alpha, p, r=ca/cb), measures the realized
row-degree distribution, and reports whether the graph is actually
heterogeneous. For small graphs it also prints the full row-column structure so
the graph shape can be inspected directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from graph_core.two_point import generate_two_point_dense_mask
except ImportError:
    from two_point import generate_two_point_dense_mask


def analyze_two_point_graph(
    N1: int,
    N2: int,
    M: int,
    alpha: float,
    p: float,
    r: float,
    seed: int,
    device: torch.device,
    include_edge_list: bool = False,
    include_row_columns: bool = False,
    include_all_pairs: bool = False,
    include_adjacency_matrix: bool = False,
) -> dict[str, object]:
    """
    Generate one two-point graph and summarize its realized degree statistics.
    """
    mask, i_idx, j_idx, E, row_degrees, ca, cb, num_ca, num_cb, p_eff, alpha_eff = (
        generate_two_point_dense_mask(
            N1=N1,
            N2=N2,
            M=M,
            alpha=alpha,
            p=p,
            r=r,
            device=device,
            seed=seed,
        )
    )

    row_degree_values = row_degrees.cpu().numpy()
    row_sums = mask.sum(dim=1).cpu().numpy().astype(np.int64)
    col_sums = mask.sum(dim=0).cpu().numpy().astype(np.int64)

    unique_row_degrees, row_degree_counts = np.unique(
        row_degree_values, return_counts=True
    )
    degree_hist = {
        int(degree): int(count)
        for degree, count in zip(unique_row_degrees, row_degree_counts, strict=True)
    }

    count_ca_realized = int(np.sum(row_degree_values == ca))
    count_cb_realized = int(np.sum(row_degree_values == cb))
    ratio_realized = float(ca / cb) if cb != 0 else float("inf")

    report: dict[str, object] = {
        "parameters": {
            "N1": N1,
            "N2": N2,
            "M": M,
            "alpha": alpha,
            "p": p,
            "r": r,
            "seed": seed,
            "device": str(device),
        },
        "resolved_degrees": {
            "ca": int(ca),
            "cb": int(cb),
            "num_ca_target": int(num_ca),
            "num_cb_target": int(num_cb),
            "p_eff": float(p_eff),
            "alpha_eff": float(alpha_eff),
            "realized_ratio_ca_over_cb": ratio_realized,
        },
        "edge_counts": {
            "E": int(E),
            "target_total_edges": int(round(alpha * M * N1)),
            "mask_sum": int(mask.sum().item()),
            "i_idx_size": int(i_idx.numel()),
            "j_idx_size": int(j_idx.numel()),
        },
        "row_distribution": {
            "unique_row_degrees": [int(v) for v in unique_row_degrees.tolist()],
            "degree_histogram": degree_hist,
            "count_ca_realized": count_ca_realized,
            "count_cb_realized": count_cb_realized,
            "row_degree_mean": float(row_degree_values.mean()) if N1 > 0 else 0.0,
            "row_degree_std": float(row_degree_values.std()) if N1 > 0 else 0.0,
            "row_degree_min": int(row_degree_values.min()) if N1 > 0 else 0,
            "row_degree_max": int(row_degree_values.max()) if N1 > 0 else 0,
        },
        "column_distribution": {
            "col_degree_mean": float(col_sums.mean()) if N2 > 0 else 0.0,
            "col_degree_std": float(col_sums.std()) if N2 > 0 else 0.0,
            "col_degree_min": int(col_sums.min()) if N2 > 0 else 0,
            "col_degree_max": int(col_sums.max()) if N2 > 0 else 0,
        },
        "checks": {
            "alpha_eff_matches_alpha_exactly": bool(np.isclose(alpha_eff, alpha, atol=1e-12)),
            "mask_sum_matches_E": bool(int(mask.sum().item()) == E),
            "E_matches_target_total_edges": bool(E == int(round(alpha * M * N1))),
            "edge_index_sizes_match_E": bool(i_idx.numel() == E and j_idx.numel() == E),
            "row_sums_match_row_degrees": bool(np.array_equal(row_sums, row_degree_values)),
            "row_degree_values_match_two_point_support": bool(
                set(unique_row_degrees.tolist()).issubset({int(ca), int(cb)})
            ),
            "heterogeneous_detected": bool(ca != cb and len(unique_row_degrees) >= 2),
        },
    }

    if include_edge_list:
        edge_pairs = list(
            zip(i_idx.cpu().tolist(), j_idx.cpu().tolist(), strict=True)
        )
        report["edge_list"] = [[int(i), int(j)] for i, j in edge_pairs]

    if include_row_columns:
        row_to_columns: dict[int, list[int]] = {}
        for row in range(N1):
            selected_cols = j_idx[i_idx == row].cpu().tolist()
            row_to_columns[row] = [int(col) for col in sorted(selected_cols)]
        report["row_to_columns"] = row_to_columns

    if include_adjacency_matrix:
        report["adjacency_matrix"] = mask.to(dtype=torch.int64).cpu().tolist()

    if include_all_pairs:
        mask_cpu = mask.to(dtype=torch.int64).cpu().numpy()
        all_pairs: list[list[int]] = []
        for row in range(N1):
            for col in range(N2):
                all_pairs.append([int(row), int(col), int(mask_cpu[row, col])])
        report["all_row_column_pairs"] = all_pairs

    return report


def print_report(report: dict[str, object]) -> None:
    """
    Print a concise human-readable verification summary.
    """
    params = report["parameters"]
    resolved = report["resolved_degrees"]
    edges = report["edge_counts"]
    row_dist = report["row_distribution"]
    col_dist = report["column_distribution"]
    checks = report["checks"]

    print("=" * 60)
    print("Two-Point Graph Verification")
    print("=" * 60)
    print(
        f"N1={params['N1']}, N2={params['N2']}, M={params['M']}, "
        f"alpha={params['alpha']}, p={params['p']}, r={params['r']}, seed={params['seed']}"
    )
    print(
        f"Resolved degrees: ca={resolved['ca']}, cb={resolved['cb']}, "
        f"num_ca={resolved['num_ca_target']}, num_cb={resolved['num_cb_target']}"
    )
    print(
        f"Realized p_eff={resolved['p_eff']:.6f}, "
        f"alpha_eff={resolved['alpha_eff']:.6f}, "
        f"ca/cb={resolved['realized_ratio_ca_over_cb']:.6f}"
    )
    print(
        f"Edges: E={edges['E']}, target_total_edges={edges['target_total_edges']}, "
        f"mask_sum={edges['mask_sum']}, "
        f"i_idx_size={edges['i_idx_size']}, j_idx_size={edges['j_idx_size']}"
    )
    print(
        f"Row degrees: unique={row_dist['unique_row_degrees']}, "
        f"mean={row_dist['row_degree_mean']:.6f}, std={row_dist['row_degree_std']:.6f}"
    )
    print(
        f"Column degrees: mean={col_dist['col_degree_mean']:.6f}, "
        f"std={col_dist['col_degree_std']:.6f}, "
        f"min={col_dist['col_degree_min']}, max={col_dist['col_degree_max']}"
    )
    print("Checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print("Row-degree histogram:")
    for degree, count in row_dist["degree_histogram"].items():
        print(f"  degree={degree}: count={count}")

    if "edge_list" in report:
        print("Observed edge list (all (row, col) pairs):")
        for row, col in report["edge_list"]:
            print(f"  ({row}, {col})")

    if "row_to_columns" in report:
        print("Observed columns for each row:")
        for row, columns in report["row_to_columns"].items():
            print(f"  row {row}: {columns}")

    if "adjacency_matrix" in report:
        print("Adjacency matrix (rows x columns, 1=observed, 0=unobserved):")
        for row_idx, row_values in enumerate(report["adjacency_matrix"]):
            print(f"  row {row_idx}: {row_values}")

    if "all_row_column_pairs" in report:
        print("All row-column pairs (row, col, observed):")
        for row, col, observed in report["all_row_column_pairs"]:
            print(f"  ({row}, {col}, {observed})")


def save_artifacts(report: dict[str, object], output_dir: Path) -> None:
    """
    Save the verification report and row-degree histogram plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "two_point_verification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    hist = report["row_distribution"]["degree_histogram"]
    degrees = [int(k) for k in hist.keys()]
    counts = [int(v) for v in hist.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(degrees, counts, color="#1f77b4", width=0.8)
    ax.set_xlabel("Row Degree")
    ax.set_ylabel("Count")
    ax.set_title("Two-Point Row-Degree Distribution")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "row_degree_histogram.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the generated two-point graph is heterogeneous."
    )
    parser.add_argument("--N1", type=int, default=20)
    parser.add_argument("--N2", type=int, default=20)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--r", type=float, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-edge-list", action="store_true")
    parser.add_argument("--show-row-columns", action="store_true")
    parser.add_argument("--show-all-pairs", action="store_true")
    parser.add_argument("--show-adjacency-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    auto_full_output = args.N1 * args.N2 <= 400

    report = analyze_two_point_graph(
        N1=args.N1,
        N2=args.N2,
        M=args.M,
        alpha=args.alpha,
        p=args.p,
        r=args.r,
        seed=args.seed,
        device=device,
        include_edge_list=args.show_edge_list or auto_full_output,
        include_row_columns=args.show_row_columns or auto_full_output,
        include_all_pairs=args.show_all_pairs or auto_full_output,
        include_adjacency_matrix=args.show_adjacency_matrix or auto_full_output,
    )
    print_report(report)

    if args.output_dir is not None:
        save_artifacts(report, args.output_dir)
        print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
