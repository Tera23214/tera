#!/usr/bin/env python
"""
G-AMP Parallel Simulation Runner with F=1 and Onsager Correction.

Runs G-AMP replicas in parallel using ProcessPoolExecutor for CPU-based
multiprocessing.  Each worker process limits PyTorch's internal thread
count to 1 via torch.set_num_threads(1) so that total CPU utilisation
equals the number of worker processes (no oversubscription).

Usage:
    python -m terao_gamp_gaussian.F_1_onsager_scaler_var.run_gamp_parallel \\
        --num-workers 8

Output format is identical to run_gamp.py (config.yaml, metrics.csv, plots).
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


# ============================================================================
# Worker function — runs in a subprocess
# ============================================================================


def _run_replica(kwargs: dict) -> dict:
    """Run a single replica in a worker process.

    Must be a top-level function (picklable for multiprocessing).
    """
    # Limit PyTorch threads to 1 inside each worker to avoid oversubscription.
    torch.set_num_threads(1)

    # Lazy import inside the worker so that the module is loaded after
    # torch.set_num_threads has been called.
    from terao_gamp_gaussian.F_1_onsager_scaler_var.core import train_single_replica

    alpha = kwargs["alpha"]
    replica_id = kwargs["replica_id"]
    seed = kwargs["seed"]
    device = torch.device(kwargs["device"])

    t0 = time.time()

    qy, final_loss, steps_taken = train_single_replica(
        alpha=alpha,
        device=device,
        seed=seed,
        N1=kwargs["N1"],
        N2=kwargs["N2"],
        M=kwargs["M"],
        max_steps=kwargs["max_steps"],
        damping=kwargs["damping"],
        use_step_damping=kwargs["use_step_damping"],
        damping_beta_scale=kwargs["damping_beta_scale"],
        damping_beta_max=kwargs["damping_beta_max"],
        noise_var=kwargs["noise_var"],
        convergence_threshold=kwargs["convergence_threshold"],
    )

    runtime = time.time() - t0

    return {
        "alpha": alpha,
        "replica_id": replica_id,
        "seed": seed,
        "qy": qy,
        "final_loss": final_loss,
        "steps_taken": steps_taken,
        "runtime": runtime,
    }


# ============================================================================
# Argument parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel G-AMP runner with F=1 + Onsager."
    )
    parser.add_argument("--N1", type=int, default=2520)
    parser.add_argument("--N2", type=int, default=2520)
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--alpha-start", type=float, default=0.5)
    parser.add_argument("--alpha-stop", type=float, default=3.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument(
        "--damping-schedule",
        type=str,
        choices=["beta", "constant"],
        default="beta",
    )
    parser.add_argument("--beta-scale", type=float, default=1e-3)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--noise-var", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-replicas", type=int, default=30)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel worker processes (default: CPU count).",
    )
    return parser.parse_args()


# ============================================================================
# Output helpers (identical to run_gamp.py)
# ============================================================================


def save_config(
    results_dir: Path, args: argparse.Namespace, device: str
) -> None:
    config = {
        "algorithm": "gamp_F_1_onsager_parallel",
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "alpha_start": args.alpha_start,
        "alpha_stop": args.alpha_stop,
        "alpha_step": args.alpha_step,
        "max_steps": args.max_steps,
        "damping": args.damping,
        "damping_schedule": args.damping_schedule,
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "noise_var": args.noise_var,
        "seed": args.seed,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "num_workers": args.num_workers,
        "device": device,
        "onsager_correction": True,
        "F_type": "constant_1",
    }
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics_csv(
    results_dir: Path,
    results: dict,
    alphas_list: list[float],
    num_replicas: int,
) -> None:
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        header = "alpha,Q_Y_mean,Q_Y_std,Loss_mean,Loss_std,Steps_mean"
        for i in range(num_replicas):
            header += f",qy_replica_{i}"
        f.write(header + "\n")

        for alpha in alphas_list:
            r = results[alpha]
            line = (
                f"{alpha},{r['qy_mean']},{r['qy_std']},"
                f"{r['loss_mean']},{r['loss_std']},{r['steps_mean']}"
            )
            for qy_v in r["qy_values"]:
                line += f",{qy_v}"
            f.write(line + "\n")


def plot_qy_vs_alpha(
    plots_dir: Path,
    results: dict,
    alphas_list: list[float],
    args: argparse.Namespace,
) -> None:
    qy_means = [results[a]["qy_mean"] for a in alphas_list]
    qy_stds = [results[a]["qy_std"] for a in alphas_list]
    qy_sems = [std / math.sqrt(args.num_replicas) for std in qy_stds]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.errorbar(
        alphas_list,
        qy_means,
        yerr=qy_sems,
        fmt="o-",
        color="#1976D2",
        markersize=6,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label="G-AMP (F=1 + Onsager)",
    )
    ax.set_xlabel(r"$\alpha$ (observation density)", fontsize=14)
    ax.set_ylabel(r"$Q_Y$", fontsize=14)
    ax.set_title(
        f"Phase Transition (G-AMP with F=1 + Onsager)\n"
        f"({args.N1}×{args.N2}, M={args.M}, {args.max_steps} steps)",
        fontsize=16,
    )
    ax.set_xlim(args.alpha_start - 0.1, args.alpha_stop + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    plt.savefig(plots_dir / "qy_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    device = "cpu"  # CPU is optimal for scatter_add workloads

    print("=" * 60)
    print("G-AMP with F=1 + Onsager Correction (PARALLEL)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Workers: {args.num_workers}")
    print(f"Matrix: {args.N1}×{args.N2}, M={args.M}")
    print(
        f"Alpha: {args.alpha_start} ~ {args.alpha_stop} "
        f"(step {args.alpha_step})"
    )
    if args.damping_schedule == "beta":
        print(
            f"Steps: {args.max_steps}, Damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"Steps: {args.max_steps}, Damping: {args.damping}")
    print(f"Replicas per alpha: {args.num_replicas}")
    print()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_gamp_F_1_onsager_{args.N1}x{args.M}"
        f"_alpha{args.alpha_start}-{args.alpha_stop}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config(results_dir, args, device)
    print(f"Results directory: {results_dir}")

    # Build task list: all (alpha, replica) combinations
    alphas = np.arange(
        args.alpha_start,
        args.alpha_stop + args.alpha_step / 2,
        args.alpha_step,
    )

    use_step_damping = args.damping_schedule == "beta"

    tasks = []
    for alpha in alphas:
        for replica_id in range(args.num_replicas):
            tasks.append(
                {
                    "alpha": float(alpha),
                    "replica_id": replica_id,
                    "seed": args.seed + replica_id * 1000,
                    "device": device,
                    "N1": args.N1,
                    "N2": args.N2,
                    "M": args.M,
                    "max_steps": args.max_steps,
                    "damping": args.damping,
                    "use_step_damping": use_step_damping,
                    "damping_beta_scale": args.beta_scale,
                    "damping_beta_max": args.beta_max,
                    "noise_var": args.noise_var,
                    "convergence_threshold": args.convergence_threshold,
                }
            )

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks} ({len(alphas)} alphas × {args.num_replicas} replicas)")
    print()

    # Parallel execution
    completed = 0
    all_results: list[dict] = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_run_replica, t): t for t in tasks}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            completed += 1

            elapsed = time.time() - start_time
            eta = elapsed / completed * (total_tasks - completed) if completed else 0

            print(
                f"[{completed}/{total_tasks}] "
                f"α={result['alpha']:.2f}, replica {result['replica_id']+1}: "
                f"Q_Y={result['qy']:.4f}, Loss={result['final_loss']:.2e}, "
                f"Steps={result['steps_taken']} ({result['runtime']:.1f}s) "
                f"ETA: {eta/3600:.1f}h"
            )

    total_time = time.time() - start_time

    # Aggregate results by alpha
    results = {}
    for alpha in alphas:
        alpha_f = float(alpha)
        alpha_results = [r for r in all_results if r["alpha"] == alpha_f]
        # Sort by replica_id to ensure deterministic ordering
        alpha_results.sort(key=lambda r: r["replica_id"])

        qy_values = [r["qy"] for r in alpha_results]
        loss_values = [r["final_loss"] for r in alpha_results]
        steps_values = [r["steps_taken"] for r in alpha_results]

        results[alpha_f] = {
            "qy_mean": np.mean(qy_values),
            "qy_std": np.std(qy_values),
            "qy_values": qy_values,
            "loss_mean": np.mean(loss_values),
            "loss_std": np.std(loss_values),
            "steps_mean": np.mean(steps_values),
        }

    # Print summary
    print("\n" + "=" * 60)
    print("Results (mean ± std)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'Q_Y':^20} | {'Loss':^20} | {'Steps':>8}")
    print("-" * 60)
    alphas_list = sorted(results.keys())
    for alpha in alphas_list:
        r = results[alpha]
        print(
            f"{alpha:6.2f} | {r['qy_mean']:8.4f} ± {r['qy_std']:<8.4f} | "
            f"{r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | "
            f"{r['steps_mean']:8.0f}"
        )

    print(f"\nTotal time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Workers: {args.num_workers}")
    print("=" * 60)

    # Save outputs
    save_metrics_csv(results_dir, results, alphas_list, args.num_replicas)
    plot_qy_vs_alpha(plots_dir, results, alphas_list, args)

    print(f"\nResults saved to: {results_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
