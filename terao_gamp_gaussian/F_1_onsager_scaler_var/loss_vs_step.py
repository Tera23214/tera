#!/usr/bin/env python
"""
Plot loss vs step for the F=1 Onsager G-AMP experiment.

This script runs multiple replicas at fixed alpha and saves:
- Mean loss vs step (linear scale)
- Mean log10(loss) vs step
- CSV files with per-step histories and replica summaries
"""

import argparse
import math
import sys
import time
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

from terao_gamp_gaussian.F_1_onsager_scaler_var.core import (
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)

#set the parameters for the experiment
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss vs step for fixed alpha.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--N1", type=int, default=1000)
    parser.add_argument("--N2", type=int, default=1000)
    parser.add_argument("--M", type=int, default=200)
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
    parser.add_argument("--noise-var", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=3)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    return parser.parse_args()


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_convergence_step(loss_history: np.ndarray, threshold: float) -> float:
    if loss_history.size < 2:
        return float("nan")

    delta = np.abs(np.diff(loss_history))
    stable_idx = np.where(delta < threshold)[0]
    if stable_idx.size == 0:
        return float("nan")

    return float(stable_idx[0] + 2)


def count_observed_edges(alpha: float, N1: int, N2: int, M: int) -> int:
    c1 = int(round(alpha * M))
    c1 = max(1, min(c1, N2))
    return N1 * c1


def save_config(results_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    num_observations = count_observed_edges(args.alpha, args.N1, args.N2, args.M)
    config = {
        "algorithm": "gamp_F_1_onsager_loss_vs_step",
        "alpha": args.alpha,
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "max_steps": args.max_steps,
        "damping": args.damping,
        "damping_schedule": args.damping_schedule,
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "noise_var": args.noise_var,
        "teacher_seed": 1,
        "graph_seed": 1,
        "noise_seed": 1,
        "student_seed_base": 100,
        "legacy_cli_seed": args.seed,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "loss_eval_interval": 1,
        "early_stop": False,
        "loss_definition": "sum_squared_error_on_observed_entries",
        "clean_loss_definition": "sum_squared_error_on_observed_entries_against_noise_free_targets",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "num_observed_entries": num_observations,
        "device": str(device),
    }

    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_loss_history(
    results_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    all_clean_losses: np.ndarray,
    all_qys: np.ndarray,
) -> None:
    history_path = results_dir / "loss_history.csv"
    mean_loss = all_losses.mean(axis=0)
    std_loss = all_losses.std(axis=0)
    log_losses = np.log10(np.clip(all_losses, 1e-30, None))
    mean_log_loss = log_losses.mean(axis=0)
    std_log_loss = log_losses.std(axis=0)
    mean_clean_loss = all_clean_losses.mean(axis=0)
    std_clean_loss = all_clean_losses.std(axis=0)
    log_clean_losses = np.log10(np.clip(all_clean_losses, 1e-30, None))
    mean_log_clean_loss = log_clean_losses.mean(axis=0)
    std_log_clean_loss = log_clean_losses.std(axis=0)
    mean_qy = all_qys.mean(axis=0)
    std_qy = all_qys.std(axis=0)

    header = [
        "step",
        "loss_mean",
        "loss_std",
        "log10_loss_mean",
        "log10_loss_std",
        "clean_loss_mean",
        "clean_loss_std",
        "log10_clean_loss_mean",
        "log10_clean_loss_std",
        "qy_mean",
        "qy_std",
    ]
    header.extend([f"loss_replica_{idx + 1}" for idx in range(all_losses.shape[0])])
    header.extend(
        [f"clean_loss_replica_{idx + 1}" for idx in range(all_clean_losses.shape[0])]
    )
    header.extend([f"qy_replica_{idx + 1}" for idx in range(all_qys.shape[0])])

    with open(history_path, "w") as f:
        f.write(",".join(header) + "\n")
        for step_idx, step in enumerate(steps):
            row = [
                str(int(step)),
                f"{mean_loss[step_idx]:.10e}",
                f"{std_loss[step_idx]:.10e}",
                f"{mean_log_loss[step_idx]:.10e}",
                f"{std_log_loss[step_idx]:.10e}",
                f"{mean_clean_loss[step_idx]:.10e}",
                f"{std_clean_loss[step_idx]:.10e}",
                f"{mean_log_clean_loss[step_idx]:.10e}",
                f"{std_log_clean_loss[step_idx]:.10e}",
                f"{mean_qy[step_idx]:.10e}",
                f"{std_qy[step_idx]:.10e}",
            ]
            row.extend(f"{loss[step_idx]:.10e}" for loss in all_losses)
            row.extend(f"{loss[step_idx]:.10e}" for loss in all_clean_losses)
            row.extend(f"{qy[step_idx]:.10e}" for qy in all_qys)
            f.write(",".join(row) + "\n")


def save_replica_summary(
    results_dir: Path,
    seeds: list[int],
    runtimes: list[float],
    final_losses: list[float],
    final_clean_losses: list[float],
    convergence_steps: list[float],
    qy_values: list[float],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "replica,seed,runtime_sec,final_loss,final_clean_loss,"
            "estimated_convergence_step,qy\n"
        )
        for idx, seed in enumerate(seeds):
            convergence_value = (
                "" if math.isnan(convergence_steps[idx]) else str(int(convergence_steps[idx]))
            )
            f.write(
                f"{idx + 1},{seed},{runtimes[idx]:.4f},{final_losses[idx]:.10e},"
                f"{final_clean_losses[idx]:.10e},{convergence_value},{qy_values[idx]:.10e}\n"
            )


def plot_linear_loss(
    plots_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    mean_loss: np.ndarray,
    std_loss: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, loss_curve in enumerate(all_losses):
        ax.plot(
            steps,
            loss_curve,
            color="#B0BEC5",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(steps, mean_loss, color="#1565C0", linewidth=2.5, label="Mean loss")
    ax.fill_between(
        steps,
        np.maximum(mean_loss - std_loss, 0.0),
        mean_loss + std_loss,
        color="#90CAF9",
        alpha=0.35,
        label="Mean ± std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Observed squared-error sum", fontsize=13)
    ax.set_title(
        f"Observed squared-error sum vs Step (alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}, "
        f"{args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_vs_step_linear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_linear_clean_loss(
    plots_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    mean_loss: np.ndarray,
    std_loss: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, loss_curve in enumerate(all_losses):
        ax.plot(
            steps,
            loss_curve,
            color="#CFD8DC",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(steps, mean_loss, color="#6A1B9A", linewidth=2.5, label="Mean clean loss")
    ax.fill_between(
        steps,
        np.maximum(mean_loss - std_loss, 0.0),
        mean_loss + std_loss,
        color="#CE93D8",
        alpha=0.35,
        label="Mean ± std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Observed squared-error sum", fontsize=13)
    ax.set_title(
        f"Observed squared-error sum vs Step (clean target, alpha={args.alpha}, "
        f"N1={args.N1}, N2={args.N2}, M={args.M}, {args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_clean_vs_step_linear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_log_loss(
    plots_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    mean_log_loss: np.ndarray,
    std_log_loss: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    log_losses = np.log10(np.clip(all_losses, 1e-30, None))

    for idx, log_curve in enumerate(log_losses):
        ax.plot(
            steps,
            log_curve,
            color="#B0BEC5",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(steps, mean_log_loss, color="#D84315", linewidth=2.5, label="Mean log10(loss)")
    ax.fill_between(
        steps,
        mean_log_loss - std_log_loss,
        mean_log_loss + std_log_loss,
        color="#FFAB91",
        alpha=0.35,
        label="Mean ± std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("log10(Observed squared-error sum)", fontsize=13)
    ax.set_title(
        f"log10(Observed squared-error sum) vs Step (alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}, "
        f"{args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_vs_step_log10.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_log_clean_loss(
    plots_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    mean_log_loss: np.ndarray,
    std_log_loss: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    log_losses = np.log10(np.clip(all_losses, 1e-30, None))

    for idx, log_curve in enumerate(log_losses):
        ax.plot(
            steps,
            log_curve,
            color="#CFD8DC",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(steps, mean_log_loss, color="#8E24AA", linewidth=2.5, label="Mean log10(clean loss)")
    ax.fill_between(
        steps,
        mean_log_loss - std_log_loss,
        mean_log_loss + std_log_loss,
        color="#E1BEE7",
        alpha=0.35,
        label="Mean ± std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("log10(Observed squared-error sum)", fontsize=13)
    ax.set_title(
        f"log10(Observed squared-error sum) vs Step (clean target, alpha={args.alpha}, "
        f"N1={args.N1}, N2={args.N2}, M={args.M}, {args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_clean_vs_step_log10.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qy(
    plots_dir: Path,
    steps: np.ndarray,
    all_qys: np.ndarray,
    mean_qy: np.ndarray,
    std_qy: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, qy_curve in enumerate(all_qys):
        ax.plot(
            steps,
            qy_curve,
            color="#B0BEC5",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(steps, mean_qy, color="#2E7D32", linewidth=2.5, label="Mean Q_Y")
    ax.fill_between(
        steps,
        mean_qy - std_qy,
        mean_qy + std_qy,
        color="#A5D6A7",
        alpha=0.35,
        label="Mean ± std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Q_Y", fontsize=13)
    ax.set_title(
        f"Q_Y vs Step (alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}, "
        f"{args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "qy_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = detect_device()

    print("=" * 60)
    print("Loss vs Step for G-AMP with F=1 + Onsager Correction")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}")
    if args.damping_schedule == "beta":
        print(
            f"max_steps={args.max_steps}, damping schedule: beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"max_steps={args.max_steps}, damping={args.damping}")
    print("Teacher / graph / noise seed: 1")
    print("Student seed rule: 100 + replica_index")
    if args.seed != 1:
        print(f"Legacy CLI seed argument {args.seed} is ignored by this fixed seed policy.")
    print(f"replicas={args.num_replicas}")
    print()

    num_observations = count_observed_edges(args.alpha, args.N1, args.N2, args.M)
    print(f"Observed entries: {num_observations}")
    print("Loss definition: sum of squared errors on observed entries")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_loss_vs_step_alpha{args.alpha}_"
        f"{args.N1}x{args.N2}_M{args.M}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config(results_dir, args, device)

    all_losses = []
    all_clean_losses = []
    all_qys = []
    qy_values = []
    final_losses = []
    final_clean_losses = []
    runtimes = []
    seeds = []
    convergence_steps = []

    total_start = time.time()
    shared_seed = 1
    student_seed_base = 100
    global_data = prepare_global_shared_data(
        device=device,
        seed=shared_seed,
        N1=args.N1,
        N2=args.N2,
        M=args.M,
        noise_var=args.noise_var,
    )
    shared_data = prepare_shared_alpha_data(
        alpha=args.alpha,
        device=device,
        seed=shared_seed,
        N1=args.N1,
        N2=args.N2,
        M=args.M,
        noise_var=args.noise_var,
        global_data=global_data,
    )

    for replica_idx in range(args.num_replicas):
        seed = student_seed_base + replica_idx
        replica_start = time.time()

        qy, final_loss, steps_taken, history = train_single_replica(
            alpha=args.alpha,
            device=device,
            seed=seed,
            N1=args.N1,
            N2=args.N2,
            M=args.M,
            max_steps=args.max_steps,
            damping=args.damping,
            use_step_damping=args.damping_schedule == "beta",
            damping_beta_scale=args.beta_scale,
            damping_beta_max=args.beta_max,
            noise_var=args.noise_var,
            convergence_threshold=args.convergence_threshold,
            return_history=True,
            loss_eval_interval=1,
            early_stop=False,
            shared_data=shared_data,
        )

        runtime = time.time() - replica_start
        loss_history = np.asarray(history["loss"], dtype=np.float64) * num_observations
        clean_loss_history = (
            np.asarray(history["loss_clean"], dtype=np.float64) * num_observations
        )
        qy_history = np.asarray(history["qy"], dtype=np.float64)
        step_history = np.asarray(history["steps"], dtype=np.int64)
        convergence_step = estimate_convergence_step(
            loss_history, args.convergence_threshold
        )

        all_losses.append(loss_history)
        all_clean_losses.append(clean_loss_history)
        all_qys.append(qy_history)
        qy_values.append(qy)
        final_losses.append(final_loss * num_observations)
        final_clean_losses.append(float(clean_loss_history[-1]))
        runtimes.append(runtime)
        seeds.append(seed)
        convergence_steps.append(convergence_step)

        convergence_text = (
            "not reached" if math.isnan(convergence_step) else str(int(convergence_step))
        )
        print(
            f"Replica {replica_idx + 1}/{args.num_replicas}: "
            f"seed={seed}, final_loss={final_loss * num_observations:.2e}, "
            f"final_clean_loss={clean_loss_history[-1]:.2e}, "
            f"final_qy={qy:.4f}, "
            f"estimated_convergence_step={convergence_text}, "
            f"steps_recorded={steps_taken}, runtime={runtime:.1f}s"
        )

    total_runtime = time.time() - total_start

    all_losses_arr = np.asarray(all_losses, dtype=np.float64)
    all_clean_losses_arr = np.asarray(all_clean_losses, dtype=np.float64)
    all_qys_arr = np.asarray(all_qys, dtype=np.float64)
    steps = step_history
    mean_loss = all_losses_arr.mean(axis=0)
    std_loss = all_losses_arr.std(axis=0)
    log_losses = np.log10(np.clip(all_losses_arr, 1e-30, None))
    mean_log_loss = log_losses.mean(axis=0)
    std_log_loss = log_losses.std(axis=0)
    clean_log_losses = np.log10(np.clip(all_clean_losses_arr, 1e-30, None))
    mean_clean_loss = all_clean_losses_arr.mean(axis=0)
    std_clean_loss = all_clean_losses_arr.std(axis=0)
    mean_log_clean_loss = clean_log_losses.mean(axis=0)
    std_log_clean_loss = clean_log_losses.std(axis=0)
    mean_qy = all_qys_arr.mean(axis=0)
    std_qy = all_qys_arr.std(axis=0)

    save_loss_history(results_dir, steps, all_losses_arr, all_clean_losses_arr, all_qys_arr)
    save_replica_summary(
        results_dir, seeds, runtimes, final_losses, final_clean_losses, convergence_steps, qy_values
    )
    plot_linear_loss(plots_dir, steps, all_losses_arr, mean_loss, std_loss, args)
    plot_linear_clean_loss(
        plots_dir, steps, all_clean_losses_arr, mean_clean_loss, std_clean_loss, args
    )
    plot_log_loss(plots_dir, steps, all_losses_arr, mean_log_loss, std_log_loss, args)
    plot_log_clean_loss(
        plots_dir,
        steps,
        all_clean_losses_arr,
        mean_log_clean_loss,
        std_log_clean_loss,
        args,
    )
    plot_qy(plots_dir, steps, all_qys_arr, mean_qy, std_qy, args)

    print()
    print(f"Mean final loss: {np.mean(final_losses):.2e}")
    print(f"Mean final clean loss: {np.mean(final_clean_losses):.2e}")
    print(f"Mean Q_Y: {np.mean(qy_values):.4f}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
