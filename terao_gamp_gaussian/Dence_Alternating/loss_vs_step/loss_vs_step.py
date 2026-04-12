#!/usr/bin/env python
"""
Plot loss vs step for the dense-mask alternating F=1 Onsager G-AMP experiment
with cosine-similarity evaluation.

One recorded step means one full W -> X sweep.
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
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Dence_Alternating.core import (
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss vs step for fixed alpha (alternating dense mask)."
    )
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--N1", type=int, default=2000)
    parser.add_argument("--N2", type=int, default=2000)
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--damping", type=float, default=0)
    parser.add_argument(
        "--damping-schedule",
        type=str,
        choices=["beta", "constant"],
        default="constant",
    )
    parser.add_argument("--beta-scale", type=float, default=1e-2)
    parser.add_argument("--beta-max", type=float, default=0.4)
    parser.add_argument("--noise-var", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=1)
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


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    config = {
        "algorithm": "gamp_Dence_Alternating_cosine_loss_vs_step",
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
        "evaluation_metric": "cosine_similarity_in_Y_space",
        "update_scheme": "alternating_W_then_X",
        "step_definition": "one_W_update_plus_one_X_update",
        "onsager_memory_schedule": "half_step",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "dense_mask": True,
        "device": str(device),
    }

    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_loss_history(
    results_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    all_cosine_similarities: np.ndarray,
) -> None:
    history_path = results_dir / "loss_history.csv"
    mean_loss = all_losses.mean(axis=0)
    std_loss = all_losses.std(axis=0)
    log_losses = np.log10(np.clip(all_losses, 1e-30, None))
    mean_log_loss = log_losses.mean(axis=0)
    std_log_loss = log_losses.std(axis=0)
    mean_cosine_similarity = all_cosine_similarities.mean(axis=0)
    std_cosine_similarity = all_cosine_similarities.std(axis=0)

    header = [
        "step",
        "loss_mean",
        "loss_std",
        "log10_loss_mean",
        "log10_loss_std",
        "cosine_similarity_mean",
        "cosine_similarity_std",
    ]
    header.extend([f"loss_replica_{idx + 1}" for idx in range(all_losses.shape[0])])
    header.extend(
        [
            f"cosine_similarity_replica_{idx + 1}"
            for idx in range(all_cosine_similarities.shape[0])
        ]
    )

    with open(history_path, "w") as f:
        f.write(",".join(header) + "\n")
        for step_idx, step in enumerate(steps):
            row = [
                str(int(step)),
                f"{mean_loss[step_idx]:.10e}",
                f"{std_loss[step_idx]:.10e}",
                f"{mean_log_loss[step_idx]:.10e}",
                f"{std_log_loss[step_idx]:.10e}",
                f"{mean_cosine_similarity[step_idx]:.10e}",
                f"{std_cosine_similarity[step_idx]:.10e}",
            ]
            row.extend(f"{loss[step_idx]:.10e}" for loss in all_losses)
            row.extend(
                f"{value[step_idx]:.10e}" for value in all_cosine_similarities
            )
            f.write(",".join(row) + "\n")


def save_replica_summary(
    results_dir: Path,
    seeds: list[int],
    runtimes: list[float],
    final_losses: list[float],
    convergence_steps: list[float],
    cosine_similarity_values: list[float],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "replica,seed,runtime_sec,final_loss,estimated_convergence_step,"
            "cosine_similarity\n"
        )
        for idx, seed in enumerate(seeds):
            convergence_value = (
                "" if math.isnan(convergence_steps[idx]) else str(int(convergence_steps[idx]))
            )
            f.write(
                f"{idx + 1},{seed},{runtimes[idx]:.4f},{final_losses[idx]:.10e},"
                f"{convergence_value},{cosine_similarity_values[idx]:.10e}\n"
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
        label="Mean +- std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Observed MSE", fontsize=13)
    ax.set_title(
        f"Observed MSE vs Step (alpha={args.alpha}, N1={args.N1}, "
        f"N2={args.N2}, M={args.M}, {args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_vs_step_linear.png", dpi=150, bbox_inches="tight")
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

    ax.plot(
        steps,
        mean_log_loss,
        color="#D84315",
        linewidth=2.5,
        label="Mean log10(loss)",
    )
    ax.fill_between(
        steps,
        mean_log_loss - std_log_loss,
        mean_log_loss + std_log_loss,
        color="#FFAB91",
        alpha=0.35,
        label="Mean +- std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("log10(Observed MSE)", fontsize=13)
    ax.set_title(
        f"log10(Observed MSE) vs Step (alpha={args.alpha}, "
        f"N1={args.N1}, N2={args.N2}, M={args.M}, {args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_vs_step_log10.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_similarity(
    plots_dir: Path,
    steps: np.ndarray,
    all_cosine_similarities: np.ndarray,
    mean_cosine_similarity: np.ndarray,
    std_cosine_similarity: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, cosine_similarity_curve in enumerate(all_cosine_similarities):
        ax.plot(
            steps,
            cosine_similarity_curve,
            color="#B0BEC5",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(
        steps,
        mean_cosine_similarity,
        color="#2E7D32",
        linewidth=2.5,
        label="Mean cosine similarity",
    )
    ax.fill_between(
        steps,
        mean_cosine_similarity - std_cosine_similarity,
        mean_cosine_similarity + std_cosine_similarity,
        color="#A5D6A7",
        alpha=0.35,
        label="Mean +- std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title(
        f"Cosine Similarity vs Step (alpha={args.alpha}, N1={args.N1}, "
        f"N2={args.N2}, M={args.M}, {args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(
        plots_dir / "cosine_similarity_vs_step.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = detect_device()

    print("=" * 60)
    print("Loss vs Step for Dense-mask Alternating G-AMP")
    print("Evaluation Metric: Cosine Similarity in Y-space")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}")
    if args.damping_schedule == "beta":
        print(
            f"max_steps={args.max_steps}, damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"max_steps={args.max_steps}, damping={args.damping}")
    print("Step definition: one W update followed by one X update")
    print("Onsager memory: advanced every half-step")
    print("Teacher / graph / noise seed: 1")
    print("Student seed rule: 100 + replica_index")
    print("Shared across run: teacher / noisy field")
    print("Shared per alpha: graph")
    print("Replica-specific: student initialization only")
    if args.seed != 1:
        print(f"Legacy CLI seed argument {args.seed} is ignored by this fixed seed policy.")
    print(f"replicas={args.num_replicas}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_loss_vs_step_Dence_Alternating_alpha{args.alpha}_"
        f"{args.N1}x{args.N2}_M{args.M}"
    )
    results_dir = Path(__file__).resolve().parent.parent / "results" / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config(results_dir, args, device)

    all_losses = []
    all_cosine_similarities = []
    cosine_similarity_values = []
    final_losses = []
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
    num_observations = int(shared_data["E"])

    print(f"Observed entries: {num_observations}")
    print("Loss definition: mean squared error on observed entries")
    print()

    for replica_idx in range(args.num_replicas):
        seed = student_seed_base + replica_idx
        replica_start = time.time()

        cosine_similarity, final_loss, steps_taken, history = train_single_replica(
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
        loss_history = np.asarray(history["loss"], dtype=np.float64)
        cosine_similarity_history = np.asarray(
            history["cosine_similarity"], dtype=np.float64
        )
        step_history = np.asarray(history["steps"], dtype=np.int64)
        convergence_step = estimate_convergence_step(
            loss_history, args.convergence_threshold
        )

        all_losses.append(loss_history)
        all_cosine_similarities.append(cosine_similarity_history)
        cosine_similarity_values.append(cosine_similarity)
        final_losses.append(final_loss)
        runtimes.append(runtime)
        seeds.append(seed)
        convergence_steps.append(convergence_step)

        convergence_text = (
            "not reached" if math.isnan(convergence_step) else str(int(convergence_step))
        )
        print(
            f"Replica {replica_idx + 1}/{args.num_replicas}: "
            f"seed={seed}, final_loss={final_loss:.10e}, "
            f"final_cosine_similarity={cosine_similarity:.10f}, "
            f"estimated_convergence_step={convergence_text}, "
            f"steps_recorded={steps_taken}, runtime={runtime:.1f}s"
        )

    total_runtime = time.time() - total_start

    all_losses_arr = np.asarray(all_losses, dtype=np.float64)
    all_cosine_similarities_arr = np.asarray(
        all_cosine_similarities, dtype=np.float64
    )
    steps = step_history
    mean_loss = all_losses_arr.mean(axis=0)
    std_loss = all_losses_arr.std(axis=0)
    log_losses = np.log10(np.clip(all_losses_arr, 1e-30, None))
    mean_log_loss = log_losses.mean(axis=0)
    std_log_loss = log_losses.std(axis=0)
    mean_cosine_similarity = all_cosine_similarities_arr.mean(axis=0)
    std_cosine_similarity = all_cosine_similarities_arr.std(axis=0)

    save_loss_history(results_dir, steps, all_losses_arr, all_cosine_similarities_arr)
    save_replica_summary(
        results_dir,
        seeds,
        runtimes,
        final_losses,
        convergence_steps,
        cosine_similarity_values,
    )
    plot_linear_loss(plots_dir, steps, all_losses_arr, mean_loss, std_loss, args)
    plot_log_loss(plots_dir, steps, all_losses_arr, mean_log_loss, std_log_loss, args)
    plot_cosine_similarity(
        plots_dir,
        steps,
        all_cosine_similarities_arr,
        mean_cosine_similarity,
        std_cosine_similarity,
        args,
    )

    print()
    print(f"Mean final loss: {np.mean(final_losses):.10e}")
    print(f"Mean cosine similarity: {np.mean(cosine_similarity_values):.10f}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
