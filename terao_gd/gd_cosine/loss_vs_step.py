#!/usr/bin/env python
"""
Plot AGD loss vs step for fixed alpha, N1, N2, and M with
cosine-similarity evaluation.

This script reuses the AGD update rules from gd.py and records the observed
loss after each step so that the loss calculation can be inspected directly.
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

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph


def compute_predictions(
    W: torch.Tensor,
    X: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
) -> torch.Tensor:
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1) / math.sqrt(M)


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    return M * ((Y - Y_pred) ** 2).sum()


def agd_step_W(
    W: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    _, M = W.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    return W - lr * grad_W


def agd_step_X(
    W: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    M, _ = X.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * W_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    return X - lr * grad_X


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    mean_sq = (tensor ** 2).mean()
    if mean_sq > 0:
        return tensor / torch.sqrt(mean_sq)
    return tensor


def compute_y_cosine_similarity(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> float:
    cross_w = W_teacher.T @ W_student
    cross_x = X_student @ X_teacher.T
    inner = torch.trace(cross_w @ cross_x)

    teacher_norm_sq = torch.trace((W_teacher.T @ W_teacher) @ (X_teacher @ X_teacher.T))
    student_norm_sq = torch.trace((W_student.T @ W_student) @ (X_student @ X_student.T))
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))

    return (inner / denom).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AGD loss vs step for fixed alpha.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--N1", type=int, default=1000)
    parser.add_argument("--N2", type=int, default=1000)
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults to gd.py-style auto scaling.",
    )
    parser.add_argument("--lr-base", type=float, default=0.1)
    parser.add_argument("--noise-var", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-replicas", type=int, default=3)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    parser.add_argument("--record-interval", type=int, default=1)
    return parser.parse_args()


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_lr(args: argparse.Namespace) -> float:
    if args.lr is not None:
        return float(args.lr)
    return float(args.lr_base * (1e6 / (args.N1 * args.N2 * args.M)))


def estimate_convergence_step(
    steps: np.ndarray, loss_history: np.ndarray, threshold: float
) -> float:
    reached = np.where(loss_history < threshold)[0]
    if reached.size == 0:
        return float("nan")
    return float(steps[reached[0]])


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    lr: float,
) -> None:
    config = {
        "algorithm": "agd_cosine_loss_vs_step",
        "alpha": args.alpha,
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "max_steps": args.max_steps,
        "lr": lr,
        "lr_base": args.lr_base,
        "noise_var": args.noise_var,
        "seed": args.seed,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "record_interval": args.record_interval,
        "device": str(device),
        "student_init": "standard_normal",
        "evaluation_metric": "cosine_similarity_in_Y_space",
    }

    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_loss_history(
    results_dir: Path,
    steps: np.ndarray,
    all_losses: np.ndarray,
    all_mses: np.ndarray,
    all_cosine_similarities: np.ndarray,
) -> None:
    history_path = results_dir / "loss_history.csv"
    mean_loss = all_losses.mean(axis=0)
    std_loss = all_losses.std(axis=0)
    mean_mse = all_mses.mean(axis=0)
    std_mse = all_mses.std(axis=0)
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
        "mse_mean",
        "mse_std",
        "cosine_similarity_mean",
        "cosine_similarity_std",
    ]
    header.extend([f"loss_replica_{idx + 1}" for idx in range(all_losses.shape[0])])
    header.extend([f"mse_replica_{idx + 1}" for idx in range(all_mses.shape[0])])
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
                f"{mean_mse[step_idx]:.10e}",
                f"{std_mse[step_idx]:.10e}",
                f"{mean_cosine_similarity[step_idx]:.10e}",
                f"{std_cosine_similarity[step_idx]:.10e}",
            ]
            row.extend(f"{loss_curve[step_idx]:.10e}" for loss_curve in all_losses)
            row.extend(f"{mse_curve[step_idx]:.10e}" for mse_curve in all_mses)
            row.extend(
                f"{curve[step_idx]:.10e}" for curve in all_cosine_similarities
            )
            f.write(",".join(row) + "\n")


def save_replica_summary(
    results_dir: Path,
    seeds: list[int],
    runtimes: list[float],
    initial_losses: list[float],
    final_losses: list[float],
    final_mses: list[float],
    cosine_similarity_values: list[float],
    convergence_steps: list[float],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "replica,seed,runtime_sec,initial_loss,final_loss,final_mse,"
            "estimated_convergence_step,cosine_similarity\n"
        )
        for idx, seed in enumerate(seeds):
            convergence_value = (
                "" if math.isnan(convergence_steps[idx]) else str(int(convergence_steps[idx]))
            )
            f.write(
                f"{idx + 1},{seed},{runtimes[idx]:.4f},{initial_losses[idx]:.10e},"
                f"{final_losses[idx]:.10e},{final_mses[idx]:.10e},"
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
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(
        f"AGD Loss vs Step (alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}, "
        f"{args.num_replicas} replicas)",
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

    ax.plot(steps, mean_log_loss, color="#D84315", linewidth=2.5, label="Mean log10(loss)")
    ax.fill_between(
        steps,
        mean_log_loss - std_log_loss,
        mean_log_loss + std_log_loss,
        color="#FFAB91",
        alpha=0.35,
        label="Mean +- std",
    )

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("log10(Loss)", fontsize=13)
    ax.set_title(
        f"AGD log10(Loss) vs Step (alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}, "
        f"{args.num_replicas} replicas)",
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
        f"AGD Cosine Similarity vs Step (alpha={args.alpha}, N1={args.N1}, "
        f"N2={args.N2}, M={args.M}, "
        f"{args.num_replicas} replicas)",
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


def run_single_replica_with_history(
    alpha: float,
    device: torch.device,
    seed: int,
    N1: int,
    N2: int,
    M: int,
    max_steps: int,
    lr: float,
    noise_var: float,
    record_interval: int,
) -> tuple[float, dict[str, list[float]]]:
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)

    graph = RandomGraph()
    i_idx, j_idx, edge_count = graph.generate(N1, N2, M, alpha, device, seed)
    if edge_count == 0:
        history = {"steps": [0], "loss": [0.0], "mse": [0.0]}
        return 0.0, history

    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y) * math.sqrt(noise_var)
    Y_noisy = Y + noise

    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32)

    history_steps: list[int] = []
    history_losses: list[float] = []
    history_mses: list[float] = []
    history_cosine_similarities: list[float] = []

    def record(step: int, W_curr: torch.Tensor, X_curr: torch.Tensor) -> None:
        Y_pred = compute_predictions(W_curr, X_curr, i_idx, j_idx, M)
        residual = Y_noisy - Y_pred
        W_eval = normalize_to_unit_variance(W_curr)
        X_eval = normalize_to_unit_variance(X_curr)
        history_steps.append(step)
        history_losses.append(float(compute_loss(Y_noisy, Y_pred, M).item()))
        history_mses.append(float((residual ** 2).mean().item()))
        history_cosine_similarities.append(
            compute_y_cosine_similarity(W_eval, X_eval, W_teacher, X_teacher)
        )

    record(0, W_hat, X_hat)

    for step in range(1, max_steps + 1):
        W_hat = agd_step_W(W_hat, X_hat, Y_noisy, i_idx, j_idx, lr)
        X_hat = agd_step_X(W_hat, X_hat, Y_noisy, i_idx, j_idx, lr)

        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)

        if step % record_interval == 0 or step == max_steps:
            record(step, W_hat, X_hat)

    cosine_similarity = compute_y_cosine_similarity(
        W_hat, X_hat, W_teacher, X_teacher
    )
    history = {
        "steps": history_steps,
        "loss": history_losses,
        "mse": history_mses,
        "cosine_similarity": history_cosine_similarities,
    }
    return cosine_similarity, history


def main() -> None:
    args = parse_args()
    device = detect_device()
    lr = resolve_lr(args)

    print("=" * 60)
    print("Loss vs Step for Alternating Gradient Descent")
    print("Evaluation Metric: Cosine Similarity in Y-space")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}")
    print(
        f"max_steps={args.max_steps}, lr={lr:.6e}, "
        f"noise_var={args.noise_var:.6e}, record_interval={args.record_interval}"
    )
    print(f"replicas={args.num_replicas}, seed={args.seed}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_agd_cosine_loss_vs_step_alpha{args.alpha}_"
        f"{args.N1}x{args.N2}_M{args.M}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config(results_dir, args, device, lr)

    all_losses = []
    all_mses = []
    all_cosine_similarities = []
    seeds = []
    runtimes = []
    initial_losses = []
    final_losses = []
    final_mses = []
    cosine_similarity_values = []
    convergence_steps = []

    total_start = time.time()

    for replica_idx in range(args.num_replicas):
        seed = args.seed + replica_idx * 1000
        replica_start = time.time()

        cosine_similarity, history = run_single_replica_with_history(
            alpha=args.alpha,
            device=device,
            seed=seed,
            N1=args.N1,
            N2=args.N2,
            M=args.M,
            max_steps=args.max_steps,
            lr=lr,
            noise_var=args.noise_var,
            record_interval=args.record_interval,
        )

        runtime = time.time() - replica_start
        steps = np.asarray(history["steps"], dtype=np.int64)
        loss_history = np.asarray(history["loss"], dtype=np.float64)
        mse_history = np.asarray(history["mse"], dtype=np.float64)
        cosine_similarity_history = np.asarray(
            history["cosine_similarity"], dtype=np.float64
        )

        all_losses.append(loss_history)
        all_mses.append(mse_history)
        all_cosine_similarities.append(cosine_similarity_history)
        seeds.append(seed)
        runtimes.append(runtime)
        initial_losses.append(float(loss_history[0]))
        final_losses.append(float(loss_history[-1]))
        final_mses.append(float(mse_history[-1]))
        cosine_similarity_values.append(cosine_similarity)
        convergence_steps.append(
            estimate_convergence_step(steps, loss_history, args.convergence_threshold)
        )

        convergence_text = (
            "not reached"
            if math.isnan(convergence_steps[-1])
            else str(int(convergence_steps[-1]))
        )
        print(
            f"Replica {replica_idx + 1}/{args.num_replicas}: "
            f"seed={seed}, initial_loss={loss_history[0]:.2e}, "
            f"final_loss={loss_history[-1]:.2e}, "
            f"final_cosine_similarity={cosine_similarity:.4f}, "
            f"estimated_convergence_step={convergence_text}, "
            f"runtime={runtime:.1f}s"
        )

    total_runtime = time.time() - total_start

    all_losses_arr = np.asarray(all_losses, dtype=np.float64)
    all_mses_arr = np.asarray(all_mses, dtype=np.float64)
    all_cosine_similarities_arr = np.asarray(
        all_cosine_similarities, dtype=np.float64
    )
    mean_loss = all_losses_arr.mean(axis=0)
    std_loss = all_losses_arr.std(axis=0)
    mean_log_loss = np.log10(np.clip(all_losses_arr, 1e-30, None)).mean(axis=0)
    std_log_loss = np.log10(np.clip(all_losses_arr, 1e-30, None)).std(axis=0)
    mean_cosine_similarity = all_cosine_similarities_arr.mean(axis=0)
    std_cosine_similarity = all_cosine_similarities_arr.std(axis=0)

    save_loss_history(
        results_dir,
        steps,
        all_losses_arr,
        all_mses_arr,
        all_cosine_similarities_arr,
    )
    save_replica_summary(
        results_dir,
        seeds,
        runtimes,
        initial_losses,
        final_losses,
        final_mses,
        cosine_similarity_values,
        convergence_steps,
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
    print(f"Mean initial loss: {np.mean(initial_losses):.2e}")
    print(f"Mean final loss: {np.mean(final_losses):.2e}")
    print(f"Mean final MSE: {np.mean(final_mses):.2e}")
    print(
        f"Mean final cosine similarity: {np.mean(cosine_similarity_values):.4f}"
    )
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
