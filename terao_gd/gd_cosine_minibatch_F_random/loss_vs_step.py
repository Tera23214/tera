#!/usr/bin/env python
"""
Plot mini-batch SGD loss vs step for the F-random observation model.

Training model:
    Y_ij = lambda / sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j + noise

The student receives the full F and estimates W, X by alternating mini-batch
SGD.
"""

from __future__ import annotations

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

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gd.gd_cosine_minibatch_F_random.gd import (
    DEFAULT_COSINE_ROW_CHUNK_SIZE,
    DEFAULT_F_DTYPE,
    DEFAULT_F_STORAGE_DEVICE,
    DEFAULT_PREDICTION_CHUNK_SIZE,
    LAMBDA,
    compute_loss,
    compute_predictions,
    compute_y_cosine_similarity,
    normalize_to_unit_variance,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    sample_minibatch_positions,
    sgd_step_W,
    sgd_step_X,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mini-batch SGD loss vs step for F-random model."
    )
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=LAMBDA)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--lr-base",
        type=float,
        default=None,
        help="Base coefficient for auto LR scaling: lr = lr_base / sqrt(batch_size).",
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--noise-var", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    parser.add_argument("--record-interval", type=int, default=100)
    parser.add_argument(
        "--f-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default=DEFAULT_F_DTYPE,
        help="Storage dtype for the shared F tensor.",
    )
    parser.add_argument(
        "--f-storage-device",
        type=str,
        choices=["same", "cpu"],
        default=DEFAULT_F_STORAGE_DEVICE,
        help="Where to store the shared F tensor.",
    )
    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        default=DEFAULT_PREDICTION_CHUNK_SIZE,
        help="Chunk size for observed-edge prediction passes.",
    )
    parser.add_argument(
        "--cosine-row-chunk-size",
        type=int,
        default=DEFAULT_COSINE_ROW_CHUNK_SIZE,
        help="Row chunk size for full Y-space cosine similarity evaluation.",
    )
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
    if args.lr_base is None:
        raise ValueError("--lr-base must be provided when --lr is omitted.")
    if args.batch_size <= 0:
        return 0.0
    return float(args.lr_base / math.sqrt(args.batch_size))


def estimate_convergence_step(
    steps: np.ndarray,
    loss_history: np.ndarray,
    threshold: float,
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
    num_observed: int,
) -> None:
    config = {
        "algorithm": "sgd_cosine_minibatch_F_random_loss_vs_step",
        "alpha": args.alpha,
        "N": args.N,
        "M": args.M,
        "lambda": args.lambda_,
        "observation_model": (
            "Y_ij = lambda/sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j + noise"
        ),
        "F_shape": [args.N, args.N, args.M],
        "F_distribution": "rademacher_pm1",
        "effective_F_values": "+/- lambda / sqrt(M)",
        "max_steps": args.max_steps,
        "lr": lr,
        "lr_base": args.lr_base,
        "batch_size": args.batch_size,
        "num_observed": num_observed,
        "noise_var": args.noise_var,
        "teacher_seed": args.seed,
        "F_seed": args.seed + 500,
        "graph_seed": args.seed,
        "noise_seed": args.seed + 1000,
        "student_seed_base": 100,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "record_interval": args.record_interval,
        "device": str(device),
        "F_dtype": args.f_dtype,
        "F_storage_device": args.f_storage_device,
        "prediction_chunk_size": args.prediction_chunk_size,
        "cosine_row_chunk_size": args.cosine_row_chunk_size,
        "student_init": "standard_normal_scaled_0.01",
        "evaluation_metric": "cosine_similarity_in_full_Y_space",
        "sampling": "with_replacement",
        "shared_teacher_F_noise_global": True,
        "shared_per_alpha_graph_noise": True,
        "replica_variation": "student_initialization_only",
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
    mean_cosine_similarity = all_cosine_similarities.mean(axis=0)
    std_cosine_similarity = all_cosine_similarities.std(axis=0)

    header = [
        "step",
        "loss_mean",
        "loss_std",
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


def plot_loss(
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
        f"F-random Mini-batch SGD Loss vs Step "
        f"(alpha={args.alpha}, N={args.N}, M={args.M}, lambda={args.lambda_})",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_vs_step.png", dpi=150, bbox_inches="tight")
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
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"F-random Cosine Similarity vs Step "
        f"(alpha={args.alpha}, N={args.N}, M={args.M}, lambda={args.lambda_})",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_similarity_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single_replica_with_history(
    shared_data: dict[str, torch.Tensor | float | int],
    device: torch.device,
    seed: int,
    N: int,
    M: int,
    max_steps: int,
    lr: float,
    batch_size: int,
    lambda_: float,
    record_interval: int,
    prediction_chunk_size: int,
    cosine_row_chunk_size: int,
) -> tuple[float, dict[str, list[float]]]:
    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    y_train = shared_data["Y_train"]
    w_teacher = shared_data["W_teacher"]
    x_teacher = shared_data["X_teacher"]
    F = shared_data["F"]
    num_observed = int(shared_data["num_observed"])

    torch.manual_seed(seed + 2000)
    w_hat = torch.randn(N, M, device=device, dtype=torch.float32)
    x_hat = torch.randn(M, N, device=device, dtype=torch.float32)

    history_steps: list[int] = []
    history_losses: list[float] = []
    history_mses: list[float] = []
    history_cosine_similarities: list[float] = []

    def record(step: int) -> None:
        y_pred = compute_predictions(
            w_hat,
            x_hat,
            F,
            i_idx,
            j_idx,
            M,
            lambda_=lambda_,
            chunk_size=prediction_chunk_size,
        )
        residual = y_train - y_pred
        history_steps.append(step)
        history_losses.append(float(compute_loss(y_train, y_pred, M).item()))
        history_mses.append(float((residual**2).mean().item()))
        history_cosine_similarities.append(
            compute_y_cosine_similarity(
                w_hat,
                x_hat,
                w_teacher,
                x_teacher,
                F,
                row_chunk_size=cosine_row_chunk_size,
            )
        )

    record(0)

    for step in range(1, max_steps + 1):
        batch_positions = sample_minibatch_positions(num_observed, batch_size, device)
        w_hat = sgd_step_W(
            w_hat,
            x_hat,
            F,
            y_train,
            i_idx,
            j_idx,
            lr,
            batch_positions,
            lambda_=lambda_,
        )
        x_hat = sgd_step_X(
            w_hat,
            x_hat,
            F,
            y_train,
            i_idx,
            j_idx,
            lr,
            batch_positions,
            lambda_=lambda_,
        )
        w_hat = normalize_to_unit_variance(w_hat)
        x_hat = normalize_to_unit_variance(x_hat)

        if step % record_interval == 0 or step == max_steps:
            record(step)

    cosine_similarity = compute_y_cosine_similarity(
        w_hat,
        x_hat,
        w_teacher,
        x_teacher,
        F,
        row_chunk_size=cosine_row_chunk_size,
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
    print("Loss vs Step for F-random Mini-batch SGD")
    print("Observation: Y_ij = lambda/sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j + noise")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"alpha={args.alpha}, N={args.N}, M={args.M}, lambda={args.lambda_}")
    print(
        f"max_steps={args.max_steps}, batch_size={args.batch_size}, "
        f"lr={lr:.6e}, noise_var={args.noise_var:.6e}, "
        f"record_interval={args.record_interval}"
    )
    print(f"replicas={args.num_replicas}, seed={args.seed}")
    print(
        f"F storage={args.f_storage_device}, F dtype={args.f_dtype}, "
        f"prediction_chunk_size={args.prediction_chunk_size}, "
        f"cosine_row_chunk_size={args.cosine_row_chunk_size}"
    )
    print()

    global_data = prepare_global_shared_data(
        device=device,
        seed=args.seed,
        N=args.N,
        M=args.M,
        noise_var=args.noise_var,
        f_storage_device=args.f_storage_device,
        f_dtype=args.f_dtype,
    )
    shared_data = prepare_shared_alpha_data(
        alpha=args.alpha,
        device=device,
        seed=args.seed,
        N=args.N,
        M=args.M,
        noise_var=args.noise_var,
        lambda_=args.lambda_,
        prediction_chunk_size=args.prediction_chunk_size,
        global_data=global_data,
    )
    num_observed = int(shared_data["num_observed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_sgd_F_random_loss_vs_step_alpha{args.alpha}_"
        f"N{args.N}_M{args.M}_lambda{args.lambda_}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_config(results_dir, args, device, lr, num_observed)

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
            shared_data=shared_data,
            device=device,
            seed=seed,
            N=args.N,
            M=args.M,
            max_steps=args.max_steps,
            lr=lr,
            batch_size=args.batch_size,
            lambda_=args.lambda_,
            record_interval=args.record_interval,
            prediction_chunk_size=args.prediction_chunk_size,
            cosine_row_chunk_size=args.cosine_row_chunk_size,
        )

        runtime = time.time() - replica_start
        steps = np.asarray(history["steps"], dtype=np.int64)
        loss_history = np.asarray(history["loss"], dtype=np.float64)
        mse_history = np.asarray(history["mse"], dtype=np.float64)
        cosine_similarity_history = np.asarray(
            history["cosine_similarity"],
            dtype=np.float64,
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
    all_cosine_similarities_arr = np.asarray(all_cosine_similarities, dtype=np.float64)
    mean_loss = all_losses_arr.mean(axis=0)
    std_loss = all_losses_arr.std(axis=0)
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
    plot_loss(plots_dir, steps, all_losses_arr, mean_loss, std_loss, args)
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
    print(f"Mean final cosine similarity: {np.mean(cosine_similarity_values):.4f}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
