#!/usr/bin/env python
"""
Alternating mini-batch SGD for sparse matrix factorization with
cosine-similarity evaluation.

This variant is based on ``terao_gd/gd_cosine`` but replaces full-batch
alternating gradient descent with observed-edge mini-batches sampled
with replacement. Graph / teacher / noisy observations are generated once per
alpha and reused across replicas. Replica-to-replica variation comes only from
student initialization.
"""

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to path(to get shared modules)
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000
N2 = 1000
M = 100

ALPHA_START =0
ALPHA_STOP = 5
ALPHA_STEP = 0.2

MAX_STEPS = 6000
BATCH_SIZE = 5000
LR_SCHEDULE = [
    (0.0, 0.5, 1e-3),
    (0.5, 1.5, 2e-3),
    (1.5, 3.0, 3e-3),
    (3.0, float("inf"), 3e-3),
]
NOISE_VAR = 0
SHARED_SEED = 1
STUDENT_SEED_BASE = 100
NUM_REPLICAS = 10
CONVERGENCE_THRESHOLD = 1e-5
LOSS_EVAL_INTERVAL = 50


# ============================================================================
# SGD Helper Functions
# ============================================================================


def compute_predictions(
    W: torch.Tensor,
    X: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """
    Compute predictions Y_pred for observed entries.

    Y_pred[c] = (1/sqrt(M)) * sum_mu W[i_c, mu] * X[mu, j_c]
    """
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1) / math.sqrt(M)


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """
    Compute the average loss per observed edge.

    The legacy implementation used ``M * sum((Y - Y_pred)^2)``. We now divide
    by the number of observed edges so the reported value is a per-edge loss.
    """
    num_observed = max(Y.numel(), 1)
    return M * ((Y - Y_pred) ** 2).sum() / num_observed


def sample_minibatch_positions(
    num_observed: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample observed-entry positions with replacement.
    """
    if num_observed <= 0 or batch_size <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.randint(
        low=0,
        high=num_observed,
        size=(batch_size,),
        device=device,
    )


def resolve_lr(
    alpha: float,
    lr_schedule: list[tuple[float, float, float]] = LR_SCHEDULE,
) -> float:
    """
    Resolve the learning rate from the configured alpha ranges.
    """
    for alpha_min, alpha_max, lr_value in lr_schedule:
        if alpha_min <= alpha < alpha_max:
            return float(lr_value)
    return float(lr_schedule[-1][2])


def compute_effective_epochs(
    num_observed: int,
    batch_size: int,
    max_steps: int,
) -> float:
    """
    Convert a fixed update budget into expected epochs.

    With replacement sampling, one epoch corresponds to sampling
    ``num_observed`` entries on average, so
    ``effective_epochs = max_steps * batch_size / num_observed``.
    """
    if num_observed <= 0 or batch_size <= 0:
        return 0.0
    return float(max_steps * batch_size / num_observed)


def sgd_step_W(
    W: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
    batch_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Mini-batch gradient step for W.
    """
    _, M = W.shape
    batch_i = i_idx[batch_positions].long()
    batch_j = j_idx[batch_positions].long()
    batch_y = Y[batch_positions]

    Y_pred = compute_predictions(W, X, batch_i, batch_j, M)
    residual = Y_pred - batch_y
    X_sel = X[:, batch_j].T

    grad_contrib = (
        2.0
        * math.sqrt(M)
        * residual.unsqueeze(1)
        * X_sel
    )

    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, batch_i.unsqueeze(1).expand(-1, M), grad_contrib)
    return W - lr * grad_W


def sgd_step_X(
    W: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
    batch_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Mini-batch gradient step for X.
    """
    M, _ = X.shape
    batch_i = i_idx[batch_positions].long()
    batch_j = j_idx[batch_positions].long()
    batch_y = Y[batch_positions]

    Y_pred = compute_predictions(W, X, batch_i, batch_j, M)
    residual = Y_pred - batch_y
    W_sel = W[batch_i, :]

    grad_contrib = (
        2.0
        * math.sqrt(M)
        * residual.unsqueeze(1)
        * W_sel
    )

    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, batch_j.unsqueeze(0).expand(M, -1), grad_contrib.T)
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
    """
    Compute cosine similarity between Y_teacher = W_teacher X_teacher and
    Y_student = W_student X_student without materializing dense Y matrices.
    """
    cross_w = W_teacher.T @ W_student
    cross_x = X_student @ X_teacher.T
    inner = torch.trace(cross_w @ cross_x)

    teacher_norm_sq = torch.trace((W_teacher.T @ W_teacher) @ (X_teacher @ X_teacher.T))
    student_norm_sq = torch.trace((W_student.T @ W_student) @ (X_student @ X_student.T))
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))

    return (inner / denom).item()


def prepare_global_shared_data(
    device: torch.device,
    seed: int = 1,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 200,
    noise_var: float = 0.0,
) -> dict[str, torch.Tensor | float | int]:
    """
    Prepare teacher matrices and a full-grid noise field once for the whole
    simulation.
    """
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)

    torch.manual_seed(seed)
    noise_full = torch.randn((N1, N2), device=device, dtype=torch.float32)
    noise_full = noise_full * math.sqrt(noise_var)

    return {
        "seed": seed,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "noise_full": noise_full,
    }


def prepare_shared_alpha_data(
    alpha: float,
    device: torch.device,
    seed: int = 1,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 200,
    noise_var: float = 0.0,
    global_data: dict[str, torch.Tensor | float | int] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """
    Prepare graph and observed values once for a single alpha.
    """
    if global_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=seed,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
        )

    graph = RandomGraph()
    i_idx, j_idx, num_observed = graph.generate(N1, N2, M, alpha, device, seed)
    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    noise_full = global_data["noise_full"]

    if num_observed == 0:
        return {
            "alpha": alpha,
            "num_observed": 0,
            "i_idx": i_idx,
            "j_idx": j_idx,
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "Y_clean": torch.empty(0, dtype=torch.float32, device=device),
            "Y_train": torch.empty(0, dtype=torch.float32, device=device),
        }

    Y_clean = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)
    Y_train = Y_clean + noise_full[i_idx.long(), j_idx.long()]

    return {
        "alpha": alpha,
        "num_observed": num_observed,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "Y_clean": Y_clean,
        "Y_train": Y_train,
    }


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
    N1: int = N1,
    N2: int = N2,
    M: int = M,
    max_steps: int = MAX_STEPS,
    lr: float | None = None,
    batch_size: int = BATCH_SIZE,
    noise_var: float = NOISE_VAR,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    loss_eval_interval: int = LOSS_EVAL_INTERVAL,
    shared_data: dict[str, torch.Tensor | float | int] | None = None,
) -> tuple[float, float, int]:
    """
    Train a single replica for a given alpha using alternating mini-batch SGD.
    """
    if shared_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=1,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
        )
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=1,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
            global_data=global_data,
        )

    num_observed = int(shared_data["num_observed"])
    if num_observed == 0:
        return 0.0, 0.0, 0

    if lr is None:
        lr = resolve_lr(alpha)

    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    Y_train = shared_data["Y_train"]
    W_teacher = shared_data["W_teacher"]
    X_teacher = shared_data["X_teacher"]
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]

    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01

    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float("inf")

    for step in range(max_steps):
        batch_positions = sample_minibatch_positions(
            num_observed=num_observed,
            batch_size=batch_size,
            device=device,
        )

        W_hat = sgd_step_W(
            W_hat,
            X_hat,
            Y_train,
            i_idx,
            j_idx,
            lr,
            batch_positions,
        )
        X_hat = sgd_step_X(
            W_hat,
            X_hat,
            Y_train,
            i_idx,
            j_idx,
            lr,
            batch_positions,
        )

        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)

        if step % loss_eval_interval == 0 or step == max_steps - 1:
            Y_pred_full = compute_predictions(W_hat, X_hat, i_idx, j_idx, M)
            loss = float(compute_loss(Y_train, Y_pred_full, M).item())
            final_loss = loss

            if abs(prev_loss - loss) < convergence_threshold:
                steps_taken = step + 1
                break
            prev_loss = loss

    cosine_similarity = compute_y_cosine_similarity(
        W_hat, X_hat, W_teacher, X_teacher
    )
    return cosine_similarity, final_loss, steps_taken


def save_metrics_csv(
    results_dir: Path,
    results: dict[float, dict[str, float | list[float]]],
    alphas_list: list[float],
    num_replicas: int,
) -> None:
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        header = (
            "alpha,lr,num_observed,effective_epochs,"
            "cosine_similarity_mean,cosine_similarity_std,"
            "Loss_mean,Loss_std,Steps_mean"
        )
        for i in range(num_replicas):
            header += f",cosine_similarity_replica_{i},loss_replica_{i}"
        f.write(header + "\n")

        for alpha in alphas_list:
            r = results[alpha]
            line = (
                f"{alpha},{r['lr']},{r['num_observed']},{r['effective_epochs']},"
                f"{r['cosine_similarity_mean']},"
                f"{r['cosine_similarity_std']},{r['loss_mean']},"
                f"{r['loss_std']},{r['steps_mean']}"
            )
            for cosine_similarity_value, loss_v in zip(
                r["cosine_similarity_values"], r["loss_values"]
            ):
                line += f",{cosine_similarity_value},{loss_v}"
            f.write(line + "\n")


def save_replica_summary(
    results_dir: Path,
    replica_records: list[dict[str, float | int]],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "alpha,lr,num_observed,effective_epochs,replica,seed,runtime_sec,"
            "final_loss,steps_taken,cosine_similarity\n"
        )
        for record in replica_records:
            f.write(
                f"{record['alpha']},{record['lr']},{record['num_observed']},"
                f"{record['effective_epochs']},{record['replica']},{record['seed']},"
                f"{record['runtime_sec']:.4f},{record['final_loss']:.10e},"
                f"{record['steps_taken']},{record['cosine_similarity']:.10e}\n"
            )


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Alternating Mini-Batch SGD - Matrix Factorization")
    print("Cosine Similarity Evaluation")
    print("=" * 60)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        print("Using: CPU")

    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Steps: {MAX_STEPS}, Batch={BATCH_SIZE}")
    print(f"LR schedule: {LR_SCHEDULE}")
    print("Effective epoch formula: max_steps * batch_size / num_observed")
    print(
        "Early stopping: "
        f"abs(loss_t - loss_(t-1)) < {CONVERGENCE_THRESHOLD} "
        f"(checked every {LOSS_EVAL_INTERVAL} step)"
    )
    print("Sampling: with replacement from observed edges")
    print("Teacher / graph / noise seed: 1")
    print("Student seed rule: 100 + replica_id")
    print("Shared per alpha: graph / observed targets")
    print("Shared across all alphas: teacher / full noise field")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_sgd_cosine_{N1}x{M}_alpha{ALPHA_START}-{ALPHA_STOP}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    config = {
        "algorithm": "sgd_cosine_minibatch",
        "N1": N1,
        "N2": N2,
        "M": M,
        "alpha_start": ALPHA_START,
        "alpha_stop": ALPHA_STOP,
        "alpha_step": ALPHA_STEP,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "lr_schedule": [
            {"alpha_min": alpha_min, "alpha_max": alpha_max, "lr": lr_value}
            for alpha_min, alpha_max, lr_value in LR_SCHEDULE
        ],
        "effective_epoch_formula": "max_steps * batch_size / num_observed",
        "noise_var": NOISE_VAR,
        "teacher_seed": SHARED_SEED,
        "graph_seed": SHARED_SEED,
        "noise_seed": SHARED_SEED,
        "student_seed_base": STUDENT_SEED_BASE,
        "num_replicas": NUM_REPLICAS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "loss_eval_interval": LOSS_EVAL_INTERVAL,
        "early_stop_metric": "abs_delta_loss_per_edge",
        "device": str(device),
        "evaluation_metric": "cosine_similarity_in_Y_space",
        "sampling": "with_replacement",
        "shared_per_alpha_graph_noise": True,
        "shared_teacher_noise_global": True,
        "replica_variation": "student_initialization_only",
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")

    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP / 2, ALPHA_STEP)
    results = {}

    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0
    replica_records: list[dict[str, float | int]] = []
    global_data = prepare_global_shared_data(
        device=device,
        seed=SHARED_SEED,
        N1=N1,
        N2=N2,
        M=M,
        noise_var=NOISE_VAR,
    )

    for alpha in alphas:
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=SHARED_SEED,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=NOISE_VAR,
            global_data=global_data,
        )
        num_observed = int(shared_data["num_observed"])
        lr_alpha = resolve_lr(float(alpha))
        effective_epochs = compute_effective_epochs(
            num_observed=num_observed,
            batch_size=BATCH_SIZE,
            max_steps=MAX_STEPS,
        )
        print(
            f"α={alpha:.2f}: E={num_observed}, lr={lr_alpha:.3e}, "
            f"effective_epochs={effective_epochs:.2f}"
        )

        cosine_similarity_values = []
        loss_values = []
        steps_values = []

        for replica_id in range(NUM_REPLICAS):
            replica_seed = STUDENT_SEED_BASE + replica_id
            t0 = time.time()
            cosine_similarity, final_loss, steps_taken = train_single_replica(
                alpha=alpha,
                device=device,
                seed=replica_seed,
                N1=N1,
                N2=N2,
                M=M,
                max_steps=MAX_STEPS,
                lr=lr_alpha,
                batch_size=BATCH_SIZE,
                noise_var=NOISE_VAR,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                shared_data=shared_data,
            )
            dt = time.time() - t0
            cosine_similarity_values.append(cosine_similarity)
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            completed += 1
            replica_records.append(
                {
                    "alpha": float(alpha),
                    "lr": lr_alpha,
                    "num_observed": num_observed,
                    "effective_epochs": effective_epochs,
                    "replica": replica_id + 1,
                    "seed": replica_seed,
                    "runtime_sec": dt,
                    "final_loss": final_loss,
                    "steps_taken": steps_taken,
                    "cosine_similarity": cosine_similarity,
                }
            )
            print(
                f"α={alpha:.2f}, replica {replica_id + 1}/{NUM_REPLICAS}: "
                f"CosSim={cosine_similarity:.4f}, Loss={final_loss:.2e}, "
                f"Steps={steps_taken} ({dt:.1f}s) [{completed}/{total_tasks}]"
            )

        results[alpha] = {
            "lr": lr_alpha,
            "num_observed": num_observed,
            "effective_epochs": effective_epochs,
            "cosine_similarity_mean": np.mean(cosine_similarity_values),
            "cosine_similarity_std": np.std(cosine_similarity_values),
            "cosine_similarity_values": cosine_similarity_values,
            "loss_mean": np.mean(loss_values),
            "loss_std": np.std(loss_values),
            "loss_values": loss_values,
            "steps_mean": np.mean(steps_values),
        }

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Results (mean ± std)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'CosSim':^20} | {'Loss':^20} | {'Steps':>8}")
    print("-" * 60)
    for alpha in sorted(results.keys()):
        r = results[alpha]
        print(
            f"{alpha:6.2f} | "
            f"{r['cosine_similarity_mean']:8.4f} ± {r['cosine_similarity_std']:<8.4f} | "
            f"{r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | "
            f"{r['steps_mean']:8.0f}"
        )

    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 60)

    print("\nGenerating plots...")
    alphas_list = sorted(results.keys())
    cosine_similarity_means = [
        results[a]["cosine_similarity_mean"] for a in alphas_list
    ]
    cosine_similarity_stds = [
        results[a]["cosine_similarity_std"] for a in alphas_list
    ]
    cosine_similarity_sems = [
        std / math.sqrt(NUM_REPLICAS) for std in cosine_similarity_stds
    ]

    save_metrics_csv(results_dir, results, alphas_list, NUM_REPLICAS)
    save_replica_summary(results_dir, replica_records)
    print(f"Metrics saved: {results_dir / 'metrics.csv'}")
    print(f"Replica summary saved: {results_dir / 'replica_summary.csv'}")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        alphas_list,
        cosine_similarity_means,
        yerr=cosine_similarity_sems,
        fmt="o-",
        color="#1E88E5",
        markersize=6,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
    )
    ax.set_xlabel(r"$\alpha$ (observation density)", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title(
        f"Phase Transition (Alternating Mini-Batch SGD)\n"
        f"({N1}×{N2}, M={M}, {MAX_STEPS} steps, {NUM_REPLICAS} replicas)",
        fontsize=16,
    )
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "cosine_similarity_vs_alpha.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.show()
    print(f"\nResults saved to: {results_dir}")
    print("Done!")
