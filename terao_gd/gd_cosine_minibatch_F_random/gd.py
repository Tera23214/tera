#!/usr/bin/env python
"""
Alternating mini-batch SGD for the F-random observation model.

Observation model:
    Y_ij = lambda / sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j + noise

where
    F: (N, N, M), F_ij,mu ~ N(0, 1)
    W: (N, M)
    X: (M, N)

F is generated once, shared with the student perfectly, and never optimized.
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph


N = 1000
M = 100
LAMBDA = 1.0

ALPHA_START = 0
ALPHA_STOP = 5.0
ALPHA_STEP = 0.2

MAX_STEPS = 500000
BATCH_SIZE = 2000
LR_SCHEDULE = [
    (0.0, 0.5, 1e-3),
    (0.5, 1.5, 1.5e-3),
    (1.5, 3.0, 1.5e-3),
    (3.0, float("inf"), 1.5e-3),
]
NOISE_VAR = 0
SHARED_SEED = 1
STUDENT_SEED_BASE = 100
NUM_REPLICAS = 5
CONVERGENCE_THRESHOLD = 1e-5
LOSS_EVAL_INTERVAL = 100
SAVE_EVERY_REPLICAS = 5


def observation_scale(M: int, lambda_: float) -> float:
    return float(lambda_ / math.sqrt(M))


def compute_full_predictions(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    M: int,
    lambda_: float = LAMBDA,
) -> torch.Tensor:
    """
    Compute predictions for every (i, j):
        Y_ij = lambda/sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j.
    """
    return observation_scale(M, lambda_) * torch.einsum("ijm,im,mj->ij", F, W, X)


def compute_predictions(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
    lambda_: float = LAMBDA,
) -> torch.Tensor:
    """
    Compute observed predictions from lambda/sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j.
    """
    i_long = i_idx.long()
    j_long = j_idx.long()
    f_sel = F[i_long, j_long, :]
    w_sel = W[i_long, :]
    x_sel = X[:, j_long].T
    return observation_scale(M, lambda_) * (f_sel * w_sel * x_sel).sum(dim=1)


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    num_observed = max(Y.numel(), 1)
    return M * ((Y - Y_pred) ** 2).sum() / num_observed


def sample_minibatch_positions(
    num_observed: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if num_observed <= 0 or batch_size <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.randint(0, num_observed, (batch_size,), device=device)


def resolve_lr(
    alpha: float,
    lr_schedule: list[tuple[float, float, float]] = LR_SCHEDULE,
) -> float:
    for alpha_min, alpha_max, lr_value in lr_schedule:
        if alpha_min <= alpha < alpha_max:
            return float(lr_value)
    return float(lr_schedule[-1][2])


def compute_effective_epochs(
    num_observed: int,
    batch_size: int,
    max_steps: int,
) -> float:
    if num_observed <= 0 or batch_size <= 0:
        return 0.0
    return float(max_steps * batch_size / num_observed)


def sgd_step_W(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
    batch_positions: torch.Tensor,
    lambda_: float = LAMBDA,
) -> torch.Tensor:
    """
    Mini-batch gradient step for W.
    """
    _, M = W.shape
    batch_i = i_idx[batch_positions].long()
    batch_j = j_idx[batch_positions].long()
    batch_y = Y[batch_positions]

    f_sel = F[batch_i, batch_j, :]
    w_sel = W[batch_i, :]
    x_sel = X[:, batch_j].T
    y_pred = observation_scale(M, lambda_) * (f_sel * w_sel * x_sel).sum(dim=1)
    residual = y_pred - batch_y

    coeff = 2.0 * M * observation_scale(M, lambda_)
    grad_contrib = coeff * residual.unsqueeze(1) * f_sel * x_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, batch_i.unsqueeze(1).expand(-1, M), grad_contrib)
    return W - lr * grad_W


def sgd_step_X(
    W: torch.Tensor,
    X: torch.Tensor,
    F: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
    batch_positions: torch.Tensor,
    lambda_: float = LAMBDA,
) -> torch.Tensor:
    """
    Mini-batch gradient step for X.
    """
    M, _ = X.shape
    batch_i = i_idx[batch_positions].long()
    batch_j = j_idx[batch_positions].long()
    batch_y = Y[batch_positions]

    f_sel = F[batch_i, batch_j, :]
    w_sel = W[batch_i, :]
    x_sel = X[:, batch_j].T
    y_pred = observation_scale(M, lambda_) * (f_sel * w_sel * x_sel).sum(dim=1)
    residual = y_pred - batch_y

    coeff = 2.0 * M * observation_scale(M, lambda_)
    grad_contrib = coeff * residual.unsqueeze(1) * f_sel * w_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, batch_j.unsqueeze(0).expand(M, -1), grad_contrib.T)
    return X - lr * grad_X


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    mean_sq = (tensor**2).mean()
    if mean_sq > 0:
        return tensor / torch.sqrt(mean_sq)
    return tensor


def compute_y_cosine_similarity(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    F: torch.Tensor,
) -> float:
    """
    Cosine similarity over the full N x N Y-space.
    """
    M = W_teacher.shape[1]
    y_teacher = compute_full_predictions(W_teacher, X_teacher, F, M)
    y_student = compute_full_predictions(W_student, X_student, F, M)
    inner = (y_teacher * y_student).sum()
    teacher_norm_sq = (y_teacher**2).sum()
    student_norm_sq = (y_student**2).sum()
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))
    return float((inner / denom).item())


def prepare_global_shared_data(
    device: torch.device,
    seed: int = SHARED_SEED,
    N: int = N,
    M: int = M,
    noise_var: float = NOISE_VAR,
) -> dict[str, torch.Tensor | float | int]:
    torch.manual_seed(seed)
    W_teacher = torch.randn(N, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N, device=device, dtype=torch.float32)

    torch.manual_seed(seed + 500)
    F = torch.randn(N, N, M, device=device, dtype=torch.float32)

    torch.manual_seed(seed + 1000)
    noise_full = torch.randn((N, N), device=device, dtype=torch.float32)
    noise_full = noise_full * math.sqrt(noise_var)

    return {
        "seed": seed,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "F": F,
        "noise_full": noise_full,
    }


def prepare_shared_alpha_data(
    alpha: float,
    device: torch.device,
    seed: int = SHARED_SEED,
    N: int = N,
    M: int = M,
    noise_var: float = NOISE_VAR,
    lambda_: float = LAMBDA,
    global_data: dict[str, torch.Tensor | float | int] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    if global_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=seed,
            N=N,
            M=M,
            noise_var=noise_var,
        )

    graph = RandomGraph()
    i_idx, j_idx, num_observed = graph.generate(N, N, M, alpha, device, seed)
    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    F = global_data["F"]
    noise_full = global_data["noise_full"]

    if num_observed == 0:
        return {
            "alpha": alpha,
            "num_observed": 0,
            "i_idx": i_idx,
            "j_idx": j_idx,
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "F": F,
            "Y_clean": torch.empty(0, dtype=torch.float32, device=device),
            "Y_train": torch.empty(0, dtype=torch.float32, device=device),
        }

    Y_clean = compute_predictions(
        W_teacher,
        X_teacher,
        F,
        i_idx,
        j_idx,
        M,
        lambda_=lambda_,
    )
    Y_train = Y_clean + noise_full[i_idx.long(), j_idx.long()]

    return {
        "alpha": alpha,
        "num_observed": num_observed,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "F": F,
        "Y_clean": Y_clean,
        "Y_train": Y_train,
    }


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
    N: int = N,
    M: int = M,
    max_steps: int = MAX_STEPS,
    lr: float | None = None,
    batch_size: int = BATCH_SIZE,
    noise_var: float = NOISE_VAR,
    lambda_: float = LAMBDA,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    loss_eval_interval: int = LOSS_EVAL_INTERVAL,
    shared_data: dict[str, torch.Tensor | float | int] | None = None,
) -> tuple[float, float, int]:
    if shared_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=SHARED_SEED,
            N=N,
            M=M,
            noise_var=noise_var,
        )
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=SHARED_SEED,
            N=N,
            M=M,
            noise_var=noise_var,
            lambda_=lambda_,
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
    F = shared_data["F"]
    N, M = W_teacher.shape

    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N, device=device, dtype=torch.float32) * 0.01

    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float("inf")

    for step in range(max_steps):
        batch_positions = sample_minibatch_positions(num_observed, batch_size, device)

        W_hat = sgd_step_W(
            W_hat,
            X_hat,
            F,
            Y_train,
            i_idx,
            j_idx,
            lr,
            batch_positions,
            lambda_=lambda_,
        )
        X_hat = sgd_step_X(
            W_hat,
            X_hat,
            F,
            Y_train,
            i_idx,
            j_idx,
            lr,
            batch_positions,
            lambda_=lambda_,
        )

        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)

        if step % loss_eval_interval == 0 or step == max_steps - 1:
            Y_pred_full = compute_predictions(
                W_hat,
                X_hat,
                F,
                i_idx,
                j_idx,
                M,
                lambda_=lambda_,
            )
            loss = float(compute_loss(Y_train, Y_pred_full, M).item())
            final_loss = loss

            if abs(prev_loss - loss) < convergence_threshold:
                steps_taken = step + 1
                break
            prev_loss = loss

    cosine_similarity = compute_y_cosine_similarity(
        W_hat,
        X_hat,
        W_teacher,
        X_teacher,
        F,
    )
    return cosine_similarity, final_loss, steps_taken


def save_metrics_csv(
    results_dir: Path,
    results: dict[float, dict[str, float | int | list[float] | list[int]]],
    alphas_list: list[float],
    num_replicas: int,
) -> None:
    csv_path = results_dir / "metrics.csv"
    lines = []
    header = (
        "alpha,lambda,lr,num_observed,effective_epochs,completed_replicas,"
        "cosine_similarity_mean,cosine_similarity_std,"
        "Loss_mean,Loss_std,Steps_mean"
    )
    for i in range(num_replicas):
        header += f",cosine_similarity_replica_{i},loss_replica_{i}"
    lines.append(header)

    for alpha in alphas_list:
        r = results[alpha]
        cosine_similarity_values = list(r["cosine_similarity_values"])
        loss_values = list(r["loss_values"])
        completed_replicas = int(r.get("completed_replicas", len(cosine_similarity_values)))
        line = (
            f"{alpha},{r['lambda']},{r['lr']},{r['num_observed']},"
            f"{r['effective_epochs']},{completed_replicas},"
            f"{r['cosine_similarity_mean']},{r['cosine_similarity_std']},"
            f"{r['loss_mean']},{r['loss_std']},{r['steps_mean']}"
        )
        for replica_idx in range(num_replicas):
            if replica_idx < len(cosine_similarity_values):
                line += f",{cosine_similarity_values[replica_idx]},{loss_values[replica_idx]}"
            else:
                line += ",,"
        lines.append(line)

    write_text_atomic(csv_path, "\n".join(lines) + "\n")


def save_replica_summary(
    results_dir: Path,
    replica_records: list[dict[str, float | int]],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    lines = [
        "alpha,lambda,lr,num_observed,effective_epochs,replica,seed,runtime_sec,"
        "final_loss,steps_taken,cosine_similarity"
    ]
    for record in replica_records:
        lines.append(
            f"{record['alpha']},{record['lambda']},{record['lr']},"
            f"{record['num_observed']},{record['effective_epochs']},"
            f"{record['replica']},{record['seed']},{record['runtime_sec']:.4f},"
            f"{record['final_loss']:.10e},{record['steps_taken']},"
            f"{record['cosine_similarity']:.10e}"
        )
    write_text_atomic(summary_path, "\n".join(lines) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def build_alpha_result(
    lambda_: float,
    lr_alpha: float,
    num_observed: int,
    effective_epochs: float,
    cosine_similarity_values: list[float],
    loss_values: list[float],
    steps_values: list[int],
) -> dict[str, float | int | list[float] | list[int]]:
    return {
        "lambda": lambda_,
        "lr": lr_alpha,
        "num_observed": num_observed,
        "effective_epochs": effective_epochs,
        "completed_replicas": len(cosine_similarity_values),
        "cosine_similarity_mean": float(np.mean(cosine_similarity_values)),
        "cosine_similarity_std": float(np.std(cosine_similarity_values)),
        "cosine_similarity_values": cosine_similarity_values.copy(),
        "loss_mean": float(np.mean(loss_values)),
        "loss_std": float(np.std(loss_values)),
        "loss_values": loss_values.copy(),
        "steps_mean": float(np.mean(steps_values)),
        "steps_values": steps_values.copy(),
    }


def save_progress_outputs(
    results_dir: Path,
    results: dict[float, dict[str, float | int | list[float] | list[int]]],
    replica_records: list[dict[str, float | int]],
    num_replicas: int,
    completed: int,
    total_tasks: int,
    start_time: float,
    status: str,
) -> None:
    alphas_list = sorted(results.keys())
    save_metrics_csv(results_dir, results, alphas_list, num_replicas)
    save_replica_summary(results_dir, replica_records)
    status_path = results_dir / "progress.yaml"
    status_data = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_tasks": completed,
        "total_tasks": total_tasks,
        "elapsed_sec": time.time() - start_time,
    }
    write_text_atomic(status_path, yaml.dump(status_data, default_flow_style=False))


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    print("=" * 60)
    print("Alternating Mini-Batch SGD - F-random Matrix Factorization")
    print("Observation: Y_ij = lambda/sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j + noise")
    print("=" * 60)

    device = detect_device()
    if device.type == "mps":
        print("Using: Apple Silicon (MPS)")
    elif device.type == "cuda":
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    else:
        print("Using: CPU")

    print(f"Matrix: N={N}, M={M}")
    print(f"F: {N}×{N}×{M}, lambda={LAMBDA}")
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
    print("Shared across run: teacher / F / full noise field")
    print("Shared per alpha: graph / observed targets")
    print("Replica-specific: student initialization only")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_sgd_F_random_{N}x{M}_lambda{LAMBDA}_"
        f"alpha{ALPHA_START}-{ALPHA_STOP}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    config = {
        "algorithm": "sgd_cosine_minibatch_F_random",
        "N": N,
        "M": M,
        "lambda": LAMBDA,
        "observation_model": (
            "Y_ij = lambda/sqrt(M) * sum_mu F_ij,mu W_i,mu X_mu,j + noise"
        ),
        "F_shape": [N, N, M],
        "F_distribution": "standard_normal",
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
        "F_seed": SHARED_SEED + 500,
        "graph_seed": SHARED_SEED,
        "noise_seed": SHARED_SEED + 1000,
        "student_seed_base": STUDENT_SEED_BASE,
        "num_replicas": NUM_REPLICAS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "loss_eval_interval": LOSS_EVAL_INTERVAL,
        "save_every_replicas": SAVE_EVERY_REPLICAS,
        "early_stop_metric": "abs_delta_loss_per_edge",
        "device": str(device),
        "evaluation_metric": "cosine_similarity_in_full_Y_space",
        "sampling": "with_replacement",
        "shared_teacher_F_noise_global": True,
        "shared_per_alpha_graph_noise": True,
        "replica_variation": "student_initialization_only",
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")

    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP / 2, ALPHA_STEP)
    results: dict[float, dict[str, float | int | list[float] | list[int]]] = {}

    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0
    replica_records: list[dict[str, float | int]] = []
    global_data = prepare_global_shared_data(
        device=device,
        seed=SHARED_SEED,
        N=N,
        M=M,
        noise_var=NOISE_VAR,
    )

    interrupted = False
    try:
        for alpha in alphas:
            alpha_key = float(alpha)
            shared_data = prepare_shared_alpha_data(
                alpha=alpha_key,
                device=device,
                seed=SHARED_SEED,
                N=N,
                M=M,
                noise_var=NOISE_VAR,
                lambda_=LAMBDA,
                global_data=global_data,
            )
            num_observed = int(shared_data["num_observed"])
            lr_alpha = resolve_lr(alpha_key)
            effective_epochs = compute_effective_epochs(
                num_observed=num_observed,
                batch_size=BATCH_SIZE,
                max_steps=MAX_STEPS,
            )
            print(
                f"alpha={alpha_key:.2f}: E={num_observed}, lr={lr_alpha:.3e}, "
                f"effective_epochs={effective_epochs:.2f}"
            )

            cosine_similarity_values: list[float] = []
            loss_values: list[float] = []
            steps_values: list[int] = []

            for replica_id in range(NUM_REPLICAS):
                replica_seed = STUDENT_SEED_BASE + replica_id
                t0 = time.time()
                cosine_similarity, final_loss, steps_taken = train_single_replica(
                    alpha=alpha_key,
                    device=device,
                    seed=replica_seed,
                    N=N,
                    M=M,
                    max_steps=MAX_STEPS,
                    lr=lr_alpha,
                    batch_size=BATCH_SIZE,
                    noise_var=NOISE_VAR,
                    lambda_=LAMBDA,
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
                        "alpha": alpha_key,
                        "lambda": LAMBDA,
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
                results[alpha_key] = build_alpha_result(
                    lambda_=LAMBDA,
                    lr_alpha=lr_alpha,
                    num_observed=num_observed,
                    effective_epochs=effective_epochs,
                    cosine_similarity_values=cosine_similarity_values,
                    loss_values=loss_values,
                    steps_values=steps_values,
                )
                print(
                    f"alpha={alpha_key:.2f}, replica {replica_id + 1}/{NUM_REPLICAS}: "
                    f"CosSim={cosine_similarity:.4f}, Loss={final_loss:.2e}, "
                    f"Steps={steps_taken} ({dt:.1f}s) [{completed}/{total_tasks}]"
                )
                if completed % SAVE_EVERY_REPLICAS == 0:
                    save_progress_outputs(
                        results_dir=results_dir,
                        results=results,
                        replica_records=replica_records,
                        num_replicas=NUM_REPLICAS,
                        completed=completed,
                        total_tasks=total_tasks,
                        start_time=start_time,
                        status="running",
                    )
                    print(f"Progress saved: {results_dir}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving partial outputs before exit...")
        save_progress_outputs(
            results_dir=results_dir,
            results=results,
            replica_records=replica_records,
            num_replicas=NUM_REPLICAS,
            completed=completed,
            total_tasks=total_tasks,
            start_time=start_time,
            status="interrupted",
        )

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

    if not results:
        print("\nNo completed replicas yet; metrics headers were saved.")
        print(f"Results saved to: {results_dir}")
        sys.exit(130 if interrupted else 0)

    alphas_list = sorted(results.keys())
    cosine_similarity_means = [results[a]["cosine_similarity_mean"] for a in alphas_list]
    cosine_similarity_stds = [results[a]["cosine_similarity_std"] for a in alphas_list]

    save_progress_outputs(
        results_dir=results_dir,
        results=results,
        replica_records=replica_records,
        num_replicas=NUM_REPLICAS,
        completed=completed,
        total_tasks=total_tasks,
        start_time=start_time,
        status="interrupted" if interrupted else "completed",
    )
    print(f"Metrics saved: {results_dir / 'metrics.csv'}")
    print(f"Replica summary saved: {results_dir / 'replica_summary.csv'}")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        alphas_list,
        cosine_similarity_means,
        yerr=cosine_similarity_stds,
        fmt="o-",
        color="#1E88E5",
        markersize=6,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
    )
    ax.set_xlabel(r"$\alpha$ (observation density)", fontsize=14)
    ax.set_ylabel("Cosine Similarity in full Y-space", fontsize=14)
    ax.set_title(
        f"Phase Transition (Mini-Batch SGD, F-random)\n"
        f"(N={N}, M={M}, lambda={LAMBDA}, {MAX_STEPS} steps, "
        f"{NUM_REPLICAS} replicas)",
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
