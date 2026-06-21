#!/usr/bin/env python
"""
Record AGD order parameters vs step for fixed alpha, N1, N2, and M.

The convergence value follows CLAUDE.md for GD/AGD:
    convergence_t = abs(loss_per_edge_t - loss_per_edge_{t-1})
where
    loss_per_edge = (M / |E_obs|) * sum_{(i,j) in E_obs} residual(i,j)^2.
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np
import torch
import yaml

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph
from terao_gd.gd_cosine.gd import (
    ORDER_PARAMETER_KEYS,
    compute_order_parameters,
    initialize_student_factors,
)


HISTORY_ORDER_PARAMETER_KEYS = [
    key for key in ORDER_PARAMETER_KEYS if key not in {"q_Y", "cosine_Y", "q_W", "q_X"}
]
HISTORY_KEYS = HISTORY_ORDER_PARAMETER_KEYS + ["loss_per_edge", "convergence"]
FINAL_PAIR_Q_KEYS = ["q_W", "q_X", "q_Y"]


def parse_init_epsilon(value: str) -> float | None:
    lowered = str(value).strip().lower()
    if lowered in {"none", "random"}:
        return None
    try:
        epsilon = float(lowered)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--init-epsilon must be a float in [0, 1], 'none', or 'random'."
        ) from exc
    if not 0.0 <= epsilon <= 1.0:
        raise argparse.ArgumentTypeError("--init-epsilon must satisfy 0 <= epsilon <= 1.")
    return epsilon


def compute_predictions(
    W: torch.Tensor,
    X: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    M: int,
    lam: float = 1.0,
) -> torch.Tensor:
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (float(lam) / math.sqrt(M)) * (W_sel * X_sel).sum(dim=1)


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    return M * ((Y - Y_pred) ** 2).sum()


def compute_loss_per_edge(
    Y: torch.Tensor,
    Y_pred: torch.Tensor,
    M: int,
) -> torch.Tensor:
    return compute_loss(Y, Y_pred, M) / max(Y.numel(), 1)


def agd_step_W(
    W: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
    lam: float,
) -> torch.Tensor:
    _, M = W.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M, lam=lam)
    residual = Y_pred - Y
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * float(lam) * math.sqrt(M) * residual.unsqueeze(1) * X_sel
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
    lam: float,
) -> torch.Tensor:
    M, _ = X.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M, lam=lam)
    residual = Y_pred - Y
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * float(lam) * math.sqrt(M) * residual.unsqueeze(1) * W_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    return X - lr * grad_X


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    mean_sq = (tensor ** 2).mean()
    if mean_sq > 0:
        return tensor / torch.sqrt(mean_sq)
    return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record AGD order parameters vs step for fixed alpha."
    )
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--N1", type=int, default=1250)
    parser.add_argument("--N2", type=int, default=1250)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument(
        "--lam",
        "--lambda",
        dest="lam",
        type=float,
        default=1.0,
        help="Signal scale lambda in Y = lambda / sqrt(M) * W X + noise.",
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Defaults to gd.py-style auto scaling.",
    )
    parser.add_argument("--lr-base", type=float, default=0.01)
    parser.add_argument("--noise-var", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument(
        "--init-epsilon",
        type=parse_init_epsilon,
        default=0.01,
        help=(
            "Use informative student initialization: epsilon * teacher + "
            "sqrt(epsilon - epsilon^2) * N(0, 1). Pass 'none' or 'random' "
            "for random Gaussian initialization."
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Directory in which to create the timestamped results directory.",
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
    return float(args.lr_base * (1e6 / (args.N1 * args.N2 * args.M)))


def estimate_convergence_step(
    steps: np.ndarray,
    convergence_history: np.ndarray,
    threshold: float,
) -> float:
    reached = np.where((steps > 0) & (convergence_history < threshold))[0]
    if reached.size == 0:
        return float("nan")
    return float(steps[reached[0]])


def mean_std_ignore_nan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_counts = np.sum(np.isfinite(values), axis=0)
    sums = np.nansum(values, axis=0)
    means = np.divide(
        sums,
        valid_counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    centered = values - means
    centered[~np.isfinite(centered)] = np.nan
    variances = np.divide(
        np.nansum(centered ** 2, axis=0),
        valid_counts,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    return means, np.sqrt(variances)


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    lr: float,
) -> None:
    config = {
        "algorithm": "agd_order_parameters_vs_step",
        "alpha": args.alpha,
        "lambda": args.lam,
        "signal_scale": "lambda / sqrt(M)",
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "max_steps": args.max_steps,
        "lr": lr,
        "lr_base": args.lr_base,
        "noise_var": args.noise_var,
        "seed": args.seed,
        "num_replicas": args.num_replicas,
        "student_init_mode": (
            "random_gaussian" if args.init_epsilon is None else "correlated_gaussian"
        ),
        "student_init_formula": (
            "N(0, 1)"
            if args.init_epsilon is None
            else "epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1)"
        ),
        "student_init_epsilon": args.init_epsilon,
        "convergence_threshold": args.convergence_threshold,
        "convergence_definition": "abs(loss_per_edge_t - loss_per_edge_{t-1})",
        "early_stop": True,
        "early_stop_metric": "convergence",
        "early_stop_rule": "stop when convergence < convergence_threshold",
        "loss_definition": (
            "(M / |E_obs|) * sum_{(i,j) in E_obs} "
            "(Y_obs[i,j] - Y_hat_t[i,j])^2, where "
            "Y_hat_t = lambda / sqrt(M) * sum_mu W_t[i,mu] X_t[mu,j]"
        ),
        "record_interval": args.record_interval,
        "record_interval_note": (
            "convergence is computed between stored records; use "
            "record_interval=1 for adjacent-step convergence"
        ),
        "device": str(device),
        "evaluation_metric": "dense_teacher_student_overlap_per_replica",
        "order_parameters": list(HISTORY_ORDER_PARAMETER_KEYS),
        "final_pair_order_parameters": list(FINAL_PAIR_Q_KEYS),
        "q_definition": (
            "final-step average over all replica pairs a<b; "
            "q_W=mean(W_hat^a*W_hat^b), q_X=mean(X_hat^a*X_hat^b), "
            "q_Y=(sum_mu sum_i W_hat^a[i,mu]W_hat^b[i,mu] "
            "sum_j X_hat^a[mu,j]X_hat^b[mu,j])/(N1*N2*M)"
        ),
        "minimum_replicas_for_q": 2,
        "unavailable_order_parameters": {
            "Q_W": "GD does not maintain a posterior variance estimate v_W.",
            "Q_X": "GD does not maintain a posterior variance estimate v_X.",
        },
        "output_files": [
            "config.yaml",
            "order_parameters_history.csv",
            "replica_summary.csv",
            "final_pair_q_summary.csv",
            "final_pair_q_pairs.csv",
            "plots/m_overlap_Y_vs_step.png",
            "plots/convergence_vs_step.png",
        ],
    }

    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_order_parameter_history(
    results_dir: Path,
    steps: np.ndarray,
    histories: dict[str, np.ndarray],
) -> None:
    history_path = results_dir / "order_parameters_history.csv"
    header = ["step"]
    for key in HISTORY_KEYS:
        header.extend([f"{key}_mean", f"{key}_std"])
    for key in HISTORY_KEYS:
        values = histories[key]
        header.extend([f"{key}_replica_{idx + 1}" for idx in range(values.shape[0])])

    summary_stats = {
        key: mean_std_ignore_nan(histories[key])
        for key in HISTORY_KEYS
    }
    lines = [",".join(header)]
    for step_idx, step in enumerate(steps):
        row = [str(int(step))]
        for key in HISTORY_KEYS:
            mean_values, std_values = summary_stats[key]
            row.extend(
                [
                    f"{mean_values[step_idx]:.10e}",
                    f"{std_values[step_idx]:.10e}",
                ]
            )
        for key in HISTORY_KEYS:
            values = histories[key]
            row.extend(f"{value:.10e}" for value in values[:, step_idx])
        lines.append(",".join(row))

    with open(history_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_replica_summary(results_dir: Path, records: list[dict[str, Any]]) -> None:
    summary_path = results_dir / "replica_summary.csv"
    header = [
        "replica",
        "seed",
        "runtime_sec",
        "estimated_convergence_step",
    ]
    header.extend(HISTORY_KEYS)

    lines = [",".join(header)]
    for record in records:
        convergence_value = (
            ""
            if math.isnan(record["estimated_convergence_step"])
            else str(int(record["estimated_convergence_step"]))
        )
        row = [
            str(int(record["replica"])),
            str(int(record["seed"])),
            f"{record['runtime_sec']:.4f}",
            convergence_value,
        ]
        row.extend(f"{record[key]:.10e}" for key in HISTORY_KEYS)
        lines.append(",".join(row))

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def compute_final_pair_q_records(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    ordered = sorted(records, key=lambda r: r["replica"])
    pair_records: list[dict[str, float]] = []

    for left_idx, left in enumerate(ordered):
        w_left = np.asarray(left["final_W"], dtype=np.float64)
        x_left = np.asarray(left["final_X"], dtype=np.float64)
        for right in ordered[left_idx + 1:]:
            w_right = np.asarray(right["final_W"], dtype=np.float64)
            x_right = np.asarray(right["final_X"], dtype=np.float64)

            if w_left.shape != w_right.shape or x_left.shape != x_right.shape:
                raise RuntimeError("Final student shapes differ across replicas.")

            n1, m_rank = w_left.shape
            m_rank_x, n2 = x_left.shape
            if m_rank != m_rank_x:
                raise RuntimeError("Inconsistent W/X rank in final student states.")

            w_cross_by_mu = np.sum(w_left * w_right, axis=0)
            x_cross_by_mu = np.sum(x_left * x_right, axis=1)
            q_y = float(np.sum(w_cross_by_mu * x_cross_by_mu) / (n1 * n2 * m_rank))

            pair_records.append(
                {
                    "replica_a": float(left["replica"]),
                    "replica_b": float(right["replica"]),
                    "q_W": float(np.mean(w_left * w_right)),
                    "q_X": float(np.mean(x_left * x_right)),
                    "q_Y": q_y,
                }
            )

    return pair_records


def compute_final_pair_q_summary(records: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    pair_records = compute_final_pair_q_records(records)
    if not pair_records:
        raise RuntimeError("At least two replicas are required to compute q.")

    summary: dict[str, tuple[float, float]] = {}
    for key in FINAL_PAIR_Q_KEYS:
        values = np.asarray([record[key] for record in pair_records], dtype=np.float64)
        summary[key] = (float(np.mean(values)), float(np.std(values)))
    return summary


def save_final_pair_q_outputs(results_dir: Path, records: list[dict[str, Any]]) -> None:
    pair_records = compute_final_pair_q_records(records)
    if not pair_records:
        return

    pairs_header = ["replica_a", "replica_b", *FINAL_PAIR_Q_KEYS]
    pair_lines = [",".join(pairs_header)]
    for record in pair_records:
        pair_lines.append(
            ",".join(
                [
                    str(int(record["replica_a"])),
                    str(int(record["replica_b"])),
                    *(f"{record[key]:.10e}" for key in FINAL_PAIR_Q_KEYS),
                ]
            )
        )
    with open(results_dir / "final_pair_q_pairs.csv", "w") as f:
        f.write("\n".join(pair_lines) + "\n")

    summary_lines = ["quantity,mean,std,num_pairs"]
    num_pairs = len(pair_records)
    for key in FINAL_PAIR_Q_KEYS:
        values = np.asarray([record[key] for record in pair_records], dtype=np.float64)
        summary_lines.append(
            f"{key},{float(np.mean(values)):.10e},{float(np.std(values)):.10e},{num_pairs}"
        )
    with open(results_dir / "final_pair_q_summary.csv", "w") as f:
        f.write("\n".join(summary_lines) + "\n")


def build_history_arrays(
    histories: list[dict[str, list[float]]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    all_steps = np.asarray(
        sorted({int(step) for history in histories for step in history["steps"]}),
        dtype=np.int64,
    )
    step_to_index = {int(step): idx for idx, step in enumerate(all_steps)}
    history_arrays = {
        key: np.full((len(histories), len(all_steps)), np.nan, dtype=np.float64)
        for key in HISTORY_KEYS
    }

    for replica_idx, history in enumerate(histories):
        for local_idx, step in enumerate(history["steps"]):
            step_idx = step_to_index[int(step)]
            for key in HISTORY_KEYS:
                history_arrays[key][replica_idx, step_idx] = history[key][local_idx]

    return all_steps, history_arrays


def plot_order_parameter(
    plots_dir: Path,
    steps: np.ndarray,
    histories: np.ndarray,
    key: str,
    title_prefix: str,
    filename: str,
    args: argparse.Namespace,
    log_y: bool = False,
) -> None:
    plot_histories = histories.astype(np.float64, copy=True)
    if log_y:
        plot_histories[plot_histories <= 0.0] = np.nan
    mean_values, std_values = mean_std_ignore_nan(plot_histories)

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, curve in enumerate(plot_histories):
        ax.plot(
            steps,
            curve,
            color="#B0BEC5",
            linewidth=1.2,
            alpha=0.6,
            label="Replica" if idx == 0 else None,
        )

    ax.plot(steps, mean_values, color="#2E7D32", linewidth=2.5, label=f"Mean {key}")
    lower_values = mean_values - std_values
    upper_values = mean_values + std_values
    if log_y:
        lower_values[lower_values <= 0.0] = np.nan
        upper_values[upper_values <= 0.0] = np.nan
    ax.fill_between(
        steps,
        lower_values,
        upper_values,
        color="#A5D6A7",
        alpha=0.35,
        label="Mean +- std",
    )

    if log_y:
        positive_values = plot_histories[np.isfinite(plot_histories)]
        if positive_values.size > 0:
            ax.set_yscale("log")
            y_min = 10.0 ** math.floor(math.log10(float(np.min(positive_values))))
            y_max = 10.0 ** math.ceil(math.log10(float(np.max(positive_values))))
            if y_min == y_max:
                y_min /= 10.0
                y_max *= 10.0
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
            ax.yaxis.set_major_formatter(
                LogFormatterMathtext(base=10.0, labelOnlyBase=True)
            )
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
            )
            ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel(key, fontsize=13)
    ax.set_title(
        f"{title_prefix} vs Step (alpha={args.alpha}, lambda={args.lam}, N1={args.N1}, "
        f"N2={args.N2}, M={args.M}, {args.num_replicas} replicas)",
        fontsize=14,
    )
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single_replica_with_history(
    alpha: float,
    device: torch.device,
    seed: int,
    N1: int,
    N2: int,
    M: int,
    lam: float,
    max_steps: int,
    lr: float,
    noise_var: float,
    record_interval: int,
    convergence_threshold: float,
    init_epsilon: float | None,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)

    W_hat, X_hat = initialize_student_factors(
        W_teacher,
        X_teacher,
        seed=seed,
        init_epsilon=init_epsilon,
    )

    graph = RandomGraph()
    i_idx, j_idx, edge_count = graph.generate(N1, N2, M, alpha, device, seed)
    Y_noisy = torch.empty(0, dtype=torch.float32, device=device)
    if edge_count > 0:
        Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M, lam=lam)
        torch.manual_seed(seed + 1000)
        noise = torch.randn_like(Y) * math.sqrt(noise_var)
        Y_noisy = Y + noise

    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    history["steps"] = []
    previous_loss: float | None = None

    def record(step: int, W_curr: torch.Tensor, X_curr: torch.Tensor) -> None:
        nonlocal previous_loss
        order_parameters = compute_order_parameters(
            W_curr,
            X_curr,
            W_teacher,
            X_teacher,
        )
        if edge_count > 0:
            Y_pred = compute_predictions(W_curr, X_curr, i_idx, j_idx, M, lam=lam)
            loss_per_edge = float(compute_loss_per_edge(Y_noisy, Y_pred, M).item())
        else:
            loss_per_edge = 0.0
        convergence = (
            float("nan")
            if previous_loss is None
            else abs(loss_per_edge - previous_loss)
        )
        previous_loss = loss_per_edge

        history["steps"].append(step)
        for key in HISTORY_ORDER_PARAMETER_KEYS:
            history[key].append(order_parameters[key])
        history["loss_per_edge"].append(loss_per_edge)
        history["convergence"].append(convergence)

    record(0, W_hat, X_hat)
    if edge_count == 0 or max_steps <= 0:
        final_values = {key: history[key][-1] for key in HISTORY_KEYS}
        final_state = {
            "W": W_hat.detach().cpu(),
            "X": X_hat.detach().cpu(),
        }
        return final_values, history, final_state

    for step in range(1, max_steps + 1):
        W_hat = agd_step_W(W_hat, X_hat, Y_noisy, i_idx, j_idx, lr, lam=lam)
        X_hat = agd_step_X(W_hat, X_hat, Y_noisy, i_idx, j_idx, lr, lam=lam)
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)

        if step % record_interval == 0 or step == max_steps:
            record(step, W_hat, X_hat)
            if history["convergence"][-1] < convergence_threshold:
                break

    final_values = {key: history[key][-1] for key in HISTORY_KEYS}
    final_state = {
        "W": W_hat.detach().cpu(),
        "X": X_hat.detach().cpu(),
    }
    return final_values, history, final_state


def main() -> None:
    args = parse_args()
    if args.num_replicas < 2:
        raise ValueError("q_W, q_X, and q_Y require --num-replicas >= 2.")
    if args.record_interval <= 0:
        raise ValueError("--record-interval must be positive.")

    device = detect_device()
    lr = resolve_lr(args)

    print("=" * 60)
    print("Order Parameters vs Step for Alternating Gradient Descent")
    print("Evaluation Metric: dense teacher-student overlap per replica")
    print("=" * 60)
    print(f"Device: {device}")
    print(
        f"alpha={args.alpha}, lambda={args.lam}, "
        f"N1={args.N1}, N2={args.N2}, M={args.M}"
    )
    print(
        f"max_steps={args.max_steps}, lr={lr:.6e}, "
        f"noise_var={args.noise_var:.6e}, record_interval={args.record_interval}"
    )
    print(f"replicas={args.num_replicas}, seed={args.seed}")
    print(
        "student_init="
        + (
            "N(0, 1)"
            if args.init_epsilon is None
            else (
                "epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1), "
                f"epsilon={args.init_epsilon}"
            )
        )
    )
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_agd_order_parameters_vs_step_alpha{args.alpha}_lambda{args.lam}_"
        f"{args.N1}x{args.N2}_M{args.M}_"
        f"initeps{args.init_epsilon if args.init_epsilon is not None else 'random'}"
    )
    results_root = args.results_root or Path(__file__).parent / "results"
    results_dir = results_root / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config(results_dir, args, device, lr)

    histories: list[dict[str, list[float]]] = []
    records: list[dict[str, Any]] = []
    total_start = time.time()

    for replica_idx in range(args.num_replicas):
        seed = args.seed + replica_idx * 1000
        replica_start = time.time()
        final_values, history, final_state = run_single_replica_with_history(
            alpha=args.alpha,
            device=device,
            seed=seed,
            N1=args.N1,
            N2=args.N2,
            M=args.M,
            lam=args.lam,
            max_steps=args.max_steps,
            lr=lr,
            noise_var=args.noise_var,
            record_interval=args.record_interval,
            convergence_threshold=args.convergence_threshold,
            init_epsilon=args.init_epsilon,
        )

        runtime = time.time() - replica_start
        steps = np.asarray(history["steps"], dtype=np.int64)
        convergence_history = np.asarray(history["convergence"], dtype=np.float64)
        convergence_step = estimate_convergence_step(
            steps,
            convergence_history,
            args.convergence_threshold,
        )
        convergence_text = (
            "not reached"
            if math.isnan(convergence_step)
            else str(int(convergence_step))
        )

        record = {
            "replica": float(replica_idx + 1),
            "seed": float(seed),
            "runtime_sec": runtime,
            "estimated_convergence_step": convergence_step,
        }
        record.update(final_values)
        record["final_W"] = final_state["W"].numpy().astype(np.float32, copy=True)
        record["final_X"] = final_state["X"].numpy().astype(np.float32, copy=True)
        records.append(record)
        histories.append(history)

        print(
            f"Replica {replica_idx + 1}/{args.num_replicas}: "
            f"seed={seed}, "
            f"final_loss_per_edge={final_values['loss_per_edge']:.2e}, "
            f"final_convergence={final_values['convergence']:.2e}, "
            f"estimated_convergence_step={convergence_text}, "
            f"runtime={runtime:.1f}s"
        )

    all_steps, history_arrays = build_history_arrays(histories)

    save_order_parameter_history(results_dir, all_steps, history_arrays)
    save_replica_summary(results_dir, records)
    save_final_pair_q_outputs(results_dir, records)
    plot_order_parameter(
        plots_dir,
        all_steps,
        history_arrays["m_overlap_Y"],
        "m_overlap_Y",
        "Dense m_overlap_Y",
        "m_overlap_Y_vs_step.png",
        args,
    )
    plot_order_parameter(
        plots_dir,
        all_steps,
        history_arrays["convergence"],
        "convergence",
        "Convergence",
        "convergence_vs_step.png",
        args,
        log_y=True,
    )

    total_runtime = time.time() - total_start
    q_summary = compute_final_pair_q_summary(records)
    print()
    print("Final replica-pair q averages:")
    for key in FINAL_PAIR_Q_KEYS:
        mean_value, std_value = q_summary[key]
        print(f"  {key}: {mean_value:.10f} +- {std_value:.10f}")
    print(
        "Mean final loss_per_edge: "
        f"{np.mean([record['loss_per_edge'] for record in records]):.2e}"
    )
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
