#!/usr/bin/env python
"""
Order-parameter-vs-step runner for one-variable symmetric random-F AMP.
"""

from __future__ import annotations

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

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Edge_Alternating.random_F_one_variable_sequentially.core import (  # noqa: E402
    DEFAULT_EDGE_CHUNK_SIZE,
    ORDER_PARAMETER_KEYS,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)

HISTORY_ORDER_PARAMETER_KEYS = list(ORDER_PARAMETER_KEYS)
FINAL_PAIR_Q_KEYS = ["q_W"]

DEFAULT_ALPHA = 1.6
DEFAULT_BETA_SCALE = 1e-2
DEFAULT_BETA_MAX = 0.4
DEFAULT_SHARED_SEED = 1
DEFAULT_STUDENT_SEED_BASE = 100
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_SAVE_EVERY_REPLICAS = 1
DEFAULT_TORCH_THREADS = 1


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.devices:
        raw_devices = [part.strip() for part in args.devices.split(",") if part.strip()]
        if not raw_devices:
            raise ValueError("--devices was provided but no devices were parsed.")
        devices = [
            dev
            if dev.startswith("cuda:") or dev in {"cpu", "mps"}
            else f"cuda:{dev}"
            for dev in raw_devices
        ]
        if any(dev.startswith("cuda:") for dev in devices) and not torch.cuda.is_available():
            raise RuntimeError("CUDA devices were requested, but CUDA is unavailable.")
        if any(dev == "mps" for dev in devices) and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is unavailable.")
        return devices

    if torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return ["mps"]
    if args.allow_cpu:
        return ["cpu" for _ in range(max(1, args.cpu_workers))]

    raise RuntimeError(
        "CUDA/MPS is unavailable. Run on a GPU node or pass --allow-cpu for a smoke test."
    )


def write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-alpha order-parameter-vs-step replicas for the one-variable "
            "symmetric random-F AMP variant."
        )
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Set the one-variable system size N. Overrides --N1 and --N2.",
    )
    parser.add_argument("--N1", type=int, default=1250)
    parser.add_argument("--N2", type=int, default=1250)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--damping", type=float, default=0)
    parser.add_argument(
        "--damping-schedule",
        type=str,
        choices=["beta", "constant"],
        default="constant",
    )
    parser.add_argument("--beta-scale", type=float, default=DEFAULT_BETA_SCALE)
    parser.add_argument("--beta-max", type=float, default=DEFAULT_BETA_MAX)
    parser.add_argument("--noise-var", type=float, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SHARED_SEED)
    parser.add_argument("--shared-seed", type=int, default=DEFAULT_SHARED_SEED)
    parser.add_argument("--student-seed-base", type=int, default=DEFAULT_STUDENT_SEED_BASE)
    parser.add_argument("--num-replicas", type=int, default=2)
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
    )
    parser.add_argument(
        "--init-epsilon",
        type=float,
        default=0.01,
        help=(
            "Use correlated student initialization: epsilon * teacher + "
            "sqrt(epsilon - epsilon^2) * N(0, 1)."
        ),
    )
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=DEFAULT_EDGE_CHUNK_SIZE,
        help="Number of observed edges processed at once.",
    )
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--save-every-replicas",
        type=int,
        default=DEFAULT_SAVE_EVERY_REPLICAS,
    )
    args = parser.parse_args()

    if args.N is not None:
        args.N1 = args.N
        args.N2 = args.N
    return args


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    devices: list[str],
    shared_metadata: dict[str, int | float],
) -> None:
    config = {
        "algorithm": "gamp_one_variable_random_F_symmetric_W_order_parameters_vs_step",
        "graph_model": "random_graph_row_degree_reused_from_biregular_generator",
        "f_mode": "random",
        "f_distribution": "rademacher_pm1",
        "effective_F_values": "+/- lambda / sqrt(M)",
        "sequential_aggregation": True,
        "one_variable": True,
        "student_matrix": "W",
        "observation_model": "Y_ij = lambda/sqrt(M) * sum_mu F_ijmu W_i,mu W_j,mu + noise",
        "self_edges": "excluded",
        "symmetric_endpoint_update": True,
        "algorithm_note": "C++ p=2 comparison-oriented one-variable update",
        "stores_F_edge": False,
        "edge_chunk_size": args.edge_chunk_size,
        "parallelism": "sequential_replicas_round_robin_over_requested_devices",
        "alpha": args.alpha,
        "lambda": args.lam,
        "N": args.N1,
        "N1_cli": args.N1,
        "N2_cli": args.N2,
        "M": args.M,
        "max_steps": args.max_steps,
        "damping": args.damping,
        "damping_schedule": args.damping_schedule,
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "noise_var": args.noise_var,
        "teacher_seed": args.shared_seed,
        "graph_seed": args.shared_seed,
        "noise_seed": args.shared_seed,
        "f_seed": args.shared_seed + 1000,
        "student_seed_base": args.student_seed_base,
        "student_init_mode": (
            "correlated_gaussian" if args.init_epsilon is not None else "random_gaussian"
        ),
        "student_init_formula": (
            "epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1)"
            if args.init_epsilon is not None
            else "N(0, 1)"
        ),
        "student_init_epsilon": args.init_epsilon,
        "legacy_cli_seed": args.seed,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "order_parameter_eval_interval": 1,
        "includes_initial_state": True,
        "initial_state_step": 0,
        "initial_convergence": "nan_no_previous_state",
        "early_stop": False,
        "devices": devices,
        "torch_threads": args.torch_threads,
        "deterministic_requested": args.deterministic,
        "order_parameters": list(HISTORY_ORDER_PARAMETER_KEYS),
        "final_pair_order_parameters": list(FINAL_PAIR_Q_KEYS),
        "q_definition": "final-step average over all replica pairs a<b; q_W=mean(m_W^a*m_W^b)",
        "minimum_replicas_for_q": 2,
        "convergence_definition": "sum_abs_pre_damping_student_proposal_minus_old divided by (N*M)",
        "convergence_plot_yscale": "log10",
        "update_scheme": "one_variable_symmetric_endpoint_update",
        "step_definition": "all edges accumulate both endpoint contributions, then W is updated synchronously",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "shared_random_F_per_alpha": True,
        "raw_edges_before_self_edge_filter": shared_metadata.get("E_raw"),
        "observed_edges_after_self_edge_filter": shared_metadata.get("E"),
        "removed_self_edges": shared_metadata.get("removed_self_edges"),
        "output_files": [
            "config.yaml",
            "order_parameters_history.csv",
            "replica_summary.csv",
            "final_pair_q_summary.csv",
            "final_pair_q_pairs.csv",
            "progress.yaml",
            "plots/m_overlap_W_vs_step.png",
            "plots/Q_W_vs_step.png",
            "plots/convergence_vs_step.png",
        ],
    }
    write_text_atomic(
        results_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )


def estimate_convergence_step_from_steps(
    steps: np.ndarray,
    convergence_history: np.ndarray,
    threshold: float,
) -> float:
    if convergence_history.size < 2:
        return float("nan")

    stable_idx = np.where((steps > 0) & (convergence_history < threshold))[0]
    if stable_idx.size == 0:
        return float("nan")
    return float(steps[stable_idx[0]])


def build_history_arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ordered = sorted(records, key=lambda r: r["replica_id"])
    steps = np.asarray(ordered[0]["steps"], dtype=np.int64)
    for record in ordered[1:]:
        record_steps = np.asarray(record["steps"], dtype=np.int64)
        if not np.array_equal(steps, record_steps):
            raise RuntimeError("Inconsistent step grids across replicas.")

    order_parameter_histories = {
        key: np.asarray(
            [record[f"{key}_history"] for record in ordered],
            dtype=np.float64,
        )
        for key in HISTORY_ORDER_PARAMETER_KEYS
    }
    return steps, order_parameter_histories


def save_order_parameter_history(
    results_dir: Path,
    steps: np.ndarray,
    order_parameter_histories: dict[str, np.ndarray],
) -> None:
    header = ["step"]
    for key in HISTORY_ORDER_PARAMETER_KEYS:
        header.extend([f"{key}_mean", f"{key}_std"])
    for key in HISTORY_ORDER_PARAMETER_KEYS:
        values = order_parameter_histories[key]
        header.extend([f"{key}_replica_{idx + 1}" for idx in range(values.shape[0])])

    lines = [",".join(header)]
    for step_idx, step in enumerate(steps):
        row = [str(int(step))]
        for key in HISTORY_ORDER_PARAMETER_KEYS:
            values = order_parameter_histories[key]
            row.extend(
                [
                    f"{values[:, step_idx].mean():.10e}",
                    f"{values[:, step_idx].std():.10e}",
                ]
            )
        for key in HISTORY_ORDER_PARAMETER_KEYS:
            values = order_parameter_histories[key]
            row.extend(f"{value:.10e}" for value in values[:, step_idx])
        lines.append(",".join(row))

    write_text_atomic(
        results_dir / "order_parameters_history.csv",
        "\n".join(lines) + "\n",
    )


def save_replica_summary(results_dir: Path, records: list[dict[str, Any]]) -> None:
    header = [
        "replica",
        "seed",
        "device",
        "runtime_sec",
        "estimated_convergence_step",
    ]
    header.extend(HISTORY_ORDER_PARAMETER_KEYS)

    lines = [",".join(header)]
    for record in sorted(records, key=lambda r: r["replica_id"]):
        convergence_value = (
            ""
            if math.isnan(record["estimated_convergence_step"])
            else str(int(record["estimated_convergence_step"]))
        )
        row = [
            str(record["replica"]),
            str(record["seed"]),
            str(record["device"]),
            f"{record['runtime_sec']:.4f}",
            convergence_value,
        ]
        row.extend(f"{record[key]:.10e}" for key in HISTORY_ORDER_PARAMETER_KEYS)
        lines.append(",".join(row))

    write_text_atomic(results_dir / "replica_summary.csv", "\n".join(lines) + "\n")


def compute_final_pair_q_records(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    ordered = sorted(records, key=lambda r: r["replica_id"])
    pair_records: list[dict[str, float]] = []
    for left_idx, left in enumerate(ordered):
        m_left = np.asarray(left["final_m_W"], dtype=np.float64)
        for right in ordered[left_idx + 1:]:
            m_right = np.asarray(right["final_m_W"], dtype=np.float64)
            if m_left.shape != m_right.shape:
                raise RuntimeError("Final W shapes differ across replicas.")
            pair_records.append(
                {
                    "replica_a": float(left["replica"]),
                    "replica_b": float(right["replica"]),
                    "q_W": float(np.mean(m_left * m_right)),
                }
            )
    return pair_records


def compute_final_pair_q_summary(records: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    pair_records = compute_final_pair_q_records(records)
    if not pair_records:
        raise RuntimeError("At least two replicas are required to compute q_W.")

    values = np.asarray([record["q_W"] for record in pair_records], dtype=np.float64)
    return {"q_W": (float(np.mean(values)), float(np.std(values)))}


def save_final_pair_q_outputs(results_dir: Path, records: list[dict[str, Any]]) -> None:
    pair_records = compute_final_pair_q_records(records)
    if not pair_records:
        return

    pair_lines = ["replica_a,replica_b,q_W"]
    for record in pair_records:
        pair_lines.append(
            f"{int(record['replica_a'])},{int(record['replica_b'])},{record['q_W']:.10e}"
        )
    write_text_atomic(results_dir / "final_pair_q_pairs.csv", "\n".join(pair_lines) + "\n")

    values = np.asarray([record["q_W"] for record in pair_records], dtype=np.float64)
    summary_lines = [
        "quantity,mean,std,num_pairs",
        f"q_W,{float(np.mean(values)):.10e},{float(np.std(values)):.10e},{len(pair_records)}",
    ]
    write_text_atomic(
        results_dir / "final_pair_q_summary.csv",
        "\n".join(summary_lines) + "\n",
    )


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
        valid_counts = np.sum(np.isfinite(plot_histories), axis=0)
        sums = np.nansum(plot_histories, axis=0)
        mean_values = np.divide(
            sums,
            valid_counts,
            out=np.full_like(sums, np.nan),
            where=valid_counts > 0,
        )
        centered = plot_histories - mean_values
        centered[~np.isfinite(centered)] = np.nan
        variances = np.divide(
            np.nansum(centered ** 2, axis=0),
            valid_counts,
            out=np.full_like(sums, np.nan),
            where=valid_counts > 0,
        )
        std_values = np.sqrt(variances)
    else:
        mean_values = plot_histories.mean(axis=0)
        std_values = plot_histories.std(axis=0)

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
        ax.set_yscale("log")
        positive_values = plot_histories[np.isfinite(plot_histories)]
        if positive_values.size > 0:
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
        f"{title_prefix} vs Step (alpha={args.alpha}, N={args.N1}, "
        f"M={args.M}, {len(histories)} replicas)",
        fontsize=14,
    )
    ax.grid(True, which="both" if log_y else "major", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_progress_outputs(
    results_dir: Path,
    records: list[dict[str, Any]],
    completed: int,
    total_tasks: int,
    start_time: float,
    status: str,
    args: argparse.Namespace,
) -> None:
    if records:
        steps, order_parameter_histories = build_history_arrays(records)
        save_order_parameter_history(results_dir, steps, order_parameter_histories)
        save_replica_summary(results_dir, records)
        if len(records) >= 2:
            save_final_pair_q_outputs(results_dir, records)

        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_order_parameter(
            plots_dir,
            steps,
            order_parameter_histories["m_overlap_W"],
            "m_overlap_W",
            "m_overlap_W",
            "m_overlap_W_vs_step.png",
            args,
        )
        plot_order_parameter(
            plots_dir,
            steps,
            order_parameter_histories["Q_W"],
            "Q_W",
            "Q_W",
            "Q_W_vs_step.png",
            args,
        )
        plot_order_parameter(
            plots_dir,
            steps,
            order_parameter_histories["convergence"],
            "convergence",
            "Convergence",
            "convergence_vs_step.png",
            args,
            log_y=True,
        )

    progress = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_tasks": completed,
        "total_tasks": total_tasks,
        "elapsed_sec": time.time() - start_time,
    }
    write_text_atomic(
        results_dir / "progress.yaml",
        yaml.safe_dump(progress, sort_keys=False),
    )


def run_replicas(
    devices: list[str],
    args: argparse.Namespace,
    results_dir: Path,
    start_time: float,
) -> list[dict[str, Any]]:
    shared_by_device: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for replica_id in range(args.num_replicas):
        device_name = devices[replica_id % len(devices)]
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if device_name not in shared_by_device:
            global_data = prepare_global_shared_data(
                device=device,
                seed=args.shared_seed,
                N=args.N1,
                M=args.M,
                noise_var=args.noise_var,
                lam=args.lam,
            )
            shared_by_device[device_name] = prepare_shared_alpha_data(
                alpha=args.alpha,
                device=device,
                seed=args.shared_seed,
                N=args.N1,
                M=args.M,
                noise_var=args.noise_var,
                lam=args.lam,
                edge_chunk_size=args.edge_chunk_size,
                global_data=global_data,
            )

        seed = args.student_seed_base + replica_id
        t0 = time.time()
        _metric, _loss, steps_taken, history, final_state = train_single_replica(
            alpha=args.alpha,
            device=device,
            seed=seed,
            N=args.N1,
            M=args.M,
            max_steps=args.max_steps,
            damping=args.damping,
            use_step_damping=args.damping_schedule == "beta",
            damping_beta_scale=args.beta_scale,
            damping_beta_max=args.beta_max,
            noise_var=args.noise_var,
            lam=args.lam,
            convergence_threshold=args.convergence_threshold,
            return_history=True,
            eval_interval=1,
            early_stop=False,
            init_epsilon=args.init_epsilon,
            edge_chunk_size=args.edge_chunk_size,
            shared_data=shared_by_device[device_name],
            return_final_state=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime = time.time() - t0

        steps = np.asarray(history["steps"], dtype=np.int64)
        order_parameter_histories = {
            key: np.asarray(history[key], dtype=np.float64)
            for key in HISTORY_ORDER_PARAMETER_KEYS
        }
        convergence_step = estimate_convergence_step_from_steps(
            steps,
            order_parameter_histories["convergence"],
            args.convergence_threshold,
        )
        record = {
            "replica_id": replica_id,
            "replica": replica_id + 1,
            "seed": seed,
            "device": device_name,
            "runtime_sec": runtime,
            "steps": steps.tolist(),
            "steps_taken": int(steps_taken),
            "estimated_convergence_step": float(convergence_step),
            "final_m_W": final_state["m_W"].numpy().astype(np.float32, copy=True),
        }
        for key in HISTORY_ORDER_PARAMETER_KEYS:
            values = order_parameter_histories[key]
            record[f"{key}_history"] = values.tolist()
            record[key] = float(values[-1]) if values.size else float("nan")
        records.append(record)

        convergence_text = (
            "not reached"
            if math.isnan(record["estimated_convergence_step"])
            else str(int(record["estimated_convergence_step"]))
        )
        print(
            f"Replica {record['replica']}/{args.num_replicas}: "
            f"device={record['device']}, seed={record['seed']}, "
            f"estimated_convergence_step={convergence_text}, "
            f"steps_taken={record['steps_taken']}, "
            f"history_points={len(record['steps'])}, "
            f"runtime={record['runtime_sec']:.1f}s [{len(records)}/{args.num_replicas}]"
        )
        if len(records) % args.save_every_replicas == 0 or len(records) == args.num_replicas:
            save_progress_outputs(
                results_dir=results_dir,
                records=records,
                completed=len(records),
                total_tasks=args.num_replicas,
                start_time=start_time,
                status="running",
                args=args,
            )

    return records


def main() -> int:
    args = parse_args()
    if args.num_replicas < 2:
        raise ValueError("q_W requires --num-replicas >= 2.")
    if args.N1 != args.N2:
        raise ValueError("The one-variable model requires N1 == N2. Use --N or matching --N1/--N2.")
    if args.edge_chunk_size <= 0:
        raise ValueError("--edge-chunk-size must be positive.")
    if args.save_every_replicas <= 0:
        raise ValueError("--save-every-replicas must be positive.")
    torch.set_num_threads(args.torch_threads)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    devices = resolve_devices(args)

    print("=" * 72)
    print("One-Variable Symmetric Random-F AMP Order Parameters vs Step")
    print("=" * 72)
    print(f"Devices: {', '.join(devices)}")
    print(f"alpha={args.alpha}, N={args.N1}, M={args.M}")
    print(f"lambda={args.lam}")
    print("Self edges: excluded")
    print("Update: each edge contributes to both endpoint rows before synchronous W update")
    print(f"edge_chunk_size={args.edge_chunk_size}; F_edge is regenerated per chunk")
    if args.damping_schedule == "beta":
        print(
            f"max_steps={args.max_steps}, damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"max_steps={args.max_steps}, damping={args.damping}")
    print("Teacher / graph / noise seed:", args.shared_seed)
    print("F seed:", args.shared_seed + 1000)
    print("Student seed rule:", f"{args.student_seed_base} + replica_index")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = (
        args.results_root
        if args.results_root is not None
        else Path(__file__).parent / "results"
    )
    results_dir = results_root / (
        f"{timestamp}_order_parameters_vs_step_Edge_Alternating_random_F_one_variable_"
        f"alpha{args.alpha}_lambda{args.lam}_{args.N1}x{args.N1}_M{args.M}_chunk{args.edge_chunk_size}"
        f"_initeps{args.init_epsilon if args.init_epsilon is not None else 'random'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Build once on the first device only for metadata stored in config.
    metadata_device = torch.device(devices[0])
    global_data = prepare_global_shared_data(
        device=metadata_device,
        seed=args.shared_seed,
        N=args.N1,
        M=args.M,
        noise_var=args.noise_var,
        lam=args.lam,
    )
    shared_metadata_data = prepare_shared_alpha_data(
        alpha=args.alpha,
        device=metadata_device,
        seed=args.shared_seed,
        N=args.N1,
        M=args.M,
        noise_var=args.noise_var,
        lam=args.lam,
        edge_chunk_size=args.edge_chunk_size,
        global_data=global_data,
    )
    shared_metadata = {
        "E_raw": int(shared_metadata_data["E_raw"]),
        "E": int(shared_metadata_data["E"]),
        "removed_self_edges": int(shared_metadata_data["removed_self_edges"]),
    }
    save_config(results_dir, args, devices, shared_metadata)
    del shared_metadata_data, global_data

    start_time = time.time()
    interrupted = False
    records: list[dict[str, Any]] = []
    try:
        records = run_replicas(
            devices=devices,
            args=args,
            results_dir=results_dir,
            start_time=start_time,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving partial results...")
    finally:
        save_progress_outputs(
            results_dir=results_dir,
            records=records,
            completed=len(records),
            total_tasks=args.num_replicas,
            start_time=start_time,
            status="interrupted" if interrupted else "completed",
            args=args,
        )

    if records:
        q_summary = compute_final_pair_q_summary(records)
        print()
        print("Final replica-pair q averages:")
        for key in FINAL_PAIR_Q_KEYS:
            mean_value, std_value = q_summary[key]
            print(f"  {key}: {mean_value:.10f} +- {std_value:.10f}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")
    print(f"Results saved to: {results_dir}")
    return 0 if not interrupted else 130


if __name__ == "__main__":
    raise SystemExit(main())

