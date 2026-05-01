#!/usr/bin/env python
"""
Parallel loss-vs-step runner for the random-F version of Dence_Alternating.

This keeps the numerical update path in ``random_F_version.core`` and only
parallelizes replica execution. Each worker owns one device, reuses the shared
fixed-alpha teacher / graph / noise data on that device, and returns the full
recorded history for aggregation in the parent process.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

# Add parent directories to path.
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))


DEFAULT_ALPHA = 3.0
DEFAULT_N1 = 200
DEFAULT_N2 = 200
DEFAULT_M = 50
DEFAULT_MAX_STEPS = 50000
DEFAULT_DAMPING = 0.0
DEFAULT_DAMPING_SCHEDULE = "constant"
DEFAULT_BETA_SCALE = 1e-2
DEFAULT_BETA_MAX = 0.4
DEFAULT_NOISE_VAR = 1e-6
DEFAULT_SHARED_SEED = 1
DEFAULT_STUDENT_SEED_BASE = 100
DEFAULT_NUM_REPLICAS = 1
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_SAVE_EVERY_REPLICAS = 1
DEFAULT_TORCH_THREADS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-alpha loss-vs-step replicas for the random-F version "
            "of Dence_Alternating in parallel, one worker process per device."
        )
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Set N1=N2=N. Overrides --N1 and --N2 when provided.",
    )
    parser.add_argument("--N1", type=int, default=DEFAULT_N1)
    parser.add_argument("--N2", type=int, default=DEFAULT_N2)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument(
        "--damping-schedule",
        type=str,
        choices=["beta", "constant"],
        default=DEFAULT_DAMPING_SCHEDULE,
    )
    parser.add_argument("--beta-scale", type=float, default=DEFAULT_BETA_SCALE)
    parser.add_argument("--beta-max", type=float, default=DEFAULT_BETA_MAX)
    parser.add_argument("--noise-var", type=float, default=DEFAULT_NOISE_VAR)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SHARED_SEED,
        help=(
            "Legacy compatibility argument. This script uses the fixed shared "
            "seed policy from random_F_version."
        ),
    )
    parser.add_argument(
        "--shared-seed",
        type=int,
        default=DEFAULT_SHARED_SEED,
        help="Teacher / graph / noise seed.",
    )
    parser.add_argument(
        "--student-seed-base",
        type=int,
        default=DEFAULT_STUDENT_SEED_BASE,
    )
    parser.add_argument("--num-replicas", type=int, default=DEFAULT_NUM_REPLICAS)
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
    )
    parser.add_argument(
        "--init-epsilon",
        type=float,
        default=None,
        help=(
            "Use informative student initialization: teacher + epsilon * N(0, 1), "
            "then mean-square normalization. Omit for random Gaussian initialization."
        ),
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated device list, e.g. 0,1,2 or cuda:0,cuda:1 or cpu. "
            "Defaults to all visible CUDA devices, otherwise MPS, otherwise CPU "
            "only when --allow-cpu is set."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run on CPU when CUDA/MPS is unavailable. Intended for smoke tests.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Number of CPU workers when --allow-cpu is used without CUDA/MPS.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help="PyTorch intra-op threads per worker.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms inside each worker.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Output root. Defaults to this script's results/ directory.",
    )
    parser.add_argument(
        "--save-every-replicas",
        type=int,
        default=DEFAULT_SAVE_EVERY_REPLICAS,
        help="Write partial outputs after this many completed replicas.",
    )
    return parser.parse_args()


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.N is not None:
        args.N1 = args.N
        args.N2 = args.N

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


def estimate_convergence_step(loss_history: np.ndarray, threshold: float) -> float:
    if loss_history.size < 2:
        return float("nan")

    delta = np.abs(np.diff(loss_history))
    stable_idx = np.where(delta < threshold)[0]
    if stable_idx.size == 0:
        return float("nan")

    return float(stable_idx[0] + 2)


def assign_replicas_to_devices(
    num_replicas: int,
    devices: list[str],
    student_seed_base: int,
) -> list[list[dict[str, int]]]:
    tasks_by_device: list[list[dict[str, int]]] = [[] for _ in devices]
    for replica_id in range(num_replicas):
        device_slot = replica_id % len(devices)
        tasks_by_device[device_slot].append(
            {
                "replica_id": replica_id,
                "seed": student_seed_base + replica_id,
            }
        )
    return tasks_by_device


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    devices: list[str],
) -> None:
    config = {
        "algorithm": "gamp_Dence_Alternating_random_F_loss_vs_step_parallel",
        "graph_model": "random_graph",
        "parallelism": "one_worker_process_per_device_one_replica_at_a_time",
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
        "teacher_seed": args.shared_seed,
        "graph_seed": args.shared_seed,
        "noise_seed": args.shared_seed,
        "student_seed_base": args.student_seed_base,
        "student_init_mode": (
            "teacher_plus_noise_normalized"
            if args.init_epsilon is not None
            else "random_gaussian"
        ),
        "student_init_epsilon": args.init_epsilon,
        "legacy_cli_seed": args.seed,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "loss_eval_interval": 1,
        "early_stop": False,
        "save_every_replicas": args.save_every_replicas,
        "devices": devices,
        "torch_threads_per_worker": args.torch_threads,
        "deterministic_requested": args.deterministic,
        "evaluation_metric": "cosine_similarity_in_Y_space",
        "update_scheme": "alternating_W_then_X",
        "step_definition": "one_W_update_plus_one_X_update",
        "onsager_memory_schedule": "half_step",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "dense_mask": True,
        "output_files": [
            "config.yaml",
            "loss_history.csv",
            "replica_summary.csv",
            "progress.yaml",
            "plots/loss_vs_step_linear.png",
            "plots/loss_vs_step_log10.png",
            "plots/cosine_similarity_vs_step.png",
        ],
    }
    write_text_atomic(
        results_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )


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

    lines = [",".join(header)]
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
        lines.append(",".join(row))

    write_text_atomic(history_path, "\n".join(lines) + "\n")


def save_replica_summary(
    results_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    lines = [
        "replica,seed,device,runtime_sec,final_loss,estimated_convergence_step,"
        "cosine_similarity"
    ]
    for record in sorted(records, key=lambda r: r["replica_id"]):
        convergence_value = (
            ""
            if math.isnan(record["estimated_convergence_step"])
            else str(int(record["estimated_convergence_step"]))
        )
        lines.append(
            f"{record['replica']},{record['seed']},{record['device']},"
            f"{record['runtime_sec']:.4f},{record['final_loss']:.10e},"
            f"{convergence_value},{record['final_cosine_similarity']:.10e}"
        )
    write_text_atomic(summary_path, "\n".join(lines) + "\n")


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
        f"N2={args.N2}, M={args.M}, {len(all_losses)} replicas)",
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
        f"N1={args.N1}, N2={args.N2}, M={args.M}, {len(all_losses)} replicas)",
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
        f"N2={args.N2}, M={args.M}, {len(all_cosine_similarities)} replicas)",
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


def build_history_arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(records, key=lambda r: r["replica_id"])
    steps = np.asarray(ordered[0]["steps"], dtype=np.int64)

    for record in ordered[1:]:
        record_steps = np.asarray(record["steps"], dtype=np.int64)
        if not np.array_equal(steps, record_steps):
            raise RuntimeError("Inconsistent step grids across replicas.")

    all_losses = np.asarray(
        [record["loss_history"] for record in ordered],
        dtype=np.float64,
    )
    all_cosine_similarities = np.asarray(
        [record["cosine_similarity_history"] for record in ordered],
        dtype=np.float64,
    )
    return steps, all_losses, all_cosine_similarities


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
        steps, all_losses, all_cosine_similarities = build_history_arrays(records)
        save_loss_history(results_dir, steps, all_losses, all_cosine_similarities)
        save_replica_summary(results_dir, records)

        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        mean_loss = all_losses.mean(axis=0)
        std_loss = all_losses.std(axis=0)
        mean_log_loss = np.log10(np.clip(all_losses, 1e-30, None)).mean(axis=0)
        std_log_loss = np.log10(np.clip(all_losses, 1e-30, None)).std(axis=0)
        mean_cosine_similarity = all_cosine_similarities.mean(axis=0)
        std_cosine_similarity = all_cosine_similarities.std(axis=0)

        plot_linear_loss(plots_dir, steps, all_losses, mean_loss, std_loss, args)
        plot_log_loss(
            plots_dir,
            steps,
            all_losses,
            mean_log_loss,
            std_log_loss,
            args,
        )
        plot_cosine_similarity(
            plots_dir,
            steps,
            all_cosine_similarities,
            mean_cosine_similarity,
            std_cosine_similarity,
            args,
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


def _worker_main(
    device_slot: int,
    device_name: str,
    tasks: list[dict[str, int]],
    worker_config: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    try:
        torch.set_num_threads(int(worker_config["torch_threads"]))

        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        if worker_config["deterministic"]:
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        from terao_gamp_gaussian.Dence_Alternating.random_F_version.core import (
            prepare_global_shared_data,
            prepare_shared_alpha_data,
            train_single_replica,
        )

        global_data = prepare_global_shared_data(
            device=device,
            seed=int(worker_config["shared_seed"]),
            N1=int(worker_config["N1"]),
            N2=int(worker_config["N2"]),
            M=int(worker_config["M"]),
            noise_var=float(worker_config["noise_var"]),
        )
        shared_data = prepare_shared_alpha_data(
            alpha=float(worker_config["alpha"]),
            device=device,
            seed=int(worker_config["shared_seed"]),
            N1=int(worker_config["N1"]),
            N2=int(worker_config["N2"]),
            M=int(worker_config["M"]),
            noise_var=float(worker_config["noise_var"]),
            global_data=global_data,
        )

        for task in tasks:
            t0 = time.time()
            cosine_similarity, final_loss, steps_taken, history = train_single_replica(
                alpha=float(worker_config["alpha"]),
                device=device,
                seed=int(task["seed"]),
                N1=int(worker_config["N1"]),
                N2=int(worker_config["N2"]),
                M=int(worker_config["M"]),
                max_steps=int(worker_config["max_steps"]),
                damping=float(worker_config["damping"]),
                use_step_damping=bool(worker_config["use_step_damping"]),
                damping_beta_scale=float(worker_config["beta_scale"]),
                damping_beta_max=float(worker_config["beta_max"]),
                noise_var=float(worker_config["noise_var"]),
                convergence_threshold=float(worker_config["convergence_threshold"]),
                return_history=True,
                loss_eval_interval=1,
                early_stop=False,
                init_epsilon=worker_config["init_epsilon"],
                shared_data=shared_data,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime = time.time() - t0

            loss_history = np.asarray(history["loss"], dtype=np.float64)
            cosine_history = np.asarray(
                history["cosine_similarity"],
                dtype=np.float64,
            )
            steps = np.asarray(history["steps"], dtype=np.int64)
            convergence_step = estimate_convergence_step(
                loss_history,
                float(worker_config["convergence_threshold"]),
            )

            result_queue.put(
                {
                    "event": "replica_done",
                    "ok": True,
                    "device_slot": device_slot,
                    "device": device_name,
                    "replica_id": int(task["replica_id"]),
                    "replica": int(task["replica_id"]) + 1,
                    "seed": int(task["seed"]),
                    "runtime_sec": runtime,
                    "steps": steps.tolist(),
                    "loss_history": loss_history.tolist(),
                    "cosine_similarity_history": cosine_history.tolist(),
                    "final_loss": float(final_loss),
                    "steps_taken": int(steps_taken),
                    "final_cosine_similarity": float(cosine_similarity),
                    "estimated_convergence_step": float(convergence_step),
                }
            )

        result_queue.put(
            {
                "event": "worker_done",
                "ok": True,
                "device_slot": device_slot,
                "device": device_name,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "event": "worker_error",
                "ok": False,
                "device_slot": device_slot,
                "device": device_name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_parallel_replicas(
    devices: list[str],
    args: argparse.Namespace,
    worker_config: dict[str, Any],
    on_replica_done: Any | None = None,
) -> list[dict[str, Any]]:
    tasks_by_device = assign_replicas_to_devices(
        num_replicas=args.num_replicas,
        devices=devices,
        student_seed_base=args.student_seed_base,
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes: list[mp.Process] = []

    for device_slot, (device_name, tasks) in enumerate(zip(devices, tasks_by_device)):
        process = ctx.Process(
            target=_worker_main,
            args=(device_slot, device_name, tasks, worker_config, result_queue),
        )
        process.start()
        processes.append(process)

    finished_workers = 0
    records: list[dict[str, Any]] = []

    try:
        while finished_workers < len(processes):
            message = result_queue.get()
            event = message.get("event")

            if event == "worker_error":
                raise RuntimeError(
                    f"Worker {message['device_slot']} on {message['device']} failed.\n"
                    f"{message['error']}\n{message['traceback']}"
                )

            if event == "worker_done":
                finished_workers += 1
                continue

            if event == "replica_done":
                records.append(message)
                if on_replica_done is not None:
                    on_replica_done(message, records)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()

    return records


def main() -> int:
    args = parse_args()
    devices = resolve_devices(args)

    if args.save_every_replicas <= 0:
        raise ValueError("--save-every-replicas must be positive.")

    print("=" * 60)
    print("Parallel Loss vs Step for Dense-mask Alternating G-AMP")
    print("Evaluation Metric: Cosine Similarity in observed F-weighted signal space")
    print("=" * 60)
    print(f"Devices: {', '.join(devices)}")
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
    print("Teacher / graph / noise seed:", args.shared_seed)
    print("Student seed rule:", f"{args.student_seed_base} + replica_index")
    if args.init_epsilon is None:
        print("Student init: random Gaussian")
    else:
        print(
            "Student init: teacher + epsilon * N(0, 1), "
            f"epsilon={args.init_epsilon} (then mean-square normalization)"
        )
    print("Shared across all replicas: fixed-alpha teacher / noisy field")
    print("Shared per alpha: graph")
    print(f"Replicas={args.num_replicas}")
    if args.seed != args.shared_seed:
        print(
            f"Legacy CLI seed argument {args.seed} is ignored by the fixed shared seed policy."
        )
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = (
        args.results_root
        if args.results_root is not None
        else Path(__file__).parent / "results"
    )
    results_dir = results_root / (
        f"{timestamp}_loss_vs_step_Dence_Alternating_random_F_alpha{args.alpha}_"
        f"{args.N1}x{args.N2}_M{args.M}"
        f"_initeps{args.init_epsilon if args.init_epsilon is not None else 'random'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    save_config(results_dir, args, devices)

    worker_config = {
        "alpha": args.alpha,
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "max_steps": args.max_steps,
        "damping": args.damping,
        "use_step_damping": args.damping_schedule == "beta",
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "noise_var": args.noise_var,
        "shared_seed": args.shared_seed,
        "convergence_threshold": args.convergence_threshold,
        "init_epsilon": args.init_epsilon,
        "torch_threads": args.torch_threads,
        "deterministic": args.deterministic,
    }

    total_tasks = args.num_replicas
    completed = 0
    start_time = time.time()
    interrupted = False
    records: list[dict[str, Any]] = []

    def on_replica_done(message: dict[str, Any], current_records: list[dict[str, Any]]) -> None:
        nonlocal completed
        completed += 1
        convergence_text = (
            "not reached"
            if math.isnan(message["estimated_convergence_step"])
            else str(int(message["estimated_convergence_step"]))
        )
        print(
            f"Replica {message['replica']}/{args.num_replicas}: "
            f"device={message['device']}, seed={message['seed']}, "
            f"final_loss={message['final_loss']:.10e}, "
            f"final_cosine_similarity={message['final_cosine_similarity']:.10f}, "
            f"estimated_convergence_step={convergence_text}, "
            f"steps_recorded={message['steps_taken']}, "
            f"runtime={message['runtime_sec']:.1f}s "
            f"[{completed}/{total_tasks}]"
        )
        if completed % args.save_every_replicas == 0 or completed == total_tasks:
            save_progress_outputs(
                results_dir=results_dir,
                records=current_records,
                completed=completed,
                total_tasks=total_tasks,
                start_time=start_time,
                status="running",
                args=args,
            )

    try:
        records = run_parallel_replicas(
            devices=devices,
            args=args,
            worker_config=worker_config,
            on_replica_done=on_replica_done,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Terminating workers and saving partial outputs...")
    except RuntimeError as exc:
        interrupted = True
        print(f"\n{exc}")

    total_runtime = time.time() - start_time
    status = "interrupted" if interrupted or completed < total_tasks else "completed"
    save_progress_outputs(
        results_dir=results_dir,
        records=records,
        completed=completed,
        total_tasks=total_tasks,
        start_time=start_time,
        status=status,
        args=args,
    )

    if records:
        final_losses = [record["final_loss"] for record in records]
        cosine_values = [record["final_cosine_similarity"] for record in records]
        print()
        print(f"Mean final loss: {np.mean(final_losses):.10e}")
        print(f"Mean cosine similarity: {np.mean(cosine_values):.10f}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")

    return 130 if status == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
