#!/usr/bin/env python
"""
GPU-per-replica parallel runner for alternating mini-batch SGD.

This runner keeps the numerical update path in
``terao_gd.gd_cosine_minibatch.gd`` and only parallelizes replica execution.
Each worker owns one device, runs at most one replica at a time, and sends
scalar results back to the parent process. Outputs are aggregated into one
result directory.
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

# Add project root to path.
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))


# Keep defaults aligned with terao_gd/gd_cosine_minibatch/gd.py.
DEFAULT_N1 = 10000
DEFAULT_N2 = 10000
DEFAULT_M = 100
DEFAULT_ALPHA_START = 0.0
DEFAULT_ALPHA_STOP = 0.0
DEFAULT_ALPHA_STEP = 0.2
DEFAULT_MAX_STEPS = 500000
DEFAULT_BATCH_SIZE = 2000
DEFAULT_LR_SCHEDULE = [
    (0.0, 0.5, 1e-3),
    (0.5, 1.5, 1.5e-3),
    (1.5, 3.0, 1.5e-3),
    (3.0, float("inf"), 1.5e-3),
]
DEFAULT_NOISE_VAR = 1.0
DEFAULT_SHARED_SEED = 1
DEFAULT_STUDENT_SEED_BASE = 100
DEFAULT_NUM_REPLICAS = 1
DEFAULT_CONVERGENCE_THRESHOLD = 1e-5
DEFAULT_LOSS_EVAL_INTERVAL = 100
DEFAULT_SAVE_EVERY_REPLICAS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gd_cosine_minibatch replicas in parallel, one GPU per replica."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Set N1=N2=N. Overrides --N1 and --N2 when provided.",
    )
    parser.add_argument("--N1", type=int, default=DEFAULT_N1)
    parser.add_argument("--N2", type=int, default=DEFAULT_N2)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--alpha-start", type=float, default=DEFAULT_ALPHA_START)
    parser.add_argument("--alpha-stop", type=float, default=DEFAULT_ALPHA_STOP)
    parser.add_argument("--alpha-step", type=float, default=DEFAULT_ALPHA_STEP)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Use a fixed learning rate. Defaults to the built-in alpha schedule.",
    )
    parser.add_argument("--noise-var", type=float, default=DEFAULT_NOISE_VAR)
    parser.add_argument("--shared-seed", type=int, default=DEFAULT_SHARED_SEED)
    parser.add_argument(
        "--student-seed-base", type=int, default=DEFAULT_STUDENT_SEED_BASE
    )
    parser.add_argument("--num-replicas", type=int, default=DEFAULT_NUM_REPLICAS)
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
    )
    parser.add_argument(
        "--loss-eval-interval",
        type=int,
        default=DEFAULT_LOSS_EVAL_INTERVAL,
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device ids, e.g. 0,1,2. "
            "Defaults to all visible CUDA devices."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run on CPU when CUDA is unavailable. Intended for smoke tests.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Number of CPU workers when --allow-cpu is used without CUDA.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
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
        help="Write partial CSV/NPZ/progress outputs after this many replicas.",
    )
    return parser.parse_args()


def resolve_lr(alpha: float, fixed_lr: float | None) -> float:
    if fixed_lr is not None:
        return float(fixed_lr)
    for alpha_min, alpha_max, lr_value in DEFAULT_LR_SCHEDULE:
        if alpha_min <= alpha < alpha_max:
            return float(lr_value)
    return float(DEFAULT_LR_SCHEDULE[-1][2])


def compute_effective_epochs(
    num_observed: int,
    batch_size: int,
    max_steps: int,
) -> float:
    if num_observed <= 0 or batch_size <= 0:
        return 0.0
    return float(max_steps * batch_size / num_observed)


def build_alpha_values(
    alpha_start: float, alpha_stop: float, alpha_step: float
) -> list[float]:
    if alpha_step <= 0:
        raise ValueError("--alpha-step must be positive.")
    values = np.arange(alpha_start, alpha_stop + alpha_step / 2.0, alpha_step)
    return [float(v) for v in values]


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.N is not None:
        args.N1 = args.N
        args.N2 = args.N

    if args.devices:
        raw_devices = [part.strip() for part in args.devices.split(",") if part.strip()]
        if not raw_devices:
            raise ValueError("--devices was provided but no devices were parsed.")
        devices = [
            dev if dev.startswith("cuda:") or dev == "cpu" else f"cuda:{dev}"
            for dev in raw_devices
        ]
        if any(dev != "cpu" for dev in devices) and not torch.cuda.is_available():
            raise RuntimeError("CUDA devices were requested, but CUDA is unavailable.")
        return devices

    if torch.cuda.is_available():
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    if args.allow_cpu:
        return ["cpu" for _ in range(max(1, args.cpu_workers))]

    raise RuntimeError(
        "CUDA is unavailable. Run on a GPU node or pass --allow-cpu for a smoke test."
    )


def assign_tasks_to_devices(
    alphas: list[float],
    num_replicas: int,
    devices: list[str],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    tasks_by_device: list[list[dict[str, Any]]] = [[] for _ in devices]
    for alpha in alphas:
        lr_alpha = resolve_lr(alpha, args.lr)
        for replica_id in range(num_replicas):
            device_slot = replica_id % len(devices)
            tasks_by_device[device_slot].append(
                {
                    "alpha": alpha,
                    "replica_id": replica_id,
                    "seed": args.student_seed_base + replica_id,
                    "lr": lr_alpha,
                }
            )
    return tasks_by_device


def assign_alpha_tasks_to_devices(
    alpha: float,
    num_replicas: int,
    devices: list[str],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    return assign_tasks_to_devices(
        alphas=[alpha],
        num_replicas=num_replicas,
        devices=devices,
        args=args,
    )


def _worker_main(
    device_slot: int,
    device_name: str,
    tasks: list[dict[str, Any]],
    worker_config: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    try:
        torch.set_num_threads(int(worker_config["torch_threads"]))

        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        if worker_config["deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        from terao_gd.gd_cosine_minibatch.gd import (
            prepare_global_shared_data,
            prepare_shared_alpha_data,
            train_single_replica,
        )

        global_data = prepare_global_shared_data(
            device=device,
            seed=worker_config["shared_seed"],
            N1=worker_config["N1"],
            N2=worker_config["N2"],
            M=worker_config["M"],
            noise_var=worker_config["noise_var"],
        )

        current_alpha: float | None = None
        shared_data: dict[str, Any] | None = None

        for task in tasks:
            alpha = float(task["alpha"])
            if current_alpha != alpha:
                shared_data = prepare_shared_alpha_data(
                    alpha=alpha,
                    device=device,
                    seed=worker_config["shared_seed"],
                    N1=worker_config["N1"],
                    N2=worker_config["N2"],
                    M=worker_config["M"],
                    noise_var=worker_config["noise_var"],
                    global_data=global_data,
                )
                current_alpha = alpha

            if shared_data is None:
                raise RuntimeError("Internal error: shared alpha data is missing.")

            num_observed = int(shared_data["num_observed"])
            effective_epochs = compute_effective_epochs(
                num_observed=num_observed,
                batch_size=worker_config["batch_size"],
                max_steps=worker_config["max_steps"],
            )

            t0 = time.time()
            cosine_similarity, final_loss, steps_taken = train_single_replica(
                alpha=alpha,
                device=device,
                seed=int(task["seed"]),
                N1=worker_config["N1"],
                N2=worker_config["N2"],
                M=worker_config["M"],
                max_steps=worker_config["max_steps"],
                lr=float(task["lr"]),
                batch_size=worker_config["batch_size"],
                noise_var=worker_config["noise_var"],
                convergence_threshold=worker_config["convergence_threshold"],
                loss_eval_interval=worker_config["loss_eval_interval"],
                shared_data=shared_data,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime = time.time() - t0

            result_queue.put(
                {
                    "event": "replica_done",
                    "ok": True,
                    "device_slot": device_slot,
                    "device": device_name,
                    "alpha": alpha,
                    "replica_id": int(task["replica_id"]),
                    "replica": int(task["replica_id"]) + 1,
                    "seed": int(task["seed"]),
                    "lr": float(task["lr"]),
                    "num_observed": num_observed,
                    "effective_epochs": effective_epochs,
                    "runtime_sec": runtime,
                    "final_loss": float(final_loss),
                    "steps_taken": int(steps_taken),
                    "cosine_similarity": float(cosine_similarity),
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


def write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def aggregate_results(
    records: list[dict[str, Any]],
    alphas: list[float],
    num_replicas: int,
) -> dict[float, dict[str, Any]]:
    results: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        alpha_records = [record for record in records if record["alpha"] == alpha]
        if not alpha_records:
            continue
        alpha_records.sort(key=lambda record: record["replica_id"])
        cosine_values = [record["cosine_similarity"] for record in alpha_records]
        loss_values = [record["final_loss"] for record in alpha_records]
        steps_values = [record["steps_taken"] for record in alpha_records]
        first = alpha_records[0]
        results[alpha] = {
            "lr": first["lr"],
            "num_observed": first["num_observed"],
            "effective_epochs": first["effective_epochs"],
            "completed_replicas": len(alpha_records),
            "cosine_similarity_mean": float(np.mean(cosine_values)),
            "cosine_similarity_std": float(np.std(cosine_values)),
            "cosine_similarity_values": cosine_values,
            "loss_mean": float(np.mean(loss_values)),
            "loss_std": float(np.std(loss_values)),
            "loss_values": loss_values,
            "steps_mean": float(np.mean(steps_values)),
            "steps_values": steps_values,
            "num_replicas_requested": num_replicas,
        }
    return results


def save_metrics_csv(
    results_dir: Path,
    results: dict[float, dict[str, Any]],
    alphas: list[float],
    num_replicas: int,
) -> None:
    csv_path = results_dir / "metrics.csv"
    lines = []
    header = (
        "alpha,lr,num_observed,effective_epochs,completed_replicas,"
        "cosine_similarity_mean,cosine_similarity_std,"
        "Loss_mean,Loss_std,Steps_mean"
    )
    for replica_idx in range(num_replicas):
        header += f",cosine_similarity_replica_{replica_idx},loss_replica_{replica_idx}"
    lines.append(header)

    for alpha in alphas:
        if alpha not in results:
            continue
        result = results[alpha]
        cosine_values = list(result["cosine_similarity_values"])
        loss_values = list(result["loss_values"])
        line = (
            f"{alpha},{result['lr']},{result['num_observed']},"
            f"{result['effective_epochs']},{result['completed_replicas']},"
            f"{result['cosine_similarity_mean']},{result['cosine_similarity_std']},"
            f"{result['loss_mean']},{result['loss_std']},{result['steps_mean']}"
        )
        for replica_idx in range(num_replicas):
            if replica_idx < len(cosine_values):
                line += f",{cosine_values[replica_idx]},{loss_values[replica_idx]}"
            else:
                line += ",,"
        lines.append(line)

    write_text_atomic(csv_path, "\n".join(lines) + "\n")


def save_replica_summary(
    results_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    summary_path = results_dir / "replica_summary.csv"
    lines = [
        "alpha,lr,num_observed,effective_epochs,replica,seed,device,"
        "runtime_sec,final_loss,steps_taken,cosine_similarity"
    ]
    for record in sorted(records, key=lambda r: (r["alpha"], r["replica_id"])):
        lines.append(
            f"{record['alpha']},{record['lr']},{record['num_observed']},"
            f"{record['effective_epochs']},{record['replica']},{record['seed']},"
            f"{record['device']},{record['runtime_sec']:.4f},"
            f"{record['final_loss']:.10e},{record['steps_taken']},"
            f"{record['cosine_similarity']:.10e}"
        )
    write_text_atomic(summary_path, "\n".join(lines) + "\n")


def save_results_npz(
    results_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    npz_path = results_dir / "results.npz"
    ordered = sorted(records, key=lambda r: (r["alpha"], r["replica_id"]))
    if not ordered:
        np.savez(npz_path, empty=np.array([], dtype=np.float32))
        return

    np.savez(
        npz_path,
        alpha=np.array([record["alpha"] for record in ordered], dtype=np.float64),
        replica=np.array([record["replica"] for record in ordered], dtype=np.int64),
        replica_id=np.array(
            [record["replica_id"] for record in ordered], dtype=np.int64
        ),
        seed=np.array([record["seed"] for record in ordered], dtype=np.int64),
        lr=np.array([record["lr"] for record in ordered], dtype=np.float64),
        num_observed=np.array(
            [record["num_observed"] for record in ordered], dtype=np.int64
        ),
        effective_epochs=np.array(
            [record["effective_epochs"] for record in ordered], dtype=np.float64
        ),
        runtime_sec=np.array(
            [record["runtime_sec"] for record in ordered], dtype=np.float64
        ),
        final_loss=np.array(
            [record["final_loss"] for record in ordered], dtype=np.float64
        ),
        steps_taken=np.array(
            [record["steps_taken"] for record in ordered], dtype=np.int64
        ),
        cosine_similarity=np.array(
            [record["cosine_similarity"] for record in ordered], dtype=np.float64
        ),
        device=np.array([record["device"] for record in ordered]),
    )


def save_progress_outputs(
    results_dir: Path,
    records: list[dict[str, Any]],
    alphas: list[float],
    num_replicas: int,
    completed: int,
    total_tasks: int,
    start_time: float,
    status: str,
) -> dict[float, dict[str, Any]]:
    results = aggregate_results(records, alphas, num_replicas)
    save_metrics_csv(results_dir, results, alphas, num_replicas)
    save_replica_summary(results_dir, records)
    save_results_npz(results_dir, records)
    progress = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_tasks": completed,
        "total_tasks": total_tasks,
        "elapsed_sec": time.time() - start_time,
    }
    write_text_atomic(
        results_dir / "progress.yaml",
        yaml.dump(progress, default_flow_style=False),
    )
    return results


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    devices: list[str],
    alphas: list[float],
) -> None:
    config = {
        "algorithm": "sgd_cosine_minibatch_parallel",
        "parallelism": "one_worker_process_per_device_one_replica_at_a_time",
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "alpha_start": args.alpha_start,
        "alpha_stop": args.alpha_stop,
        "alpha_step": args.alpha_step,
        "alphas": alphas,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_schedule": [
            {"alpha_min": alpha_min, "alpha_max": alpha_max, "lr": lr_value}
            for alpha_min, alpha_max, lr_value in DEFAULT_LR_SCHEDULE
        ],
        "effective_epoch_formula": "max_steps * batch_size / num_observed",
        "noise_var": args.noise_var,
        "teacher_seed": args.shared_seed,
        "graph_seed": args.shared_seed,
        "noise_seed": args.shared_seed,
        "student_seed_base": args.student_seed_base,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "loss_eval_interval": args.loss_eval_interval,
        "save_every_replicas": args.save_every_replicas,
        "devices": devices,
        "torch_threads_per_worker": args.torch_threads,
        "deterministic_requested": args.deterministic,
        "output_files": [
            "config.yaml",
            "metrics.csv",
            "replica_summary.csv",
            "results.npz",
            "progress.yaml",
            "plots/cosine_similarity_vs_alpha.png",
        ],
        "evaluation_metric": "cosine_similarity_in_Y_space",
        "sampling": "with_replacement",
        "shared_per_alpha_graph_noise": True,
        "shared_teacher_noise_global": True,
        "replica_variation": "student_initialization_only",
        "exact_seed_policy": "same as gd_cosine_minibatch",
    }
    write_text_atomic(
        results_dir / "config.yaml",
        yaml.dump(config, default_flow_style=False),
    )


def plot_results(
    results_dir: Path,
    results: dict[float, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    completed_alphas = sorted(results.keys())
    if not completed_alphas:
        return

    means = [results[alpha]["cosine_similarity_mean"] for alpha in completed_alphas]
    stds = [results[alpha]["cosine_similarity_std"] for alpha in completed_alphas]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        completed_alphas,
        means,
        yerr=stds,
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
        f"Phase Transition (Parallel Alternating Mini-Batch SGD)\n"
        f"({args.N1}x{args.N2}, M={args.M}, {args.max_steps} steps, "
        f"{args.num_replicas} replicas)",
        fontsize=16,
    )
    ax.set_xlim(args.alpha_start - 0.1, args.alpha_stop + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_similarity_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(results: dict[float, dict[str, Any]], total_time: float) -> None:
    print("\n" + "=" * 60)
    print("Results (mean +- std)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'CosSim':^20} | {'Loss':^20} | {'Steps':>8}")
    print("-" * 60)
    for alpha in sorted(results.keys()):
        result = results[alpha]
        print(
            f"{alpha:6.2f} | "
            f"{result['cosine_similarity_mean']:8.4f} +- "
            f"{result['cosine_similarity_std']:<8.4f} | "
            f"{result['loss_mean']:8.2e} +- {result['loss_std']:<8.2e} | "
            f"{result['steps_mean']:8.0f}"
        )
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 3600.0:.2f}h)")
    print("=" * 60)


def run_alpha_batch(
    alpha: float,
    devices: list[str],
    args: argparse.Namespace,
    worker_config: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks_by_device = assign_alpha_tasks_to_devices(
        alpha=alpha,
        num_replicas=args.num_replicas,
        devices=devices,
        args=args,
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
    alpha_records: list[dict[str, Any]] = []

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
                alpha_records.append(message)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()

    alpha_records.sort(key=lambda record: record["replica_id"])
    if len(alpha_records) != args.num_replicas:
        raise RuntimeError(
            f"alpha={alpha} completed with {len(alpha_records)} replicas, "
            f"expected {args.num_replicas}."
        )

    return alpha_records


def main() -> int:
    args = parse_args()
    devices = resolve_devices(args)
    alphas = build_alpha_values(args.alpha_start, args.alpha_stop, args.alpha_step)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = (
        args.results_root
        if args.results_root is not None
        else Path(__file__).parent / "results"
    )
    results_dir = results_root / (
        f"{timestamp}_sgd_cosine_parallel_{args.N1}x{args.M}_"
        f"alpha{args.alpha_start}-{args.alpha_stop}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Parallel Alternating Mini-Batch SGD - Matrix Factorization")
    print("Cosine Similarity Evaluation")
    print("=" * 72)
    print(f"Matrix: {args.N1}x{args.N2}, M={args.M}")
    print(f"Alpha: {args.alpha_start} ~ {args.alpha_stop} (step {args.alpha_step})")
    print(f"Steps: {args.max_steps}, Batch={args.batch_size}")
    print(f"LR: fixed {args.lr:.3e}" if args.lr is not None else f"LR schedule: {DEFAULT_LR_SCHEDULE}")
    print(f"Replicas per alpha: {args.num_replicas}")
    print(f"Devices: {', '.join(devices)}")
    print("Execution rule: one worker process per device, one active replica per device")
    print("Teacher / graph / noise seed:", args.shared_seed)
    print("Student seed rule:", f"{args.student_seed_base} + replica_id")
    print(f"Results directory: {results_dir}")
    print()

    save_config(results_dir, args, devices, alphas)

    total_tasks = len(alphas) * args.num_replicas
    if total_tasks == 0:
        raise RuntimeError("No replica tasks were generated.")

    worker_config = {
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "noise_var": args.noise_var,
        "shared_seed": args.shared_seed,
        "convergence_threshold": args.convergence_threshold,
        "loss_eval_interval": args.loss_eval_interval,
        "torch_threads": args.torch_threads,
        "deterministic": args.deterministic,
    }

    start_time = time.time()
    completed = 0
    records: list[dict[str, Any]] = []
    interrupted = False

    try:
        for alpha in alphas:
            alpha_records = run_alpha_batch(
                alpha=alpha,
                devices=devices,
                args=args,
                worker_config=worker_config,
            )
            records.extend(alpha_records)

            for message in alpha_records:
                completed += 1
                elapsed = time.time() - start_time
                eta = elapsed / completed * (total_tasks - completed) if completed else 0.0
                print(
                    f"[{completed}/{total_tasks}] "
                    f"device={message['device']} "
                    f"alpha={message['alpha']:.2f}, "
                    f"replica {message['replica']}/{args.num_replicas}: "
                    f"CosSim={message['cosine_similarity']:.4f}, "
                    f"Loss={message['final_loss']:.2e}, "
                    f"Steps={message['steps_taken']} "
                    f"({message['runtime_sec']:.1f}s) ETA={eta / 3600.0:.1f}h"
                )

            save_progress_outputs(
                results_dir=results_dir,
                records=records,
                alphas=alphas,
                num_replicas=args.num_replicas,
                completed=completed,
                total_tasks=total_tasks,
                start_time=start_time,
                status="running",
            )
            print(f"alpha={alpha:.2f} completed with {len(alpha_records)}/{args.num_replicas} replicas")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Terminating workers and saving partial outputs...")
    except RuntimeError as exc:
        interrupted = True
        print(f"\n{exc}")

    total_time = time.time() - start_time
    status = "interrupted" if interrupted or completed < total_tasks else "completed"
    results = save_progress_outputs(
        results_dir=results_dir,
        records=records,
        alphas=alphas,
        num_replicas=args.num_replicas,
        completed=completed,
        total_tasks=total_tasks,
        start_time=start_time,
        status=status,
    )
    plot_results(results_dir, results, args)
    print_summary(results, total_time)
    print(f"\nMetrics saved: {results_dir / 'metrics.csv'}")
    print(f"Replica summary saved: {results_dir / 'replica_summary.csv'}")
    print(f"Single-file results saved: {results_dir / 'results.npz'}")
    print(f"Results saved to: {results_dir}")

    return 130 if status == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
