#!/usr/bin/env python
"""
Parallel noise sweep for sequential random-F Edge_Alternating G-AMP.

This fixes alpha and sweeps noise_var, writing a CSV summary and a plot with
noise on the horizontal axis and cosine_Y similarity on the vertical axis.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))


from terao_gamp_gaussian.Edge_Alternating.random_F_sequentially.core import (  # noqa: E402
    DEFAULT_EDGE_CHUNK_SIZE,
    ORDER_PARAMETER_KEYS,
)


DEFAULT_N1 = 5000
DEFAULT_N2 = 5000
DEFAULT_M = 400
DEFAULT_ALPHA = 5
DEFAULT_NOISE_START = 2.5
DEFAULT_NOISE_STOP = 4
DEFAULT_NOISE_STEP = 0.5
DEFAULT_MAX_STEPS = 100
DEFAULT_DAMPING = 0.0
DEFAULT_DAMPING_SCHEDULE = "constant"
DEFAULT_BETA_SCALE = 1e-3
DEFAULT_BETA_MAX = DEFAULT_DAMPING
DEFAULT_SHARED_SEED = 1
DEFAULT_STUDENT_SEED_BASE = 100
DEFAULT_NUM_REPLICAS = 1
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_INIT_EPSILON = 0.01
DEFAULT_TORCH_THREADS = 1
DEFAULT_SAVE_EVERY_REPLICAS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fix alpha and sweep noise_var for sequential random-F "
            "Edge_Alternating G-AMP."
        )
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
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--noise-start", type=float, default=DEFAULT_NOISE_START)
    parser.add_argument("--noise-stop", type=float, default=DEFAULT_NOISE_STOP)
    parser.add_argument("--noise-step", type=float, default=DEFAULT_NOISE_STEP)
    parser.add_argument(
        "--noise-values",
        type=str,
        default=None,
        help="Comma-separated noise_var values. Overrides start/stop/step.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic x-axis. All completed noise values must be positive.",
    )
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
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=DEFAULT_EDGE_CHUNK_SIZE,
        help=(
            "Number of observed edges processed at once. Lower this to reduce "
            "peak memory at the cost of runtime."
        ),
    )
    parser.add_argument("--shared-seed", type=int, default=DEFAULT_SHARED_SEED)
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
        default=DEFAULT_INIT_EPSILON,
        help=(
            "Use informative student initialization: epsilon * teacher + "
            "sqrt(epsilon - epsilon^2) * N(0, 1). Must satisfy 0 <= epsilon <= 1."
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


def build_noise_values(args: argparse.Namespace) -> list[float]:
    if args.noise_values:
        values = [
            float(part.strip())
            for part in args.noise_values.split(",")
            if part.strip()
        ]
        if not values:
            raise ValueError("--noise-values was provided but no values were parsed.")
        return values

    if args.noise_step <= 0:
        raise ValueError("--noise-step must be positive.")
    if args.noise_stop < args.noise_start:
        raise ValueError("--noise-stop must be greater than or equal to --noise-start.")

    values = np.arange(
        args.noise_start,
        args.noise_stop + args.noise_step / 2.0,
        args.noise_step,
    )
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


def assign_tasks_to_devices(
    noise_values: list[float],
    num_replicas: int,
    devices: list[str],
    student_seed_base: int,
) -> list[list[dict[str, Any]]]:
    tasks_by_device: list[list[dict[str, Any]]] = [[] for _ in devices]
    for noise_var in noise_values:
        for replica_id in range(num_replicas):
            device_slot = replica_id % len(devices)
            tasks_by_device[device_slot].append(
                {
                    "noise_var": noise_var,
                    "replica_id": replica_id,
                    "seed": student_seed_base + replica_id,
                }
            )
    return tasks_by_device


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
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        from terao_gamp_gaussian.Edge_Alternating.random_F_sequentially.core import (
            prepare_global_shared_data,
            prepare_shared_alpha_data,
            train_single_replica,
        )

        current_noise_var: float | None = None
        shared_data: dict[str, Any] | None = None

        for task in tasks:
            noise_var = float(task["noise_var"])
            if current_noise_var != noise_var:
                global_data = prepare_global_shared_data(
                    device=device,
                    seed=int(worker_config["shared_seed"]),
                    N1=int(worker_config["N1"]),
                    N2=int(worker_config["N2"]),
                    M=int(worker_config["M"]),
                    noise_var=noise_var,
                    lam=float(worker_config["lam"]),
                )
                shared_data = prepare_shared_alpha_data(
                    alpha=float(worker_config["alpha"]),
                    device=device,
                    seed=int(worker_config["shared_seed"]),
                    N1=int(worker_config["N1"]),
                    N2=int(worker_config["N2"]),
                    M=int(worker_config["M"]),
                    noise_var=noise_var,
                    lam=float(worker_config["lam"]),
                    edge_chunk_size=int(worker_config["edge_chunk_size"]),
                    global_data=global_data,
                )
                current_noise_var = noise_var

            if shared_data is None:
                raise RuntimeError("Internal error: shared noise data is missing.")

            num_observed = int(shared_data["E"])

            t0 = time.time()
            cosine_Y, final_loss, steps_taken, history = train_single_replica(
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
                noise_var=noise_var,
                lam=float(worker_config["lam"]),
                convergence_threshold=float(worker_config["convergence_threshold"]),
                return_history=True,
                loss_eval_interval=1,
                early_stop=False,
                init_epsilon=worker_config["init_epsilon"],
                edge_chunk_size=int(worker_config["edge_chunk_size"]),
                shared_data=shared_data,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if device.type == "mps":
                torch.mps.synchronize()
            runtime = time.time() - t0

            loss_history = np.asarray(history["loss"], dtype=np.float64)
            cosine_Y_history = np.asarray(history["cosine_Y"], dtype=np.float64)
            steps = np.asarray(history["steps"], dtype=np.int64)
            final_order_parameters = {
                key: float(np.asarray(history[key], dtype=np.float64)[-1])
                for key in ORDER_PARAMETER_KEYS
            }
            if cosine_Y_history.size:
                max_cosine_Y_idx = int(np.argmax(cosine_Y_history))
                min_loss_idx = int(np.argmin(loss_history))
                initial_cosine_Y = float(cosine_Y_history[0])
                initial_loss = float(loss_history[0])
                step1_matches = np.where(steps == 1)[0]
                step1_idx = int(step1_matches[0]) if step1_matches.size else min(1, cosine_Y_history.size - 1)
                step1_cosine_Y = float(cosine_Y_history[step1_idx])
                step1_loss = float(loss_history[step1_idx])
                max_cosine_Y = float(cosine_Y_history[max_cosine_Y_idx])
                max_cosine_Y_step = int(steps[max_cosine_Y_idx])
                min_loss = float(loss_history[min_loss_idx])
                min_loss_step = int(steps[min_loss_idx])
            else:
                initial_cosine_Y = float("nan")
                initial_loss = float("nan")
                step1_cosine_Y = float("nan")
                step1_loss = float("nan")
                max_cosine_Y = float("nan")
                max_cosine_Y_step = -1
                min_loss = float("nan")
                min_loss_step = -1

            message = {
                "event": "replica_done",
                "ok": True,
                "device_slot": device_slot,
                "device": device_name,
                "alpha": float(worker_config["alpha"]),
                "noise_var": noise_var,
                "replica_id": int(task["replica_id"]),
                "replica": int(task["replica_id"]) + 1,
                "seed": int(task["seed"]),
                "num_observed": num_observed,
                "runtime_sec": runtime,
                "final_loss": float(final_loss),
                "steps_taken": int(steps_taken),
                "initial_cosine_Y": initial_cosine_Y,
                "step1_cosine_Y": step1_cosine_Y,
                "max_cosine_Y": max_cosine_Y,
                "max_cosine_Y_step": max_cosine_Y_step,
                "initial_loss": initial_loss,
                "step1_loss": step1_loss,
                "min_loss": min_loss,
                "min_loss_step": min_loss_step,
            }
            message.update(final_order_parameters)
            message["cosine_Y"] = float(cosine_Y)
            result_queue.put(message)

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
    noise_values: list[float],
    devices: list[str],
    args: argparse.Namespace,
    worker_config: dict[str, Any],
    on_replica_done: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    tasks_by_device = assign_tasks_to_devices(
        noise_values=noise_values,
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


def write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def aggregate_results(
    records: list[dict[str, Any]],
    noise_values: list[float],
    num_replicas: int,
) -> dict[float, dict[str, Any]]:
    results: dict[float, dict[str, Any]] = {}
    for noise_var in noise_values:
        noise_records = [
            record for record in records if record["noise_var"] == noise_var
        ]
        if not noise_records:
            continue
        noise_records.sort(key=lambda record: record["replica_id"])
        cosine_Y_values = [record["cosine_Y"] for record in noise_records]
        loss_values = [record["final_loss"] for record in noise_records]
        steps_values = [record["steps_taken"] for record in noise_records]
        initial_cosine_Y_values = [
            record["initial_cosine_Y"] for record in noise_records
        ]
        step1_cosine_Y_values = [
            record["step1_cosine_Y"] for record in noise_records
        ]
        max_cosine_Y_values = [
            record["max_cosine_Y"] for record in noise_records
        ]
        max_cosine_Y_step_values = [
            record["max_cosine_Y_step"] for record in noise_records
        ]
        initial_loss_values = [record["initial_loss"] for record in noise_records]
        step1_loss_values = [record["step1_loss"] for record in noise_records]
        min_loss_values = [record["min_loss"] for record in noise_records]
        min_loss_step_values = [record["min_loss_step"] for record in noise_records]
        first = noise_records[0]
        result = {
            "alpha": float(first["alpha"]),
            "num_observed": int(first["num_observed"]),
            "completed_replicas": len(noise_records),
            "initial_cosine_Y_mean": float(np.mean(initial_cosine_Y_values)),
            "initial_cosine_Y_std": float(np.std(initial_cosine_Y_values)),
            "initial_cosine_Y_values": initial_cosine_Y_values,
            "step1_cosine_Y_mean": float(np.mean(step1_cosine_Y_values)),
            "step1_cosine_Y_std": float(np.std(step1_cosine_Y_values)),
            "step1_cosine_Y_values": step1_cosine_Y_values,
            "max_cosine_Y_mean": float(np.mean(max_cosine_Y_values)),
            "max_cosine_Y_std": float(np.std(max_cosine_Y_values)),
            "max_cosine_Y_values": max_cosine_Y_values,
            "max_cosine_Y_step_mean": float(np.mean(max_cosine_Y_step_values)),
            "max_cosine_Y_step_values": max_cosine_Y_step_values,
            "loss_mean": float(np.mean(loss_values)),
            "loss_std": float(np.std(loss_values)),
            "loss_values": loss_values,
            "initial_loss_mean": float(np.mean(initial_loss_values)),
            "initial_loss_std": float(np.std(initial_loss_values)),
            "initial_loss_values": initial_loss_values,
            "step1_loss_mean": float(np.mean(step1_loss_values)),
            "step1_loss_std": float(np.std(step1_loss_values)),
            "step1_loss_values": step1_loss_values,
            "min_loss_mean": float(np.mean(min_loss_values)),
            "min_loss_std": float(np.std(min_loss_values)),
            "min_loss_values": min_loss_values,
            "min_loss_step_mean": float(np.mean(min_loss_step_values)),
            "min_loss_step_values": min_loss_step_values,
            "steps_mean": float(np.mean(steps_values)),
            "steps_values": steps_values,
            "num_replicas_requested": num_replicas,
        }
        for key in ORDER_PARAMETER_KEYS:
            values = [record[key] for record in noise_records]
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
            result[f"{key}_values"] = values
        results[noise_var] = result
    return results


def save_metrics_csv(
    results_dir: Path,
    results: dict[float, dict[str, Any]],
    noise_values: list[float],
    num_replicas: int,
) -> None:
    csv_path = results_dir / "metrics.csv"
    lines = []
    header = (
        "noise_var,alpha,num_observed,completed_replicas,cosine_Y_mean,"
        "cosine_Y_std,initial_cosine_Y_mean,"
        "initial_cosine_Y_std,step1_cosine_Y_mean,"
        "step1_cosine_Y_std,max_cosine_Y_mean,"
        "max_cosine_Y_std,max_cosine_Y_step_mean,loss_mean,loss_std,"
        "initial_loss_mean,initial_loss_std,step1_loss_mean,step1_loss_std,"
        "min_loss_mean,min_loss_std,min_loss_step_mean,steps_mean"
    )
    for key in ORDER_PARAMETER_KEYS:
        if key != "cosine_Y":
            header += f",{key}_mean,{key}_std"
    for replica_idx in range(num_replicas):
        header += (
            f",cosine_Y_replica_{replica_idx},"
            f"initial_cosine_Y_replica_{replica_idx},"
            f"step1_cosine_Y_replica_{replica_idx},"
            f"max_cosine_Y_replica_{replica_idx},"
            f"max_cosine_Y_step_replica_{replica_idx},"
            f"loss_replica_{replica_idx},"
            f"initial_loss_replica_{replica_idx},"
            f"step1_loss_replica_{replica_idx},"
            f"min_loss_replica_{replica_idx},"
            f"min_loss_step_replica_{replica_idx},"
            f"steps_replica_{replica_idx}"
        )
    lines.append(header)

    for noise_var in noise_values:
        if noise_var not in results:
            continue
        result = results[noise_var]
        cosine_Y_values = list(result["cosine_Y_values"])
        initial_cosine_Y_values = list(result["initial_cosine_Y_values"])
        step1_cosine_Y_values = list(result["step1_cosine_Y_values"])
        max_cosine_Y_values = list(result["max_cosine_Y_values"])
        max_cosine_Y_step_values = list(result["max_cosine_Y_step_values"])
        loss_values = list(result["loss_values"])
        initial_loss_values = list(result["initial_loss_values"])
        step1_loss_values = list(result["step1_loss_values"])
        min_loss_values = list(result["min_loss_values"])
        min_loss_step_values = list(result["min_loss_step_values"])
        steps_values = list(result["steps_values"])
        line = (
            f"{noise_var},{result['alpha']},{result['num_observed']},"
            f"{result['completed_replicas']},"
            f"{result['cosine_Y_mean']},{result['cosine_Y_std']},"
            f"{result['initial_cosine_Y_mean']},"
            f"{result['initial_cosine_Y_std']},"
            f"{result['step1_cosine_Y_mean']},"
            f"{result['step1_cosine_Y_std']},"
            f"{result['max_cosine_Y_mean']},"
            f"{result['max_cosine_Y_std']},"
            f"{result['max_cosine_Y_step_mean']},"
            f"{result['loss_mean']},{result['loss_std']},"
            f"{result['initial_loss_mean']},{result['initial_loss_std']},"
            f"{result['step1_loss_mean']},{result['step1_loss_std']},"
            f"{result['min_loss_mean']},{result['min_loss_std']},"
            f"{result['min_loss_step_mean']},{result['steps_mean']}"
        )
        for key in ORDER_PARAMETER_KEYS:
            if key != "cosine_Y":
                line += f",{result[f'{key}_mean']},{result[f'{key}_std']}"
        for replica_idx in range(num_replicas):
            if replica_idx < len(cosine_Y_values):
                line += (
                    f",{cosine_Y_values[replica_idx]},"
                    f"{initial_cosine_Y_values[replica_idx]},"
                    f"{step1_cosine_Y_values[replica_idx]},"
                    f"{max_cosine_Y_values[replica_idx]},"
                    f"{max_cosine_Y_step_values[replica_idx]},"
                    f"{loss_values[replica_idx]},"
                    f"{initial_loss_values[replica_idx]},"
                    f"{step1_loss_values[replica_idx]},"
                    f"{min_loss_values[replica_idx]},"
                    f"{min_loss_step_values[replica_idx]},"
                    f"{steps_values[replica_idx]}"
                )
            else:
                line += "," * 11
        lines.append(line)

    write_text_atomic(csv_path, "\n".join(lines) + "\n")


def save_replica_summary(results_dir: Path, records: list[dict[str, Any]]) -> None:
    summary_path = results_dir / "replica_summary.csv"
    lines = [
        "noise_var,alpha,num_observed,replica,seed,device,runtime_sec,"
        "final_loss,initial_loss,step1_loss,min_loss,min_loss_step,"
        "steps_taken,cosine_Y,initial_cosine_Y,"
        "step1_cosine_Y,max_cosine_Y,max_cosine_Y_step,"
        + ",".join(key for key in ORDER_PARAMETER_KEYS if key != "cosine_Y")
    ]
    for record in sorted(records, key=lambda r: (r["noise_var"], r["replica_id"])):
        lines.append(
            f"{record['noise_var']},{record['alpha']},{record['num_observed']},"
            f"{record['replica']},{record['seed']},{record['device']},"
            f"{record['runtime_sec']:.4f},{record['final_loss']:.10e},"
            f"{record['initial_loss']:.10e},{record['step1_loss']:.10e},"
            f"{record['min_loss']:.10e},{record['min_loss_step']},"
            f"{record['steps_taken']},{record['cosine_Y']:.10e},"
            f"{record['initial_cosine_Y']:.10e},"
            f"{record['step1_cosine_Y']:.10e},"
            f"{record['max_cosine_Y']:.10e},{record['max_cosine_Y_step']},"
            + ",".join(
                f"{record[key]:.10e}" for key in ORDER_PARAMETER_KEYS if key != "cosine_Y"
            )
        )
    write_text_atomic(summary_path, "\n".join(lines) + "\n")


def save_results_npz(results_dir: Path, records: list[dict[str, Any]]) -> None:
    npz_path = results_dir / "results.npz"
    ordered = sorted(records, key=lambda r: (r["noise_var"], r["replica_id"]))
    if not ordered:
        np.savez(npz_path, empty=np.array([], dtype=np.float32))
        return

    np.savez(
        npz_path,
        noise_var=np.array(
            [record["noise_var"] for record in ordered],
            dtype=np.float64,
        ),
        alpha=np.array([record["alpha"] for record in ordered], dtype=np.float64),
        replica=np.array([record["replica"] for record in ordered], dtype=np.int64),
        replica_id=np.array(
            [record["replica_id"] for record in ordered],
            dtype=np.int64,
        ),
        seed=np.array([record["seed"] for record in ordered], dtype=np.int64),
        num_observed=np.array(
            [record["num_observed"] for record in ordered],
            dtype=np.int64,
        ),
        runtime_sec=np.array(
            [record["runtime_sec"] for record in ordered],
            dtype=np.float64,
        ),
        final_loss=np.array(
            [record["final_loss"] for record in ordered],
            dtype=np.float64,
        ),
        initial_loss=np.array(
            [record["initial_loss"] for record in ordered],
            dtype=np.float64,
        ),
        step1_loss=np.array(
            [record["step1_loss"] for record in ordered],
            dtype=np.float64,
        ),
        min_loss=np.array(
            [record["min_loss"] for record in ordered],
            dtype=np.float64,
        ),
        min_loss_step=np.array(
            [record["min_loss_step"] for record in ordered],
            dtype=np.int64,
        ),
        steps_taken=np.array(
            [record["steps_taken"] for record in ordered],
            dtype=np.int64,
        ),
        cosine_Y=np.array(
            [record["cosine_Y"] for record in ordered],
            dtype=np.float64,
        ),
        initial_cosine_Y=np.array(
            [record["initial_cosine_Y"] for record in ordered],
            dtype=np.float64,
        ),
        step1_cosine_Y=np.array(
            [record["step1_cosine_Y"] for record in ordered],
            dtype=np.float64,
        ),
        max_cosine_Y=np.array(
            [record["max_cosine_Y"] for record in ordered],
            dtype=np.float64,
        ),
        max_cosine_Y_step=np.array(
            [record["max_cosine_Y_step"] for record in ordered],
            dtype=np.int64,
        ),
        device=np.array([record["device"] for record in ordered]),
        **{
            key: np.array([record[key] for record in ordered], dtype=np.float64)
            for key in ORDER_PARAMETER_KEYS
            if key != "cosine_Y"
        },
    )


def plot_results(
    results_dir: Path,
    results: dict[float, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    completed_noises = sorted(results.keys())
    if not completed_noises:
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    def configure_x_axis(ax: plt.Axes) -> None:
        if args.log_x:
            if any(noise <= 0 for noise in completed_noises):
                raise ValueError("--log-x requires all completed noise values to be positive.")
            ax.set_xscale("log")
        elif len(completed_noises) == 1:
            noise = completed_noises[0]
            width = 0.1 if noise == 0 else abs(noise) * 0.1
            ax.set_xlim(noise - width, noise + width)

    def errorbar_plot(
        filename: str,
        mean_key: str,
        std_key: str,
        ylabel: str,
        title: str,
        *,
        ylim: tuple[float, float] | None = None,
        yscale: str | None = None,
        color: str = "#1976D2",
    ) -> None:
        means = [results[noise][mean_key] for noise in completed_noises]
        stds = [results[noise][std_key] for noise in completed_noises]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(
            completed_noises,
            means,
            yerr=stds,
            fmt="o-",
            color=color,
            markersize=6,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
        )
        ax.set_xlabel("noise_var", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(
            f"{title} (alpha={args.alpha})\n"
            f"({args.N1}x{args.N2}, M={args.M}, {args.max_steps} steps, "
            f"{args.num_replicas} replicas)",
            fontsize=16,
        )
        configure_x_axis(ax)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yscale is not None:
            ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    errorbar_plot(
        "cosine_Y_vs_noise.png",
        "cosine_Y_mean",
        "cosine_Y_std",
        "Final cosine_Y similarity",
        "Final cosine_Y Similarity vs Noise",
        ylim=(-0.05, 1.05),
    )
    errorbar_plot(
        "max_cosine_Y_vs_noise.png",
        "max_cosine_Y_mean",
        "max_cosine_Y_std",
        "Max cosine_Y similarity",
        "Max cosine_Y Similarity vs Noise",
        ylim=(-0.05, 1.05),
        color="#2E7D32",
    )
    errorbar_plot(
        "step1_cosine_Y_vs_noise.png",
        "step1_cosine_Y_mean",
        "step1_cosine_Y_std",
        "Step 1 cosine_Y similarity",
        "Step 1 cosine_Y Similarity vs Noise",
        ylim=(-0.05, 1.05),
        color="#EF6C00",
    )
    errorbar_plot(
        "final_loss_vs_noise.png",
        "loss_mean",
        "loss_std",
        "Final observed loss",
        "Final Loss vs Noise",
        yscale="log",
        color="#6A1B9A",
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    series = [
        ("step 1", "step1_cosine_Y_mean", "#EF6C00"),
        ("max", "max_cosine_Y_mean", "#2E7D32"),
        ("final", "cosine_Y_mean", "#1976D2"),
    ]
    for label, key, color in series:
        ax.plot(
            completed_noises,
            [results[noise][key] for noise in completed_noises],
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax.set_xlabel("noise_var", fontsize=14)
    ax.set_ylabel("cosine_Y similarity", fontsize=14)
    ax.set_title(
        f"cosine_Y Summary vs Noise (alpha={args.alpha})\n"
        f"({args.N1}x{args.N2}, M={args.M}, {args.max_steps} steps)",
        fontsize=16,
    )
    configure_x_axis(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_Y_summary_vs_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_progress_outputs(
    results_dir: Path,
    records: list[dict[str, Any]],
    noise_values: list[float],
    num_replicas: int,
    completed: int,
    total_tasks: int,
    start_time: float,
    status: str,
    args: argparse.Namespace,
) -> dict[float, dict[str, Any]]:
    results = aggregate_results(records, noise_values, num_replicas)
    save_metrics_csv(results_dir, results, noise_values, num_replicas)
    save_replica_summary(results_dir, records)
    save_results_npz(results_dir, records)
    plot_results(results_dir, results, args)

    progress = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_tasks": completed,
        "total_tasks": total_tasks,
        "completed_noise_values": sorted(results.keys()),
        "elapsed_sec": time.time() - start_time,
    }
    write_text_atomic(
        results_dir / "progress.yaml",
        yaml.safe_dump(progress, sort_keys=False),
    )
    return results


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    devices: list[str],
    noise_values: list[float],
) -> None:
    config = {
        "algorithm": "gamp_Edge_Alternating_random_F_sequential_noise_sweep_parallel",
        "graph_model": "random_graph",
        "f_mode": "random",
        "f_distribution": "rademacher_pm1",
        "f_values": [-1, 1],
        "effective_F_values": [
            f"-{args.lam}/sqrt(M)",
            f"+{args.lam}/sqrt(M)",
        ],
        "sequential_aggregation": True,
        "stores_F_edge_tensor": False,
        "edge_chunk_size": args.edge_chunk_size,
        "parallelism": "one_worker_process_per_device_one_replica_at_a_time",
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "alpha": args.alpha,
        "lambda": args.lam,
        "noise_start": args.noise_start,
        "noise_stop": args.noise_stop,
        "noise_step": args.noise_step,
        "noise_values": noise_values,
        "max_steps": args.max_steps,
        "damping": args.damping,
        "damping_schedule": args.damping_schedule,
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "teacher_seed": args.shared_seed,
        "graph_seed": args.shared_seed,
        "noise_seed": args.shared_seed,
        "f_seed": args.shared_seed + 1000,
        "student_seed_base": args.student_seed_base,
        "student_init_mode": "correlated_gaussian",
        "student_init_formula": (
            "student = epsilon * teacher + "
            "sqrt(epsilon - epsilon^2) * N(0, 1)"
        ),
        "student_init_epsilon": args.init_epsilon,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "early_stop": False,
        "loss_eval_interval": 1,
        "includes_initial_state": True,
        "initial_state_step": 0,
        "save_every_replicas": args.save_every_replicas,
        "devices": devices,
        "torch_threads_per_worker": args.torch_threads,
        "deterministic_requested": args.deterministic,
        "evaluation_metric": "dense_cosine_Y_order_parameter",
        "order_parameters": list(ORDER_PARAMETER_KEYS),
        "cosine_Y_domain": "dense_all_N1_N2_pairs_including_unobserved",
        "update_scheme": "alternating_W_then_X",
        "onsager_memory_schedule": "half_step",
        "shared_teacher_global": True,
        "shared_graph_per_alpha": True,
        "shared_random_F_per_alpha": True,
        "shared_noise_base_seed": True,
        "replica_variation": "student_initialization_only",
        "output_files": [
            "config.yaml",
            "metrics.csv",
            "replica_summary.csv",
            "results.npz",
            "progress.yaml",
            "plots/cosine_Y_vs_noise.png",
            "plots/max_cosine_Y_vs_noise.png",
            "plots/step1_cosine_Y_vs_noise.png",
            "plots/final_loss_vs_noise.png",
            "plots/cosine_Y_summary_vs_noise.png",
        ],
    }
    write_text_atomic(
        results_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )


def print_summary(results: dict[float, dict[str, Any]], total_time: float) -> None:
    print("\n" + "=" * 92)
    print("Results (mean +- std)")
    print("=" * 92)
    print(
        f"{'Noise':>12} | {'FinalCosineY':^15} | {'MaxCosineY':^15} | "
        f"{'Step1CosineY':^15} | {'Loss':^13} | {'Steps':>8}"
    )
    print("-" * 92)
    for noise_var in sorted(results.keys()):
        result = results[noise_var]
        print(
            f"{noise_var:12.4e} | "
            f"{result['cosine_Y_mean']:6.4f} +- "
            f"{result['cosine_Y_std']:<6.4f} | "
            f"{result['max_cosine_Y_mean']:6.4f} +- "
            f"{result['max_cosine_Y_std']:<6.4f} | "
            f"{result['step1_cosine_Y_mean']:6.4f} +- "
            f"{result['step1_cosine_Y_std']:<6.4f} | "
            f"{result['loss_mean']:8.2e} | "
            f"{result['steps_mean']:8.0f}"
        )
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 3600.0:.2f}h)")
    print("=" * 92)


def main() -> int:
    args = parse_args()
    devices = resolve_devices(args)
    noise_values = build_noise_values(args)

    if args.save_every_replicas <= 0:
        raise ValueError("--save-every-replicas must be positive.")
    if args.edge_chunk_size <= 0:
        raise ValueError("--edge-chunk-size must be positive.")
    if not 0.0 <= args.init_epsilon <= 1.0:
        raise ValueError("--init-epsilon must satisfy 0 <= epsilon <= 1.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = (
        args.results_root
        if args.results_root is not None
        else Path(__file__).parent / "results"
    )
    results_dir = results_root / (
        f"{timestamp}_gamp_Edge_Alternating_random_F_sequential_noise_sweep_"
        f"{args.N1}x{args.N2}_M{args.M}_alpha{args.alpha}_"
        f"noise{noise_values[0]}-{noise_values[-1]}"
        f"_chunk{args.edge_chunk_size}"
        f"_initeps{args.init_epsilon if args.init_epsilon is not None else 'random'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Parallel Edge_Alternating G-AMP - Sequential Random-F Noise Sweep")
    print("=" * 72)
    print(f"Matrix: {args.N1}x{args.N2}, M={args.M}")
    print(f"Alpha: {args.alpha}")
    print(f"Lambda: {args.lam}")
    print(f"F: random Rademacher +/-1, effective scale lambda/sqrt(M)")
    print(f"Edge chunk size: {args.edge_chunk_size}")
    print(f"Noise values: {noise_values}")
    if args.damping_schedule == "beta":
        print(
            f"Steps: {args.max_steps}, Damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"Steps: {args.max_steps}, Damping: {args.damping}")
    print(f"Replicas per noise: {args.num_replicas}")
    print(f"Devices: {', '.join(devices)}")
    print("Execution rule: one worker process per device, one active replica per device")
    print("Teacher / graph / noise seed:", args.shared_seed)
    print("F seed:", args.shared_seed + 1000)
    print("Student seed rule:", f"{args.student_seed_base} + replica_id")
    print(
        "Student init: epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1), "
        f"epsilon={args.init_epsilon}"
    )
    print(f"Partial save cadence: every {args.save_every_replicas} replicas")
    print(f"Results directory: {results_dir}")
    print()

    save_config(results_dir, args, devices, noise_values)

    total_tasks = len(noise_values) * args.num_replicas
    if total_tasks == 0:
        raise RuntimeError("No replica tasks were generated.")

    worker_config = {
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "alpha": args.alpha,
        "max_steps": args.max_steps,
        "damping": args.damping,
        "use_step_damping": args.damping_schedule == "beta",
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "lam": args.lam,
        "edge_chunk_size": args.edge_chunk_size,
        "shared_seed": args.shared_seed,
        "convergence_threshold": args.convergence_threshold,
        "init_epsilon": args.init_epsilon,
        "torch_threads": args.torch_threads,
        "deterministic": args.deterministic,
    }

    start_time = time.time()
    completed = 0
    records: list[dict[str, Any]] = []
    interrupted = False

    def on_replica_done(message: dict[str, Any], current_records: list[dict[str, Any]]) -> None:
        nonlocal completed
        completed += 1
        elapsed = time.time() - start_time
        eta = elapsed / completed * (total_tasks - completed) if completed else 0.0
        print(
            f"[{completed}/{total_tasks}] "
            f"device={message['device']} "
            f"noise={message['noise_var']:.4e}, "
            f"replica {message['replica']}/{args.num_replicas}: "
            f"cosine_Y={message['cosine_Y']:.4f}, "
            f"MaxCosineY={message['max_cosine_Y']:.4f}, "
            f"Step1CosineY={message['step1_cosine_Y']:.4f}, "
            f"Loss={message['final_loss']:.2e}, "
            f"Steps={message['steps_taken']} "
            f"({message['runtime_sec']:.1f}s) ETA={eta / 3600.0:.2f}h"
        )

        if completed % args.save_every_replicas == 0 or completed == total_tasks:
            save_progress_outputs(
                results_dir=results_dir,
                records=current_records,
                noise_values=noise_values,
                num_replicas=args.num_replicas,
                completed=completed,
                total_tasks=total_tasks,
                start_time=start_time,
                status="running",
                args=args,
            )

    try:
        records = run_parallel_replicas(
            noise_values=noise_values,
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

    total_time = time.time() - start_time
    status = "interrupted" if interrupted or completed < total_tasks else "completed"
    results = save_progress_outputs(
        results_dir=results_dir,
        records=records,
        noise_values=noise_values,
        num_replicas=args.num_replicas,
        completed=completed,
        total_tasks=total_tasks,
        start_time=start_time,
        status=status,
        args=args,
    )
    print_summary(results, total_time)
    print(f"\nMetrics saved: {results_dir / 'metrics.csv'}")
    print(f"Final cosine_Y plot saved: {results_dir / 'plots' / 'cosine_Y_vs_noise.png'}")
    print(f"cosine_Y summary plot saved: {results_dir / 'plots' / 'cosine_Y_summary_vs_noise.png'}")
    print(f"Replica summary saved: {results_dir / 'replica_summary.csv'}")
    print(f"Single-file results saved: {results_dir / 'results.npz'}")
    print(f"Results saved to: {results_dir}")

    return 130 if status == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
