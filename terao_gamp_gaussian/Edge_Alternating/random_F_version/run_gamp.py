#!/usr/bin/env python
"""
Parallel runner for the random-F version of Edge_Alternating G-AMP.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.multiprocessing as mp
import yaml

# Add parent directories to path.
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Edge_Alternating.random_graph_version.run_gamp import (
    DEFAULT_ALPHA_START,
    DEFAULT_ALPHA_STEP,
    DEFAULT_ALPHA_STOP,
    DEFAULT_BETA_MAX,
    DEFAULT_BETA_SCALE,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_DAMPING,
    DEFAULT_DAMPING_SCHEDULE,
    DEFAULT_INIT_EPSILON,
    DEFAULT_M,
    DEFAULT_MAX_STEPS,
    DEFAULT_N1,
    DEFAULT_N2,
    DEFAULT_NOISE_VAR,
    DEFAULT_NUM_REPLICAS,
    DEFAULT_SAVE_EVERY_REPLICAS,
    DEFAULT_SHARED_SEED,
    DEFAULT_STUDENT_SEED_BASE,
    DEFAULT_TORCH_THREADS,
    aggregate_results,
    assign_tasks_to_devices,
    build_alpha_values,
    plot_results,
    print_summary,
    resolve_devices,
    save_metrics_csv,
    save_progress_outputs,
    save_replica_summary,
    save_results_npz,
    write_text_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the random-F version of Edge_Alternating G-AMP in parallel, "
            "one worker process per device."
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
    parser.add_argument("--alpha-start", type=float, default=DEFAULT_ALPHA_START)
    parser.add_argument("--alpha-stop", type=float, default=DEFAULT_ALPHA_STOP)
    parser.add_argument("--alpha-step", type=float, default=DEFAULT_ALPHA_STEP)
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
            "Use informative student initialization: teacher + epsilon * N(0, 1), "
            "then mean-square normalization. Set DEFAULT_INIT_EPSILON=None for "
            "random Gaussian initialization."
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

        from terao_gamp_gaussian.Edge_Alternating.random_F_version.core import (
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

        current_alpha: float | None = None
        shared_data: dict[str, Any] | None = None

        for task in tasks:
            alpha = float(task["alpha"])
            if current_alpha != alpha:
                shared_data = prepare_shared_alpha_data(
                    alpha=alpha,
                    device=device,
                    seed=int(worker_config["shared_seed"]),
                    N1=int(worker_config["N1"]),
                    N2=int(worker_config["N2"]),
                    M=int(worker_config["M"]),
                    noise_var=float(worker_config["noise_var"]),
                    global_data=global_data,
                )
                current_alpha = alpha

            if shared_data is None:
                raise RuntimeError("Internal error: shared alpha data is missing.")

            num_observed = int(shared_data["E"])

            t0 = time.time()
            cosine_similarity, final_loss, steps_taken = train_single_replica(
                alpha=alpha,
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
                init_epsilon=worker_config["init_epsilon"],
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
                    "num_observed": num_observed,
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


def run_parallel_replicas(
    alphas: list[float],
    devices: list[str],
    args: argparse.Namespace,
    worker_config: dict[str, Any],
    on_replica_done: Callable[[dict[str, Any], list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    tasks_by_device = assign_tasks_to_devices(
        alphas=alphas,
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


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    devices: list[str],
    alphas: list[float],
) -> None:
    config = {
        "algorithm": "gamp_Edge_Alternating_random_F_parallel",
        "graph_model": "random_graph",
        "f_mode": "random",
        "parallelism": "one_worker_process_per_device_one_replica_at_a_time",
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "alpha_start": args.alpha_start,
        "alpha_stop": args.alpha_stop,
        "alpha_step": args.alpha_step,
        "alphas": alphas,
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
            "teacher_plus_noise_normalized"
            if args.init_epsilon is not None
            else "random_gaussian"
        ),
        "student_init_epsilon": args.init_epsilon,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "save_every_replicas": args.save_every_replicas,
        "devices": devices,
        "torch_threads_per_worker": args.torch_threads,
        "deterministic_requested": args.deterministic,
        "evaluation_metric": "cosine_similarity_in_observed_signal_space",
        "update_scheme": "alternating_W_then_X",
        "onsager_memory_schedule": "half_step",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "shared_random_F_per_alpha": True,
        "replica_variation": "student_initialization_only",
        "output_files": [
            "config.yaml",
            "metrics.csv",
            "replica_summary.csv",
            "results.npz",
            "progress.yaml",
            "plots/cosine_similarity_vs_alpha.png",
        ],
    }
    write_text_atomic(
        results_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False),
    )


def main() -> int:
    args = parse_args()
    devices = resolve_devices(args)
    alphas = build_alpha_values(args.alpha_start, args.alpha_stop, args.alpha_step)

    if args.save_every_replicas <= 0:
        raise ValueError("--save-every-replicas must be positive.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = (
        args.results_root
        if args.results_root is not None
        else Path(__file__).parent / "results"
    )
    results_dir = results_root / (
        f"{timestamp}_gamp_Edge_Alternating_random_F_parallel_"
        f"{args.N1}x{args.M}_alpha{args.alpha_start}-{args.alpha_stop}"
        f"_initeps{args.init_epsilon if args.init_epsilon is not None else 'random'}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Parallel Edge_Alternating G-AMP - Random F Version")
    print("=" * 72)
    print(f"Matrix: {args.N1}x{args.N2}, M={args.M}")
    print(f"Alpha: {args.alpha_start} ~ {args.alpha_stop} (step {args.alpha_step})")
    if args.damping_schedule == "beta":
        print(
            f"Steps: {args.max_steps}, Damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"Steps: {args.max_steps}, Damping: {args.damping}")
    print(f"Replicas per alpha: {args.num_replicas}")
    print(f"Devices: {', '.join(devices)}")
    print("Execution rule: one worker process per device, one active replica per device")
    print("Teacher / graph / noise seed:", args.shared_seed)
    print("F seed:", args.shared_seed + 1000)
    print("Student seed rule:", f"{args.student_seed_base} + replica_id")
    if args.init_epsilon is None:
        print("Student init: random Gaussian")
    else:
        print(
            "Student init: teacher + epsilon * N(0, 1), "
            f"epsilon={args.init_epsilon} (then mean-square normalization)"
        )
    print(f"Partial save cadence: every {args.save_every_replicas} replicas")
    print(f"Results directory: {results_dir}")

    save_config(results_dir, args, devices, alphas)

    worker_config = {
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

    total_tasks = len(alphas) * args.num_replicas
    completed = 0
    start_time = time.time()
    interrupted = False
    records: list[dict[str, Any]] = []

    def on_replica_done(message: dict[str, Any], current_records: list[dict[str, Any]]) -> None:
        nonlocal completed
        completed += 1
        elapsed = time.time() - start_time
        remaining = max(total_tasks - completed, 0)
        eta_hours = 0.0 if completed == 0 else (elapsed / completed) * remaining / 3600.0
        print(
            f"[{completed}/{total_tasks}] device={message['device']} "
            f"alpha={message['alpha']:.2f}, replica {message['replica']}/{args.num_replicas}: "
            f"CosSim={message['cosine_similarity']:.4f}, "
            f"Loss={message['final_loss']:.2e}, Steps={message['steps_taken']} "
            f"({message['runtime_sec']:.1f}s) ETA={eta_hours:.2f}h"
        )
        if completed % args.save_every_replicas == 0 or completed == total_tasks:
            save_progress_outputs(
                results_dir=results_dir,
                records=current_records,
                alphas=alphas,
                num_replicas=args.num_replicas,
                completed=completed,
                total_tasks=total_tasks,
                start_time=start_time,
                status="running",
                args=args,
            )

    try:
        records = run_parallel_replicas(
            alphas=alphas,
            devices=devices,
            args=args,
            worker_config=worker_config,
            on_replica_done=on_replica_done,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving partial results...")
    finally:
        status = "interrupted" if interrupted else "completed"
        results = save_progress_outputs(
            results_dir=results_dir,
            records=records,
            alphas=alphas,
            num_replicas=args.num_replicas,
            completed=len(records),
            total_tasks=total_tasks,
            start_time=start_time,
            status=status,
            args=args,
        )

    total_time = time.time() - start_time
    print_summary(results, total_time)
    print(f"Metrics saved: {results_dir / 'metrics.csv'}")
    print(f"Replica summary saved: {results_dir / 'replica_summary.csv'}")
    print(f"Single-file results saved: {results_dir / 'results.npz'}")
    print(f"Results saved to: {results_dir}")
    return 0 if not interrupted else 130


if __name__ == "__main__":
    raise SystemExit(main())
