#!/usr/bin/env python
"""
Parallel loss-vs-step runner for sequentially aggregated F=1 Edge_Alternating.
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

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Edge_Alternating.F_1_sequentially.core import (  # noqa: E402
    DEFAULT_EDGE_CHUNK_SIZE,
)
from terao_gamp_gaussian.Edge_Alternating.random_F_version.loss_vs_step import (  # noqa: E402
    DEFAULT_ALPHA,
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
)
from terao_gamp_gaussian.Edge_Alternating.random_graph_version.loss_vs_step import (  # noqa: E402
    assign_replicas_to_devices,
    build_history_arrays,
    resolve_devices,
    save_progress_outputs,
    write_text_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-alpha loss-vs-step replicas for the sequentially aggregated "
            "F=1 Edge_Alternating variant."
        )
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Set N1=N2=N. Overrides --N1 and --N2 when provided.",
    )
    parser.add_argument("--N1", type=int, default=2500)
    parser.add_argument("--N2", type=int, default=2500)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--damping", type=float, default=0)
    parser.add_argument(
        "--damping-schedule",
        type=str,
        choices=["beta", "constant"],
        default="constant",
    )
    parser.add_argument("--beta-scale", type=float, default=DEFAULT_BETA_SCALE)
    parser.add_argument("--beta-max", type=float, default=DEFAULT_BETA_MAX)
    parser.add_argument("--noise-var", type=float, default=0.333)
    parser.add_argument("--seed", type=int, default=DEFAULT_SHARED_SEED)
    parser.add_argument("--shared-seed", type=int, default=DEFAULT_SHARED_SEED)
    parser.add_argument("--student-seed-base", type=int, default=DEFAULT_STUDENT_SEED_BASE)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=DEFAULT_CONVERGENCE_THRESHOLD,
    )
    parser.add_argument(
        "--init-epsilon",
        type=float,
        default=1,
        help=(
            "Use informative student initialization: epsilon * teacher + "
            "sqrt(epsilon - epsilon^2) * N(0, 1)."
        ),
    )
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=DEFAULT_EDGE_CHUNK_SIZE,
        help=(
            "Number of observed edges processed at once. Lower this to reduce "
            "peak memory at the cost of runtime."
        ),
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


def save_config(results_dir: Path, args: argparse.Namespace, devices: list[str]) -> None:
    config = {
        "algorithm": "gamp_Edge_Alternating_F_1_loss_vs_step_sequential",
        "graph_model": "random_graph",
        "f_mode": "fixed",
        "f_distribution": "constant_one",
        "f_value": 1.0,
        "effective_F_values": "+lambda / sqrt(M)",
        "sequential_aggregation": True,
        "stores_F_edge": False,
        "edge_chunk_size": args.edge_chunk_size,
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
        "loss_eval_interval": 1,
        "includes_initial_state": True,
        "initial_state_step": 0,
        "early_stop": False,
        "save_every_replicas": args.save_every_replicas,
        "devices": devices,
        "torch_threads_per_worker": args.torch_threads,
        "deterministic_requested": args.deterministic,
        "evaluation_metric": "cosine_similarity_in_observed_signal_space",
        "update_scheme": "alternating_W_then_X",
        "step_definition": "one_W_update_plus_one_X_update",
        "onsager_memory_schedule": "half_step",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "shared_F_per_alpha": "constant_one",
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


def estimate_convergence_step_from_steps(
    steps: np.ndarray,
    loss_history: np.ndarray,
    threshold: float,
) -> float:
    if loss_history.size < 2:
        return float("nan")

    delta = np.abs(np.diff(loss_history))
    stable_idx = np.where(delta < threshold)[0]
    if stable_idx.size == 0:
        return float("nan")

    return float(steps[stable_idx[0] + 1])


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

        from terao_gamp_gaussian.Edge_Alternating.F_1_sequentially.core import (
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
            edge_chunk_size=int(worker_config["edge_chunk_size"]),
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
                edge_chunk_size=int(worker_config["edge_chunk_size"]),
                shared_data=shared_data,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime = time.time() - t0

            loss_history = np.asarray(history["loss"], dtype=np.float64)
            cosine_history = np.asarray(history["cosine_similarity"], dtype=np.float64)
            steps = np.asarray(history["steps"], dtype=np.int64)
            convergence_step = estimate_convergence_step_from_steps(
                steps,
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

        result_queue.put({"event": "worker_done", "ok": True, "device_slot": device_slot, "device": device_name})
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
    if args.edge_chunk_size <= 0:
        raise ValueError("--edge-chunk-size must be positive.")
    if args.save_every_replicas <= 0:
        raise ValueError("--save-every-replicas must be positive.")

    print("=" * 72)
    print("Sequential F=1 Edge_Alternating Loss vs Step")
    print("=" * 72)
    print(f"Devices: {', '.join(devices)}")
    print(f"alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}")
    print(f"edge_chunk_size={args.edge_chunk_size}; F is fixed to 1")
    if args.damping_schedule == "beta":
        print(
            f"max_steps={args.max_steps}, damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"max_steps={args.max_steps}, damping={args.damping}")
    print("Step definition: one W update followed by one X update")
    print("Teacher / graph / noise seed:", args.shared_seed)
    print("Student seed rule:", f"{args.student_seed_base} + replica_index")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = args.results_root if args.results_root is not None else Path(__file__).parent / "results"
    results_dir = results_root / (
        f"{timestamp}_loss_vs_step_Edge_Alternating_F_1_sequential_"
        f"alpha{args.alpha}_{args.N1}x{args.N2}_M{args.M}_chunk{args.edge_chunk_size}"
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
        "edge_chunk_size": args.edge_chunk_size,
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
            f"steps_taken={message['steps_taken']}, "
            f"history_points={len(message['steps'])}, "
            f"runtime={message['runtime_sec']:.1f}s [{completed}/{total_tasks}]"
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
        print("\nInterrupted. Saving partial results...")
    finally:
        save_progress_outputs(
            results_dir=results_dir,
            records=records,
            completed=len(records),
            total_tasks=total_tasks,
            start_time=start_time,
            status="interrupted" if interrupted else "completed",
            args=args,
        )

    if records:
        _, all_losses, all_cosine_similarities = build_history_arrays(records)
        print()
        print(f"Mean final loss: {all_losses.mean(axis=0)[-1]:.10e}")
        print(f"Mean cosine similarity: {all_cosine_similarities.mean(axis=0)[-1]:.10f}")
    print(f"Total runtime: {time.time() - start_time:.1f}s")
    print(f"Results saved to: {results_dir}")
    return 0 if not interrupted else 130


if __name__ == "__main__":
    raise SystemExit(main())
