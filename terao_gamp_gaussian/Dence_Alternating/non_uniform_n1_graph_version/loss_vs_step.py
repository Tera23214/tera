#!/usr/bin/env python
"""
Plot loss vs step for the non-uniform N1 graph version of the dense-mask
alternating F=1 Onsager G-AMP experiment.

One recorded step means one full W -> X sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Dence_Alternating.non_uniform_n1_graph_version.core import (
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)


def _load_random_loss_vs_step_helpers():
    helper_path = (
        Path(__file__).resolve().parent.parent
        / "random_graph_version"
        / "loss_vs_step.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_random_graph_loss_vs_step_helpers",
        helper_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {helper_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_random_loss_vs_step = _load_random_loss_vs_step_helpers()
detect_device = _random_loss_vs_step.detect_device
estimate_convergence_step = _random_loss_vs_step.estimate_convergence_step
plot_cosine_similarity = _random_loss_vs_step.plot_cosine_similarity
plot_linear_loss = _random_loss_vs_step.plot_linear_loss
plot_log_loss = _random_loss_vs_step.plot_log_loss
save_loss_history = _random_loss_vs_step.save_loss_history
save_replica_summary = _random_loss_vs_step.save_replica_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss vs step for fixed alpha (non-uniform N1 graph)."
    )
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--N1", type=int, default=2000)
    parser.add_argument("--N2", type=int, default=2000)
    parser.add_argument("--M", type=int, default=200)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=10)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--damping", type=float, default=0)
    parser.add_argument(
        "--damping-schedule",
        type=str,
        choices=["beta", "constant"],
        default="constant",
    )
    parser.add_argument("--beta-scale", type=float, default=1e-2)
    parser.add_argument("--beta-max", type=float, default=0.4)
    parser.add_argument("--noise-var", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    return parser.parse_args()


def save_config(
    results_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    shared_data: dict[str, object],
) -> None:
    config = {
        "algorithm": "gamp_Dence_Alternating_non_uniform_n1_cosine_loss_vs_step",
        "graph_model": "non_uniform_n1_graph",
        "alpha": args.alpha,
        "N1": args.N1,
        "N2": args.N2,
        "M": args.M,
        "p": args.p,
        "r": args.r,
        "p_eff": shared_data.get("p_eff"),
        "alpha_eff": shared_data.get("alpha_eff"),
        "ca": shared_data.get("ca"),
        "cb": shared_data.get("cb"),
        "num_ca": shared_data.get("num_ca"),
        "num_cb": shared_data.get("num_cb"),
        "E": int(shared_data["E"]),
        "max_steps": args.max_steps,
        "damping": args.damping,
        "damping_schedule": args.damping_schedule,
        "beta_scale": args.beta_scale,
        "beta_max": args.beta_max,
        "noise_var": args.noise_var,
        "teacher_seed": 1,
        "graph_seed": 1,
        "noise_seed": 1,
        "student_seed_base": 100,
        "legacy_cli_seed": args.seed,
        "num_replicas": args.num_replicas,
        "convergence_threshold": args.convergence_threshold,
        "loss_eval_interval": 1,
        "early_stop": False,
        "evaluation_metric": "cosine_similarity_in_Y_space",
        "update_scheme": "alternating_W_then_X",
        "step_definition": "one_W_update_plus_one_X_update",
        "onsager_memory_schedule": "half_step",
        "shared_teacher_noise_global": True,
        "shared_graph_per_alpha": True,
        "dense_mask": True,
        "device": str(device),
    }

    config_path = results_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    device = detect_device()

    print("=" * 60)
    print("Loss vs Step for Dense-mask Alternating G-AMP")
    print("Graph model: non-uniform N1 two-point row degree")
    print("Evaluation Metric: Cosine Similarity in Y-space")
    print("=" * 60)
    print(f"Device: {device}")
    print(
        f"alpha={args.alpha}, N1={args.N1}, N2={args.N2}, M={args.M}, "
        f"p={args.p}, r={args.r}"
    )
    if args.damping_schedule == "beta":
        print(
            f"max_steps={args.max_steps}, damping schedule: "
            f"beta=max(1-step*{args.beta_scale}, {args.beta_max})"
        )
    else:
        print(f"max_steps={args.max_steps}, damping={args.damping}")
    print("Step definition: one W update followed by one X update")
    print("Onsager memory: advanced every half-step")
    print("Teacher / graph / noise seed: 1")
    print("Student seed rule: 100 + replica_index")
    print("Shared across run: teacher / noisy field")
    print("Shared per alpha: graph")
    print("Replica-specific: student initialization only")
    if args.seed != 1:
        print(
            f"Legacy CLI seed argument {args.seed} is ignored by this fixed seed policy."
        )
    print(f"replicas={args.num_replicas}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_loss_vs_step_Dence_Alternating_non_uniform_n1_alpha{args.alpha}_"
        f"{args.N1}x{args.N2}_M{args.M}_p{args.p}_r{args.r}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_losses = []
    all_cosine_similarities = []
    cosine_similarity_values = []
    final_losses = []
    runtimes = []
    seeds = []
    convergence_steps = []

    total_start = time.time()
    shared_seed = 1
    student_seed_base = 100
    global_data = prepare_global_shared_data(
        device=device,
        seed=shared_seed,
        N1=args.N1,
        N2=args.N2,
        M=args.M,
        noise_var=args.noise_var,
    )
    shared_data = prepare_shared_alpha_data(
        alpha=args.alpha,
        device=device,
        seed=shared_seed,
        N1=args.N1,
        N2=args.N2,
        M=args.M,
        noise_var=args.noise_var,
        p=args.p,
        r=args.r,
        global_data=global_data,
    )
    save_config(results_dir, args, device, shared_data)

    num_observations = int(shared_data["E"])
    print(f"Observed entries: {num_observations}")
    print(
        "Resolved degrees: "
        f"ca={shared_data.get('ca')}, cb={shared_data.get('cb')}, "
        f"num_ca={shared_data.get('num_ca')}, num_cb={shared_data.get('num_cb')}"
    )
    print(
        "Realized parameters: "
        f"p_eff={shared_data.get('p_eff')}, alpha_eff={shared_data.get('alpha_eff')}"
    )
    print("Loss definition: mean squared error on observed entries")
    print()

    for replica_idx in range(args.num_replicas):
        seed = student_seed_base + replica_idx
        replica_start = time.time()

        cosine_similarity, final_loss, steps_taken, history = train_single_replica(
            alpha=args.alpha,
            device=device,
            seed=seed,
            N1=args.N1,
            N2=args.N2,
            M=args.M,
            max_steps=args.max_steps,
            damping=args.damping,
            use_step_damping=args.damping_schedule == "beta",
            damping_beta_scale=args.beta_scale,
            damping_beta_max=args.beta_max,
            noise_var=args.noise_var,
            convergence_threshold=args.convergence_threshold,
            return_history=True,
            loss_eval_interval=1,
            early_stop=False,
            p=args.p,
            r=args.r,
            shared_data=shared_data,
        )

        runtime = time.time() - replica_start
        loss_history = np.asarray(history["loss"], dtype=np.float64)
        cosine_similarity_history = np.asarray(
            history["cosine_similarity"], dtype=np.float64
        )
        step_history = np.asarray(history["steps"], dtype=np.int64)
        convergence_step = estimate_convergence_step(
            loss_history, args.convergence_threshold
        )

        all_losses.append(loss_history)
        all_cosine_similarities.append(cosine_similarity_history)
        cosine_similarity_values.append(cosine_similarity)
        final_losses.append(final_loss)
        runtimes.append(runtime)
        seeds.append(seed)
        convergence_steps.append(convergence_step)

        convergence_text = (
            "not reached" if math.isnan(convergence_step) else str(int(convergence_step))
        )
        print(
            f"Replica {replica_idx + 1}/{args.num_replicas}: "
            f"seed={seed}, final_loss={final_loss:.10e}, "
            f"final_cosine_similarity={cosine_similarity:.10f}, "
            f"estimated_convergence_step={convergence_text}, "
            f"steps_recorded={steps_taken}, runtime={runtime:.1f}s"
        )

    total_runtime = time.time() - total_start

    all_losses_arr = np.asarray(all_losses, dtype=np.float64)
    all_cosine_similarities_arr = np.asarray(
        all_cosine_similarities, dtype=np.float64
    )
    steps = step_history
    mean_loss = all_losses_arr.mean(axis=0)
    std_loss = all_losses_arr.std(axis=0)
    log_losses = np.log10(np.clip(all_losses_arr, 1e-30, None))
    mean_log_loss = log_losses.mean(axis=0)
    std_log_loss = log_losses.std(axis=0)
    mean_cosine_similarity = all_cosine_similarities_arr.mean(axis=0)
    std_cosine_similarity = all_cosine_similarities_arr.std(axis=0)

    save_loss_history(results_dir, steps, all_losses_arr, all_cosine_similarities_arr)
    save_replica_summary(
        results_dir,
        seeds,
        runtimes,
        final_losses,
        convergence_steps,
        cosine_similarity_values,
    )
    plot_linear_loss(plots_dir, steps, all_losses_arr, mean_loss, std_loss, args)
    plot_log_loss(plots_dir, steps, all_losses_arr, mean_log_loss, std_log_loss, args)
    plot_cosine_similarity(
        plots_dir,
        steps,
        all_cosine_similarities_arr,
        mean_cosine_similarity,
        std_cosine_similarity,
        args,
    )

    print()
    print(f"Mean final loss: {np.mean(final_losses):.10e}")
    print(f"Mean cosine similarity: {np.mean(cosine_similarity_values):.10f}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
