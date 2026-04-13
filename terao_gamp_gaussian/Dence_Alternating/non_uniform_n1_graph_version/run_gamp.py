#!/usr/bin/env python
"""
Dense-mask alternating G-AMP simulation runner with the non-uniform N1 graph
version based on graph_core.two_point.
"""

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ============================================================================
# Configuration
# ============================================================================

N1 = 2000
N2 = 2000
M = 200

ALPHA_START = 0.2
ALPHA_STOP = 5.0
ALPHA_STEP = 0.2

P = 0.5
R = 0.5

MAX_STEPS = 10000
DAMPING = 0
USE_STEP_DAMPING = False
DAMPING_BETA_SCALE = 1e-3
DAMPING_BETA_MAX = DAMPING
NOISE_VAR = 1e-5
SHARED_SEED = 1
STUDENT_SEED_BASE = 100
NUM_REPLICAS = 5
CONVERGENCE_THRESHOLD = 1e-5


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    print("=" * 60)
    print("Dense-mask Alternating G-AMP (non_uniform_n1_graph_version)")
    print("=" * 60)

    device = select_device()
    if device.type == "mps":
        print("Using: Apple Silicon (MPS)")
    elif device.type == "cuda":
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    else:
        print("Using: CPU")

    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Two-point parameters: p={P}, r={R}")
    if USE_STEP_DAMPING:
        print(
            f"Steps: {MAX_STEPS}, Damping schedule: "
            f"beta=max(1-step*{DAMPING_BETA_SCALE}, {DAMPING_BETA_MAX})"
        )
    else:
        print(f"Steps: {MAX_STEPS}, Damping: {DAMPING}")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_gamp_Dence_Alternating_non_uniform_n1_{N1}x{M}"
        f"_alpha{ALPHA_START}-{ALPHA_STOP}_p{P}_r{R}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    config = {
        "algorithm": "gamp_Dence_Alternating_non_uniform_n1_graph_version",
        "graph_model": "non_uniform_n1_graph",
        "N1": N1,
        "N2": N2,
        "M": M,
        "alpha_start": ALPHA_START,
        "alpha_stop": ALPHA_STOP,
        "alpha_step": ALPHA_STEP,
        "p": P,
        "r": R,
        "max_steps": MAX_STEPS,
        "damping": DAMPING,
        "use_step_damping": USE_STEP_DAMPING,
        "damping_beta_scale": DAMPING_BETA_SCALE,
        "damping_beta_max": DAMPING_BETA_MAX,
        "noise_var": NOISE_VAR,
        "teacher_seed": SHARED_SEED,
        "graph_seed": SHARED_SEED,
        "noise_seed": SHARED_SEED,
        "student_seed_base": STUDENT_SEED_BASE,
        "num_replicas": NUM_REPLICAS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "device": str(device),
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP / 2, ALPHA_STEP)
    results = {}

    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0
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
            p=P,
            r=R,
            global_data=global_data,
        )
        cosine_similarity_values = []
        loss_values = []
        steps_values = []

        for replica_id in range(NUM_REPLICAS):
            seed = STUDENT_SEED_BASE + replica_id
            t0 = time.time()

            cosine_similarity, final_loss, steps_taken = train_single_replica(
                alpha=alpha,
                device=device,
                seed=seed,
                N1=N1,
                N2=N2,
                M=M,
                max_steps=MAX_STEPS,
                damping=DAMPING,
                use_step_damping=USE_STEP_DAMPING,
                damping_beta_scale=DAMPING_BETA_SCALE,
                damping_beta_max=DAMPING_BETA_MAX,
                noise_var=NOISE_VAR,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                p=P,
                r=R,
                shared_data=shared_data,
            )

            dt = time.time() - t0
            cosine_similarity_values.append(cosine_similarity)
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            completed += 1

            print(
                f"alpha={alpha:.2f}, replica {replica_id + 1}/{NUM_REPLICAS}: "
                f"CosSim={cosine_similarity:.10f}, Loss={final_loss:.10e}, "
                f"Steps={steps_taken} ({dt:.1f}s) [{completed}/{total_tasks}]"
            )

        results[alpha] = {
            "cosine_similarity_mean": np.mean(cosine_similarity_values),
            "cosine_similarity_std": np.std(cosine_similarity_values),
            "cosine_similarity_values": cosine_similarity_values,
            "loss_mean": np.mean(loss_values),
            "loss_std": np.std(loss_values),
            "steps_mean": np.mean(steps_values),
        }

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    alphas_list = sorted(results.keys())
    cosine_similarity_means = [results[a]["cosine_similarity_mean"] for a in alphas_list]
    cosine_similarity_stds = [results[a]["cosine_similarity_std"] for a in alphas_list]
    cosine_similarity_sems = [
        std / math.sqrt(NUM_REPLICAS) for std in cosine_similarity_stds
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        alphas_list,
        cosine_similarity_means,
        yerr=cosine_similarity_sems,
        fmt="o-",
        color="#D81B60",
        markersize=6,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label="Non-uniform N1 graph version",
    )
    ax.set_xlabel(r"$\alpha$ (observation density)", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title("Cosine Similarity vs Alpha", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cosine_similarity_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
