#!/usr/bin/env python
"""
Dense-mask G-AMP simulation runner with scalar-variance Onsager correction and
cosine-similarity evaluation.

Graph / teacher / noisy observations are generated once per alpha and reused
across replicas. Replica-to-replica variation comes only from student
initialization.
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
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Dence_scaler_var_cosine.core import (
    prepare_shared_alpha_data,
    train_single_replica,
)


N1 = 220
N2 = 220
M = 200

ALPHA_START = 0.5
ALPHA_STOP = 3.0
ALPHA_STEP = 0.1

MAX_STEPS = 500
DAMPING = 0.5
USE_STEP_DAMPING = True
DAMPING_BETA_SCALE = 1e-3
DAMPING_BETA_MAX = DAMPING
NOISE_VAR = 1e-10
LAM = 1.0
SEED = 42
NUM_REPLICAS = 30
CONVERGENCE_THRESHOLD = 1e-6


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    print("=" * 60)
    print("Dense-mask G-AMP (scalar var + cosine similarity)")
    print("=" * 60)

    device = select_device()
    if device.type == "cuda":
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    elif device.type == "mps":
        print("Using: Apple Silicon (MPS)")
    else:
        print("Using: CPU")

    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    if USE_STEP_DAMPING:
        print(
            f"Steps: {MAX_STEPS}, Damping schedule: "
            f"beta=max(1-step*{DAMPING_BETA_SCALE}, {DAMPING_BETA_MAX})"
        )
    else:
        print(f"Steps: {MAX_STEPS}, Damping: {DAMPING}")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print("Shared per alpha: graph / teacher / noisy observation")
    print("Replica-specific: student initialization only")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_gamp_Dence_scaler_var_cosine_{N1}x{M}"
        f"_alpha{ALPHA_START}-{ALPHA_STOP}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    config = {
        "algorithm": "gamp_Dence_scaler_var_cosine",
        "backend": "dense_mask",
        "N1": N1,
        "N2": N2,
        "M": M,
        "alpha_start": ALPHA_START,
        "alpha_stop": ALPHA_STOP,
        "alpha_step": ALPHA_STEP,
        "max_steps": MAX_STEPS,
        "damping": DAMPING,
        "use_step_damping": USE_STEP_DAMPING,
        "damping_beta_scale": DAMPING_BETA_SCALE,
        "damping_beta_max": DAMPING_BETA_MAX,
        "noise_var": NOISE_VAR,
        "lam": LAM,
        "seed": SEED,
        "num_replicas": NUM_REPLICAS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "device": str(device),
        "onsager_correction": True,
        "F_type": "constant_1",
        "evaluation_metric": "cosine_similarity_in_Y_space",
        "shared_per_alpha_graph_noise": True,
        "replica_variation": "student_initialization_only",
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")

    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP / 2, ALPHA_STEP)
    results = {}

    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0

    for alpha_id, alpha in enumerate(alphas):
        shared_seed = SEED + alpha_id * 10000
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=shared_seed,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=NOISE_VAR,
            lam=LAM,
        )

        cosine_similarity_values = []
        loss_values = []
        steps_values = []

        for replica_id in range(NUM_REPLICAS):
            replica_seed = SEED + replica_id * 1000
            t0 = time.time()

            cosine_similarity, final_loss, steps_taken = train_single_replica(
                device=device,
                seed=replica_seed,
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
                lam=LAM,
                shared_data=shared_data,
            )

            dt = time.time() - t0
            cosine_similarity_values.append(cosine_similarity)
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            completed += 1

            print(
                f"α={alpha:.2f}, replica {replica_id + 1}/{NUM_REPLICAS}: "
                f"CosSim={cosine_similarity:.4f}, Loss={final_loss:.2e}, "
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
            f"{r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | {r['steps_mean']:8.0f}"
        )

    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 60)

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
        color="#1976D2",
        markersize=6,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label="Dense-mask G-AMP",
    )
    ax.set_xlabel(r"$\alpha$ (observation density)", fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title(
        f"Dense-mask G-AMP (scalar var + cosine)\n"
        f"({N1}×{N2}, M={M}, {MAX_STEPS} steps)",
        fontsize=16,
    )
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    plot_path = plots_dir / "cosine_similarity_vs_alpha.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        header = (
            "alpha,cosine_similarity_mean,cosine_similarity_std,"
            "Loss_mean,Loss_std,Steps_mean"
        )
        for i in range(NUM_REPLICAS):
            header += f",cosine_similarity_replica_{i}"
        f.write(header + "\n")

        for alpha in alphas_list:
            r = results[alpha]
            line = (
                f"{alpha},{r['cosine_similarity_mean']},{r['cosine_similarity_std']},"
                f"{r['loss_mean']},{r['loss_std']},{r['steps_mean']}"
            )
            for cosine_similarity_value in r["cosine_similarity_values"]:
                line += f",{cosine_similarity_value}"
            f.write(line + "\n")

    print(f"Metrics saved: {csv_path}")
    print(f"\nResults saved to: {results_dir}")
    print("Done!")
