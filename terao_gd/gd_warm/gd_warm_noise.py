#!/usr/bin/env python
"""
Warm Start with Observation Noise.

Two noise sources:
1) Observation noise on Y (sigma_y)
2) Initialization correlation epsilon on W/X

Q_Y is computed against the noise-free teacher matrices.
"""

#%%

import sys
import math
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directory to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000
N2 = 1000
M = 10

ALPHA_START = 0.5
ALPHA_STOP = 7.0
ALPHA_STEP = 0.5

MAX_STEPS = 3000
LR_BASE = 1.0  # Base learning rate (will be scaled by E)
SEED = 42
NUM_REPLICAS = 5

# Observation noise (standard deviation)
SIGMA_Y = 0.1

# Initialization epsilon. Finite values use
# student = epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1).
SIGMA_INIT_VALUES = [float('inf'), 0.0, 0.1, 0.5, 1.0]  # inf = cold start

CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions(W, X, i_idx, j_idx, M):
    """Compute Y_pred = (1/sqrt(M)) * sum_mu W_i,mu X_mu,j for observed edges."""
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1) / math.sqrt(M)


def compute_loss(Y, Y_pred, M):
    """Compute MSE loss with M factor to preserve gradient scale."""
    return M * ((Y - Y_pred) ** 2).sum()


@torch.compile(mode="reduce-overhead")
def agd_step_W(W, X, Y, i_idx, j_idx, lr):
    Y_pred = compute_predictions(W, X, i_idx, j_idx, W.shape[1])
    residual = Y_pred - Y
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * math.sqrt(W.shape[1]) * residual.unsqueeze(1) * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, W.shape[1]), grad_contrib)
    return W - lr * grad_W


@torch.compile(mode="reduce-overhead")
def agd_step_X(W, X, Y, i_idx, j_idx, lr):
    Y_pred = compute_predictions(W, X, i_idx, j_idx, W.shape[1])
    residual = Y_pred - Y
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * math.sqrt(W.shape[1]) * residual.unsqueeze(1) * W_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(W.shape[1], -1), grad_contrib.T)
    return X - lr * grad_X


def normalize_to_unit_mean_square(tensor):
    """Normalize so that mean square equals 1."""
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq)


def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """Compute Q_Y overlap using theoretical normalization."""
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student
    inner_product = (Y_teacher * Y_student).sum()
    return (inner_product / (N1 * N2 * M)).item()


def train_single_replica(alpha, sigma_init, sigma_y, device, seed):
    """Train a single replica with observation noise and warm-start noise."""
    # Generate teacher
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)

    # Generate graph
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    if C == 0:
        return 0.0, 0.0, 0

    # Generate clean observations
    Y_clean = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)

    # Add observation noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y_clean) * sigma_y
    Y_noisy = Y_clean + noise

    # Initialize student
    torch.manual_seed(seed + 2000)
    if math.isinf(sigma_init):
        W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
        X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    else:
        if not 0.0 <= sigma_init <= 1.0:
            raise ValueError(f"epsilon_init must satisfy 0 <= epsilon <= 1, got {sigma_init}.")
        noise_scale = math.sqrt(sigma_init - sigma_init * sigma_init)
        W_hat = sigma_init * W_teacher + noise_scale * torch.randn(
            N1, M, device=device, dtype=torch.float32
        )
        X_hat = sigma_init * X_teacher + noise_scale * torch.randn(
            M, N2, device=device, dtype=torch.float32
        )

    # AGD loop with LR scaled by sqrt of edge count
    # Using sqrt(C) instead of C for better balance between convergence and stability
    lr_scaled = LR_BASE / math.sqrt(C)
    final_loss = 0.0
    steps_taken = MAX_STEPS

    for step in range(MAX_STEPS):
        W_hat = agd_step_W(W_hat, X_hat, Y_noisy, i_idx, j_idx, lr_scaled)
        X_hat = agd_step_X(W_hat, X_hat, Y_noisy, i_idx, j_idx, lr_scaled)
        W_hat = normalize_to_unit_mean_square(W_hat)
        X_hat = normalize_to_unit_mean_square(X_hat)

        if step % 100 == 0 or step == MAX_STEPS - 1:
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx, M)
            loss = compute_loss(Y_noisy, Y_pred, M).item()
            final_loss = loss
            if loss < CONVERGENCE_THRESHOLD:
                steps_taken = step + 1
                break

    qy = compute_qy(W_hat, X_hat, W_teacher, X_teacher)
    return qy, final_loss, steps_taken


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Warm Start with Observation Noise")
    print("Phase Transition Analysis")
    print("=" * 60)

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        print("Using: CPU")

    print(f"Matrix: {N1}x{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Sigma_Y: {SIGMA_Y}")
    print(f"Epsilon_init values: {SIGMA_INIT_VALUES}")
    print(f"Replicas per (alpha, epsilon_init): {NUM_REPLICAS}")
    print()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_warm_noise_sigmaY{SIGMA_Y}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Save configuration
    config = {
        "N1": N1,
        "N2": N2,
        "M": M,
        "alpha_start": ALPHA_START,
        "alpha_stop": ALPHA_STOP,
        "alpha_step": ALPHA_STEP,
        "sigma_y": SIGMA_Y,
        "epsilon_init_values": [str(s) for s in SIGMA_INIT_VALUES],
        "student_init_formula": "epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1)",
        "num_replicas": NUM_REPLICAS,
        "max_steps": MAX_STEPS,
        "device": str(device),
    }
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP / 2, ALPHA_STEP)
    results = {s: {} for s in SIGMA_INIT_VALUES}

    total_tasks = len(alphas) * len(SIGMA_INIT_VALUES) * NUM_REPLICAS
    completed = 0
    start_time = time.time()

    for sigma_init in SIGMA_INIT_VALUES:
        print(f"\n--- epsilon_init = {sigma_init} ---")
        for alpha in alphas:
            qy_values = []
            for rep in range(NUM_REPLICAS):
                seed = SEED + rep * 1000
                qy, loss, steps = train_single_replica(alpha, sigma_init, SIGMA_Y, device, seed)
                qy_values.append(qy)
                completed += 1

            mean_qy = np.mean(qy_values)
            std_qy = np.std(qy_values)
            results[sigma_init][alpha] = {"mean": mean_qy, "std": std_qy, "values": qy_values}

            print(
                f"epsilon_init={sigma_init}, alpha={alpha:.2f}: "
                f"Q_Y = {mean_qy:.4f} +- {std_qy:.4f} [{completed}/{total_tasks}]"
            )

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    # Plot
    print("\nGenerating plots...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, sigma_init in enumerate(SIGMA_INIT_VALUES):
        alphas_list = sorted(results[sigma_init].keys())
        means = [results[sigma_init][a]["mean"] for a in alphas_list]
        stds = [results[sigma_init][a]["std"] for a in alphas_list]
        sems = [s / np.sqrt(NUM_REPLICAS) for s in stds]

        ax.errorbar(
            alphas_list,
            means,
            yerr=sems,
            fmt=f"{markers[idx % len(markers)]}-",
            color=colors[idx % len(colors)],
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            label=f"epsilon_init = {sigma_init}",
        )

    ax.set_xlabel(r"$\alpha$ (observation density)", fontsize=14)
    ax.set_ylabel(r"$Q_Y$ (reconstruction quality)", fontsize=14)
    ax.set_title(
        f"Warm Start with Observation Noise\n"
        f"({N1}x{N2}, M={M}, sigma_y={SIGMA_Y}, {NUM_REPLICAS} replicas)",
        fontsize=16,
    )
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    plot_path = plots_dir / f"qy_vs_alpha_sigmaY{SIGMA_Y}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.show()

    # Save CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        header = "alpha"
        for s in SIGMA_INIT_VALUES:
            header += f",Q_Y_mean_sigmaInit{s},Q_Y_std_sigmaInit{s}"
        f.write(header + "\n")

        for alpha in sorted(alphas):
            line = f"{alpha}"
            for s in SIGMA_INIT_VALUES:
                r = results[s].get(alpha, {"mean": 0, "std": 0})
                line += f",{r['mean']},{r['std']}"
            f.write(line + "\n")

    print(f"Metrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")

# %%
