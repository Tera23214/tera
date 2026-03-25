#!/usr/bin/env python
"""
Noisy AGD: Alternating Gradient Descent with Noisy Observations.

This script adds Gaussian noise to the teacher's observations
and measures how the noise level affects student performance.

Model:
    Y_observed = Y_true + ε,  where ε ~ N(0, σ²)

Key concept:
- α (observation density) is FIXED
- Noise variance σ² is swept
- Measures Q_Y (student-teacher overlap)

Expected results:
- Low noise: Q_Y ≈ 1 (good recovery)
- High noise: Q_Y degrades

Loss function: L = sum((Y_noisy - Y_pred)^2)
Optimization: Alternating updates of W and X using gradient descent.
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

# Add parent directory to path (to get smf modules)
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns  
M = 10      # Rank (hidden dimension)

# Fixed alpha
ALPHA = 5.0  # Observation density (fix this, vary noise instead)

# Noise variance sweep
NOISE_VAR_START = 0.0
NOISE_VAR_STOP = 0.01
NOISE_VAR_STEP = 0.002

MAX_STEPS = 3000
LR = 0.01          # Learning rate
SEED = 42
NUM_REPLICAS = 10  # Number of replicas per noise level

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
    M: int,                # Rank for 1/√M scaling
) -> torch.Tensor:
    """
    Compute predictions Y_pred for observed entries.
    
    Y_pred[c] = (1/√M) * sum_mu W[i_c, mu] * X[mu, j_c]
    """
    W_sel = W[i_idx.long(), :]       # (C, M)
    X_sel = X[:, j_idx.long()].T     # (C, M)
    
    Y_pred = (W_sel * X_sel).sum(dim=1) / math.sqrt(M)  # (C,)
    return Y_pred


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """Compute MSE loss: L = M * sum((Y - Y_pred)^2)"""
    return M * ((Y - Y_pred) ** 2).sum()


@torch.compile(mode="reduce-overhead")
def agd_step_W(
    W: torch.Tensor,   # (N1, M)
    X: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,   # (C,)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Gradient descent step for W (fixing X)."""
    N1, M = W.shape
    
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y
    
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * X_sel
    
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    
    W_new = W - lr * grad_W
    return W_new


@torch.compile(mode="reduce-overhead")
def agd_step_X(
    W: torch.Tensor,   # (N1, M)
    X: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,   # (C,)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Gradient descent step for X (fixing W)."""
    M, N2 = X.shape
    
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y
    
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * W_sel
    
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    
    X_new = X - lr * grad_X
    return X_new


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor so that mean square equals 1."""
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq)


def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """
    Compute Q_Y overlap using theoretical normalization.
    Q_Y = <Y_teacher, Y_student> / (N1 * N2 * M)
    """
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student
    
    inner_product = (Y_teacher * Y_student).sum()
    
    return (inner_product / (N1 * N2 * M)).item()


def train_single_replica_noisy(
    noise_var: float,
    device: torch.device,
    seed: int,
):
    """
    Train a single replica with noisy observations.
    """
    # Generate teacher for this replica
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph (observed entries)
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, ALPHA, device, seed)
    
    if C == 0:
        return 0.0
    
    # Generate TRUE Y (without noise)
    Y_true = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)
    
    # Add Gaussian noise: Y_noisy = Y_true + ε, ε ~ N(0, noise_var)
    torch.manual_seed(seed + 500)
    noise = torch.randn_like(Y_true) * math.sqrt(noise_var)
    Y_noisy = Y_true + noise
    
    # Initialize student randomly
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    
    # Alternating Gradient Descent loop (using NOISY observations)
    for step in range(MAX_STEPS):
        W_hat = agd_step_W(W_hat, X_hat, Y_noisy, i_idx, j_idx, LR)
        X_hat = agd_step_X(W_hat, X_hat, Y_noisy, i_idx, j_idx, LR)
        
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
    
    # Compute Q_Y (comparing with TRUE teacher, not noisy)
    qy = compute_qy(W_hat, X_hat, W_teacher, X_teacher)
    
    return qy


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Noisy AGD: Effect of Observation Noise")
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
    
    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Alpha (fixed): {ALPHA}")
    print(f"Noise σ²: {NOISE_VAR_START} ~ {NOISE_VAR_STOP} (step {NOISE_VAR_STEP})")
    print(f"Steps: {MAX_STEPS}, LR={LR}")
    print(f"Replicas per noise level: {NUM_REPLICAS}")
    print()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = f"{timestamp}_agd_noisy_{N1}x{M}_alpha{ALPHA}"
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'agd_noisy',
        'N1': N1,
        'N2': N2,
        'M': M,
        'alpha': ALPHA,
        'noise_var_start': NOISE_VAR_START,
        'noise_var_stop': NOISE_VAR_STOP,
        'noise_var_step': NOISE_VAR_STEP,
        'max_steps': MAX_STEPS,
        'lr': LR,
        'seed': SEED,
        'num_replicas': NUM_REPLICAS,
        'device': str(device),
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")
    
    # Run simulations
    noise_vars = np.arange(NOISE_VAR_START, NOISE_VAR_STOP + NOISE_VAR_STEP/2, NOISE_VAR_STEP)
    results = {}
    
    start_time = time.time()
    total_tasks = len(noise_vars) * NUM_REPLICAS
    completed = 0
    
    for nv in noise_vars:
        qy_values = []
        for replica_id in range(NUM_REPLICAS):
            seed = SEED + replica_id * 1000
            t0 = time.time()
            qy = train_single_replica_noisy(nv, device, seed)
            dt = time.time() - t0
            qy_values.append(qy)
            completed += 1
            print(f"σ²={nv:.2e}, replica {replica_id+1}/{NUM_REPLICAS}: Q_Y={qy:.4f} ({dt:.1f}s) [{completed}/{total_tasks}]")
        
        results[nv] = {
            'mean': np.mean(qy_values),
            'std': np.std(qy_values),
            'values': qy_values
        }
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results (mean ± std)")
    print("=" * 60)
    for nv in sorted(results.keys()):
        r = results[nv]
        print(f"σ²={nv:.2e}: Q_Y = {r['mean']:.4f} ± {r['std']:.4f}")
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 60)
    
    # Plot Q_Y vs Noise Variance with error bars
    print("\nGenerating plots...")
    
    noise_list = sorted(results.keys())
    qy_means = [results[n]['mean'] for n in noise_list]
    qy_stds = [results[n]['std'] for n in noise_list]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(noise_list, qy_means, yerr=qy_stds,
                fmt='o-', color='#E53935', markersize=6, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5)
    ax.set_xlabel(r'Noise variance $\sigma^2$', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (student-teacher overlap)', fontsize=14)
    ax.set_title(f'Effect of Observation Noise (AGD)\n({N1}×{N2}, M={M}, α={ALPHA}, {NUM_REPLICAS} replicas)', fontsize=16)
    
    # Dynamic x-axis limits
    x_margin = max(NOISE_VAR_STEP * 0.5, (NOISE_VAR_STOP - NOISE_VAR_START) * 0.05)
    ax.set_xlim(NOISE_VAR_START - x_margin, NOISE_VAR_STOP + x_margin)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create plots subdirectory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Save plot
    plot_path = plots_dir / "qy_vs_noise.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save results to CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        header = "noise_var,Q_Y_mean,Q_Y_std"
        for i in range(NUM_REPLICAS):
            header += f",replica_{i}"
        f.write(header + "\n")
        
        for nv in noise_list:
            r = results[nv]
            line = f"{nv},{r['mean']},{r['std']}"
            for v in r['values']:
                line += f",{v}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"\nResults saved to: {results_dir}")
    
    print("Done!")

# %%
