#!/usr/bin/env python
"""
M-Independence Test: Verify that critical point α_c is independent of rank M.

This simulation runs AGD for multiple M values with alpha scans,
then plots all curves together to show that α_c is the same for all M.

Based on Dense Limit theory: α_c should not depend on M when N >> M >> 1.
"""

import sys
import math
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns

# Multiple M values to test
M_VALUES = [5, 10, 20, 30]

# Alpha scan range (focused around expected critical region)
ALPHA_START = 1.5
ALPHA_STOP = 4.0
ALPHA_STEP = 0.25

MAX_STEPS = 10000
LR_BASE = 0.01
SEED = 42
NUM_REPLICAS = 5  # Replicas per (M, alpha) pair
CONVERGENCE_THRESHOLD = 1e-6  # Early stopping threshold

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions(W, X, i_idx, j_idx):
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1)


def compute_loss(Y, Y_pred):
    return ((Y - Y_pred) ** 2).sum()


@torch.compile(mode="reduce-overhead")
def agd_step_W(W, X, Y, i_idx, j_idx, lr):
    N1, M = W.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * residual.unsqueeze(1) * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    return W - lr * grad_W


@torch.compile(mode="reduce-overhead")
def agd_step_X(W, X, Y, i_idx, j_idx, lr):
    M, N2 = X.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * residual.unsqueeze(1) * W_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    return X - lr * grad_X


def normalize_to_unit_variance(tensor):
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq)


def compute_qy(W_student, X_student, W_teacher, X_teacher):
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student
    inner_product = (Y_teacher * Y_student).sum()
    return (inner_product / (N1 * N2 * M)).item()


def train_single_replica(alpha, M, device, seed):
    """Train a single replica for given alpha and M."""
    # Scale learning rate by M (larger M needs smaller lr)
    lr = LR_BASE * (1e6 / (N1 * N2)) / M
    
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0
    
    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx)
    
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    
    for step in range(MAX_STEPS):
        W_hat = agd_step_W(W_hat, X_hat, Y, i_idx, j_idx, lr)
        X_hat = agd_step_X(W_hat, X_hat, Y, i_idx, j_idx, lr)
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
        
        # Early stopping: check every 100 steps
        if step % 100 == 0 and step > 0:
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx)
            loss = compute_loss(Y, Y_pred).item()
            if loss < CONVERGENCE_THRESHOLD:
                break
    
    return compute_qy(W_hat, X_hat, W_teacher, X_teacher)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("M-Independence Test: α_c vs M")
    print("Verifying Dense Limit prediction")
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
    
    print(f"Matrix: {N1}×{N2}")
    print(f"M values: {M_VALUES}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Replicas per (M, α): {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_m_independence"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'N1': N1, 'N2': N2,
        'M_values': M_VALUES,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'max_steps': MAX_STEPS,
        'num_replicas': NUM_REPLICAS,
        'device': str(device),
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    all_results = {}  # {M: {alpha: {'mean': ..., 'std': ...}}}
    
    total_tasks = len(M_VALUES) * len(alphas) * NUM_REPLICAS
    completed = 0
    start_time = time.time()
    
    for M in M_VALUES:
        print(f"\n--- M = {M} ---")
        all_results[M] = {}
        
        for alpha in alphas:
            qy_values = []
            for rep in range(NUM_REPLICAS):
                seed = SEED + rep * 1000 + M * 100
                qy = train_single_replica(alpha, M, device, seed)
                qy_values.append(qy)
                completed += 1
            
            mean_qy = np.mean(qy_values)
            std_qy = np.std(qy_values)
            all_results[M][alpha] = {'mean': mean_qy, 'std': std_qy, 'values': qy_values}
            
            print(f"M={M:2d}, α={alpha:.2f}: Q_Y = {mean_qy:.4f} ± {std_qy:.4f} [{completed}/{total_tasks}]")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot all M curves together
    print("\nGenerating plots...")
    
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, M in enumerate(M_VALUES):
        alphas_list = sorted(all_results[M].keys())
        means = [all_results[M][a]['mean'] for a in alphas_list]
        stds = [all_results[M][a]['std'] for a in alphas_list]
        sems = [s / np.sqrt(NUM_REPLICAS) for s in stds]
        
        ax.errorbar(alphas_list, means, yerr=sems,
                    fmt=f'{markers[idx]}-', color=colors[idx],
                    markersize=8, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=f'M = {M}')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'Phase Transition for Different M Values\n({N1}×{N2}, {NUM_REPLICAS} replicas each)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label=r'$Q_Y = 0.5$ (critical)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "qy_vs_alpha_multi_M.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plots_dir / 'qy_vs_alpha_multi_M.png'}")
    plt.show()
    
    # Estimate critical points (α where Q_Y crosses 0.5)
    print("\n" + "=" * 60)
    print("Estimated Critical Points")
    print("=" * 60)
    
    critical_points = {}
    for M in M_VALUES:
        alphas_list = sorted(all_results[M].keys())
        means = [all_results[M][a]['mean'] for a in alphas_list]
        
        # Find α where Q_Y crosses 0.5 (linear interpolation)
        alpha_c = None
        for i in range(len(means) - 1):
            if means[i] < 0.5 <= means[i+1]:
                # Linear interpolation
                alpha_c = alphas_list[i] + (0.5 - means[i]) / (means[i+1] - means[i]) * ALPHA_STEP
                break
        
        critical_points[M] = alpha_c
        if alpha_c:
            print(f"M = {M:2d}: α_c ≈ {alpha_c:.3f}")
        else:
            print(f"M = {M:2d}: α_c not found in range")
    
    # Save results to CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        f.write("M,alpha,Q_Y_mean,Q_Y_std\n")
        for M in M_VALUES:
            for alpha in sorted(all_results[M].keys()):
                r = all_results[M][alpha]
                f.write(f"{M},{alpha},{r['mean']},{r['std']}\n")
    
    # Save critical points
    with open(results_dir / "critical_points.csv", 'w') as f:
        f.write("M,alpha_c\n")
        for M, alpha_c in critical_points.items():
            f.write(f"{M},{alpha_c if alpha_c else 'NA'}\n")
    
    print(f"\nMetrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")
