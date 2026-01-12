#!/usr/bin/env python
"""
Finite Size Scaling Analysis for Sparse Matrix Factorization.

This script runs AGD for various system sizes N to study finite-size effects
on the phase transition.

Parameters:
- N_VALUES: List of system sizes to test
- M = N * 0.01 (1% of N)
- N1 = N2 = N
- Cold Start only
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
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

# System sizes to test (modify as needed)
N_VALUES = [100, 300, 1000]  # N1 = N2 = N, M = N * 0.01

# Alpha scan parameters
ALPHA_START = 1.0
ALPHA_STOP = 4.0
ALPHA_STEP = 0.5

# Training parameters
MAX_STEPS = 3000
LR_BASE = 0.01
SEED = 42
NUM_REPLICAS = 3  # Replicas per (N, alpha)
CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions(W, X, i_idx, j_idx):
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1)


def compute_loss(Y, Y_pred):
    return ((Y - Y_pred) ** 2).sum()


def agd_step_W(W, X, Y, i_idx, j_idx, lr):
    N1, M = W.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * residual.unsqueeze(1) * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    return W - lr * grad_W


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


def train_single_replica(N, M, alpha, device, seed):
    """Train a single replica for given N, M, alpha."""
    N1 = N2 = N
    lr = LR_BASE * (1e6 / (N1 * N2))
    
    # Generate teacher
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0, 0.0, 0
    
    # Generate Y
    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx)
    
    # Initialize student (Cold Start)
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    
    # AGD loop
    final_loss = 0.0
    steps_taken = MAX_STEPS
    
    for step in range(MAX_STEPS):
        W_hat = agd_step_W(W_hat, X_hat, Y, i_idx, j_idx, lr)
        X_hat = agd_step_X(W_hat, X_hat, Y, i_idx, j_idx, lr)
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
        
        if step % 100 == 0 or step == MAX_STEPS - 1:
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx)
            loss = compute_loss(Y, Y_pred).item()
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
    print("Finite Size Scaling Analysis")
    print("AGD for Various System Sizes N")
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
    
    print(f"N values: {N_VALUES}")
    print(f"M = N * 0.01")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Replicas per (N, alpha): {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_finite_size"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'N_values': N_VALUES,
        'M_formula': 'N * 0.01',
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'num_replicas': NUM_REPLICAS,
        'max_steps': MAX_STEPS,
        'lr_base': LR_BASE,
        'device': str(device),
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {N: {} for N in N_VALUES}
    
    total_tasks = len(N_VALUES) * len(alphas) * NUM_REPLICAS
    completed = 0
    start_time = time.time()
    
    for N in N_VALUES:
        M = max(1, int(N * 0.01))  # M = 1% of N, at least 1
        print(f"\n--- N = {N}, M = {M} ---")
        
        for alpha in alphas:
            qy_values = []
            for rep in range(NUM_REPLICAS):
                seed = SEED + rep * 1000
                t0 = time.time()
                qy, loss, steps = train_single_replica(N, M, alpha, device, seed)
                dt = time.time() - t0
                qy_values.append(qy)
                completed += 1
                print(f"N={N}, α={alpha:.2f}, rep={rep+1}: Q_Y={qy:.4f} ({dt:.1f}s) [{completed}/{total_tasks}]")
            
            mean_qy = np.mean(qy_values)
            std_qy = np.std(qy_values)
            results[N][alpha] = {'mean': mean_qy, 'std': std_qy, 'values': qy_values}
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Create plot
    print("\nGenerating plot...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Color scheme for different N values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_VALUES)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, N in enumerate(N_VALUES):
        M = max(1, int(N * 0.01))
        label = f'N={N}, M={M}'
        
        alphas_list = sorted(results[N].keys())
        means = [results[N][a]['mean'] for a in alphas_list]
        stds = [results[N][a]['std'] for a in alphas_list]
        sems = [s / np.sqrt(NUM_REPLICAS) for s in stds]
        
        ax.errorbar(alphas_list, means, yerr=sems,
                    fmt=f'{markers[idx % len(markers)]}-', 
                    color=colors[idx],
                    markersize=8, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=label)
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'Finite Size Scaling: Phase Transition vs System Size\n(M = N × 0.01, {NUM_REPLICAS} replicas)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plot_path = plots_dir / "qy_vs_alpha_finite_size.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        header = "N,M,alpha,Q_Y_mean,Q_Y_std"
        f.write(header + "\n")
        
        for N in N_VALUES:
            M = max(1, int(N * 0.01))
            for alpha in sorted(results[N].keys()):
                r = results[N][alpha]
                f.write(f"{N},{M},{alpha},{r['mean']},{r['std']}\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")

# %%
