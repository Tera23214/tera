#!/usr/bin/env python
"""
Warm Start vs Cold Start Comparison for Phase Transition Analysis.

This script compares different initialization strategies (epsilon values)
to investigate whether the phase transition is first-order or second-order.

- epsilon = inf: Cold Start (random initialization)
- epsilon = 0: Perfect Warm Start (start exactly at teacher)
- epsilon > 0: Warm Start with perturbation

If curves differ significantly, it suggests first-order transition (hysteresis).
If curves overlap, it suggests second-order transition (continuous).
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

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns  
M = 10      # Rank (hidden dimension)

ALPHA_START = 1.5
ALPHA_STOP = 3.5
ALPHA_STEP = 0.25

MAX_STEPS = 3000
LR_BASE = 0.01
LR = LR_BASE * (1e6 / (N1 * N2))
SEED = 42
NUM_REPLICAS = 5

# Epsilon values to compare
# inf = Cold Start (random init), 0 = Perfect Warm Start (start at teacher)
EPSILON_VALUES = [float('inf'), 1.0, 0.5, 0.1, 0.0]

CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions(W, X, i_idx, j_idx, M):
    """Compute Y_pred = (1/√M) * sum_mu W_iμ X_μj"""
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1) / math.sqrt(M)


def compute_loss(Y, Y_pred, M):
    """Compute MSE loss with M factor to preserve gradient scale"""
    return M * ((Y - Y_pred) ** 2).sum()


@torch.compile(mode="reduce-overhead")
def agd_step_W(W, X, Y, i_idx, j_idx, lr):
    N1, M = W.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    return W - lr * grad_W


@torch.compile(mode="reduce-overhead")
def agd_step_X(W, X, Y, i_idx, j_idx, lr):
    M, N2 = X.shape
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * W_sel
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


def train_single_replica(
    alpha: float,
    epsilon: float,
    device: torch.device,
    seed: int,
):
    """
    Train a single replica with given epsilon (initialization strategy).
    
    Args:
        alpha: Observation density
        epsilon: Perturbation strength
            - inf: Cold Start (random initialization)
            - 0: Start exactly at teacher
            - >0: Teacher + epsilon * noise
        device: torch device
        seed: Random seed
    
    Returns:
        tuple: (qy, final_loss, steps_taken)
    """
    # Generate teacher
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0, 0.0, 0
    
    # Generate Y (observations)
    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)
    
    # Initialize student based on epsilon
    torch.manual_seed(seed + 2000)
    
    if epsilon == float('inf'):
        # Cold Start: random N(0,1) initialization
        W_hat = torch.randn(N1, M, device=device, dtype=torch.float32)
        X_hat = torch.randn(M, N2, device=device, dtype=torch.float32)
    elif epsilon == 0.0:
        # Perfect Warm Start: exactly at teacher
        W_hat = W_teacher.clone()
        X_hat = X_teacher.clone()
    else:
        # Warm Start: teacher + perturbation
        W_hat = W_teacher + epsilon * torch.randn(N1, M, device=device, dtype=torch.float32)
        X_hat = X_teacher + epsilon * torch.randn(M, N2, device=device, dtype=torch.float32)
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
    
    # AGD loop
    final_loss = 0.0
    steps_taken = MAX_STEPS
    
    for step in range(MAX_STEPS):
        W_hat = agd_step_W(W_hat, X_hat, Y, i_idx, j_idx, LR)
        X_hat = agd_step_X(W_hat, X_hat, Y, i_idx, j_idx, LR)
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
        
        if step % 100 == 0 or step == MAX_STEPS - 1:
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx, M)
            loss = compute_loss(Y, Y_pred, M).item()
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
    print("Warm Start vs Cold Start Comparison")
    print("Phase Transition Type Investigation")
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
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Replicas per (alpha, epsilon): {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_warm_start"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'epsilon_values': [str(e) for e in EPSILON_VALUES],
        'num_replicas': NUM_REPLICAS,
        'max_steps': MAX_STEPS,
        'device': str(device),
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {eps: {} for eps in EPSILON_VALUES}
    
    total_tasks = len(alphas) * len(EPSILON_VALUES) * NUM_REPLICAS
    completed = 0
    start_time = time.time()
    
    for eps in EPSILON_VALUES:
        eps_label = "inf" if eps == float('inf') else f"{eps}"
        print(f"\n--- Epsilon = {eps_label} ---")
        
        for alpha in alphas:
            qy_values = []
            for rep in range(NUM_REPLICAS):
                seed = SEED + rep * 1000
                qy, loss, steps = train_single_replica(alpha, eps, device, seed)
                qy_values.append(qy)
                completed += 1
            
            mean_qy = np.mean(qy_values)
            std_qy = np.std(qy_values)
            results[eps][alpha] = {'mean': mean_qy, 'std': std_qy, 'values': qy_values}
            
            print(f"ε={eps_label}, α={alpha:.2f}: Q_Y = {mean_qy:.4f} ± {std_qy:.4f} [{completed}/{total_tasks}]")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Create plots
    print("\nGenerating plots...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Color scheme
    colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, eps in enumerate(EPSILON_VALUES):
        eps_label = "∞ (Cold)" if eps == float('inf') else f"{eps}" if eps > 0 else "0 (Teacher)"
        
        alphas_list = sorted(results[eps].keys())
        means = [results[eps][a]['mean'] for a in alphas_list]
        stds = [results[eps][a]['std'] for a in alphas_list]
        sems = [s / np.sqrt(NUM_REPLICAS) for s in stds]
        
        ax.errorbar(alphas_list, means, yerr=sems,
                    fmt=f'{markers[idx]}-', color=colors[idx],
                    markersize=8, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=f'ε = {eps_label}')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'Warm Start Analysis: Phase Transition Type\n({N1}×{N2}, M={M}, {NUM_REPLICAS} replicas)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plot_path = plots_dir / "qy_vs_alpha_epsilon_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        header = "alpha"
        for eps in EPSILON_VALUES:
            eps_str = "inf" if eps == float('inf') else f"{eps}"
            header += f",Q_Y_mean_eps{eps_str},Q_Y_std_eps{eps_str}"
        f.write(header + "\n")
        
        for alpha in sorted(alphas):
            line = f"{alpha}"
            for eps in EPSILON_VALUES:
                r = results[eps].get(alpha, {'mean': 0, 'std': 0})
                line += f",{r['mean']},{r['std']}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")

# %%
