#!/usr/bin/env python
"""
AGD Alpha Continuation Method for Metastable State Analysis.

This script implements the α-continuation method with AGD:
1. Start from teacher matrices at low α (or random at high α)
2. Gradually change α (add/remove observations)
3. Use previous solution as initial condition

This allows observation of metastable states and hysteresis behavior.
AGD directly optimizes W, X so the continuation is more stable than G-AMP.
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

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 2000    # Small system for testing
N2 = 2000
M = 20

ALPHA_MIN = 0.5
ALPHA_MAX = 4.0
ALPHA_STEPS = 19  # Number of alpha points

MAX_STEPS_PER_ALPHA = 2000  # Maximum steps (early stopping may finish sooner)
LR = 0.001
SEED = 42
NUM_REPLICAS = 5

# Early stopping parameters
CONVERGENCE_THRESHOLD = 1e-8  # Stop when loss change is below this
CONVERGENCE_CHECK_INTERVAL = 20  # Check every N steps

# Initialization noise levels (distance from teacher)
# 0 = exact teacher, inf = random initialization
SIGMA_INIT_VALUES = [0.0, 0.1, 0.5, 1.0, float('inf')]

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions_with_F(W, X, F, i_idx, j_idx, M):
    """Compute Y_pred = (1/√M) Σ_μ F[c,μ] W[i,μ] X[μ,j]"""
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)


def normalize_to_unit_variance(tensor):
    """Normalize tensor so that mean square equals 1."""
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq + 1e-10)


def compute_qy(W_s, X_s, W_t, X_t):
    """Compute Q_Y overlap."""
    N1, M = W_t.shape
    N2 = X_t.shape[1]
    Y_t = W_t @ X_t
    Y_s = W_s @ X_s
    return ((Y_t * Y_s).sum() / (N1 * N2 * M)).item()


def agd_step_W_with_F(W, X, Y, F, i_idx_long, j_idx_long, i_idx_exp, lr):
    """AGD step for W with F (using pre-computed indices)."""
    N1, M = W.shape
    W_sel = W[i_idx_long, :]
    X_sel = X[:, j_idx_long].T
    Y_pred = (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)
    residual = Y_pred - Y
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * F * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx_exp, grad_contrib)
    return W - lr * grad_W


def agd_step_X_with_F(W, X, Y, F, i_idx_long, j_idx_long, j_idx_exp, lr):
    """AGD step for X with F (using pre-computed indices)."""
    M, N2 = X.shape
    W_sel = W[i_idx_long, :]
    X_sel = X[:, j_idx_long].T
    Y_pred = (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)
    residual = Y_pred - Y
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * F * W_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx_exp, grad_contrib.T)
    return X - lr * grad_X


# ============================================================================
# Alpha Continuation Training
# ============================================================================

def train_continuation(
    sigma_init: float,  # Initialization noise: 0=teacher, inf=random
    device: torch.device,
    seed: int,
):
    """
    Train using α-continuation method with AGD.
    
    Args:
        sigma_init: Initialization noise standard deviation
            - 0: Start exactly at teacher
            - >0: Start at teacher + sigma_init * noise
            - inf: Random initialization
        device: torch device
        seed: Random seed
    
    Returns:
        dict: {alpha: qy} results
    """
    # Generate alpha sequence (always ascending)
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_STEPS)
    
    # Generate teacher matrices
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate full graph at maximum alpha
    graph = RandomGraph()
    i_idx_full, j_idx_full, C_max = graph.generate(N1, N2, M, ALPHA_MAX, device, seed)
    
    if C_max == 0:
        return {}
    
    # Generate full F and Y at maximum alpha
    torch.manual_seed(seed + 500)
    F_full = torch.randn(C_max, M, device=device, dtype=torch.float32)
    
    # Compute Y for full graph
    W_sel_full = W_teacher[i_idx_full.long(), :]
    X_sel_full = X_teacher[:, j_idx_full.long()].T
    Y_full = (1.0 / math.sqrt(M)) * (F_full * W_sel_full * X_sel_full).sum(dim=1)
    
    # Initialize W, X based on sigma_init
    torch.manual_seed(seed + 2000)
    
    if math.isinf(sigma_init):
        # Random initialization (cold start)
        W = torch.randn(N1, M, device=device) * 0.01
        X = torch.randn(M, N2, device=device) * 0.01
    elif sigma_init == 0.0:
        # Exact teacher (perfect warm start)
        W = W_teacher.clone()
        X = X_teacher.clone()
    else:
        # Teacher + noise
        W = W_teacher + sigma_init * torch.randn(N1, M, device=device)
        X = X_teacher + sigma_init * torch.randn(M, N2, device=device)
        W = normalize_to_unit_variance(W)
        X = normalize_to_unit_variance(X)
    
    results = {}
    
    for alpha in alphas:
        # Compute number of observations for this alpha
        C = max(1, int((alpha / ALPHA_MAX) * C_max))
        
        # Use subset of observations
        i_idx = i_idx_full[:C]
        j_idx = j_idx_full[:C]
        F = F_full[:C]
        Y = Y_full[:C]
        
        # Pre-compute indices for this alpha (optimization)
        i_idx_long = i_idx.long()
        j_idx_long = j_idx.long()
        i_idx_exp = i_idx_long.unsqueeze(1).expand(-1, M).contiguous()
        j_idx_exp = j_idx_long.unsqueeze(0).expand(M, -1).contiguous()
        
        # Run AGD for this alpha with early stopping
        prev_loss = float('inf')
        for step in range(MAX_STEPS_PER_ALPHA):
            W = agd_step_W_with_F(W, X, Y, F, i_idx_long, j_idx_long, i_idx_exp, LR)
            X = agd_step_X_with_F(W, X, Y, F, i_idx_long, j_idx_long, j_idx_exp, LR)
            W = normalize_to_unit_variance(W)
            X = normalize_to_unit_variance(X)
            
            # Early stopping check
            if step % CONVERGENCE_CHECK_INTERVAL == 0:
                W_sel = W[i_idx_long, :]
                X_sel = X[:, j_idx_long].T
                Y_pred = (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)
                loss = ((Y - Y_pred) ** 2).mean().item()
                if abs(prev_loss - loss) < CONVERGENCE_THRESHOLD:
                    break
                prev_loss = loss
        
        # Compute Q_Y at this alpha
        qy = compute_qy(W, X, W_teacher, X_teacher)
        results[alpha] = qy
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGD Alpha Continuation Method")
    print("Metastable State Analysis")
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
    print(f"Alpha: {ALPHA_MIN} ~ {ALPHA_MAX} ({ALPHA_STEPS} points)")
    print(f"Max steps per alpha: {MAX_STEPS_PER_ALPHA}, LR={LR}")
    print(f"Replicas: {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_agd_continuation"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'agd_continuation',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_min': ALPHA_MIN,
        'alpha_max': ALPHA_MAX,
        'alpha_steps': ALPHA_STEPS,
        'max_steps_per_alpha': MAX_STEPS_PER_ALPHA,
        'lr': LR,
        'num_replicas': NUM_REPLICAS,
        'sigma_init_values': [str(s) for s in SIGMA_INIT_VALUES],
        'device': str(device),
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    start_time = time.time()
    
    # Results storage: {sigma_init: {alpha: [qy values]}}
    all_results = {s: {} for s in SIGMA_INIT_VALUES}
    
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_STEPS)
    for s in SIGMA_INIT_VALUES:
        for alpha in alphas:
            all_results[s][alpha] = []
    
    # Run for each sigma_init (sequential - parallel was slower on MPS)
    total_tasks = len(SIGMA_INIT_VALUES) * NUM_REPLICAS
    completed = 0
    
    for sigma_init in SIGMA_INIT_VALUES:
        s_label = "∞" if math.isinf(sigma_init) else f"{sigma_init}"
        print(f"\n--- σ_init = {s_label} ---")
        
        for rep in range(NUM_REPLICAS):
            seed = SEED + rep * 1000
            results = train_continuation(sigma_init, device, seed)
            for alpha, qy in results.items():
                all_results[sigma_init][alpha].append(qy)
            completed += 1
            print(f"Replica {rep+1}/{NUM_REPLICAS} done [{completed}/{total_tasks}]")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Compute statistics
    alphas_list = sorted(alphas)
    stats = {}
    for s in SIGMA_INIT_VALUES:
        stats[s] = {
            'means': [np.mean(all_results[s][a]) for a in alphas_list],
            'stds': [np.std(all_results[s][a]) for a in alphas_list],
            'sems': [np.std(all_results[s][a]) / np.sqrt(NUM_REPLICAS) for a in alphas_list],
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("Results (mean ± SEM)")
    print("=" * 80)
    header = f"{'Alpha':>6}"
    for s in SIGMA_INIT_VALUES:
        s_label = "inf" if math.isinf(s) else f"{s}"
        header += f" | σ={s_label:^8}"
    print(header)
    print("-" * 80)
    for i, alpha in enumerate(alphas_list):
        line = f"{alpha:6.2f}"
        for s in SIGMA_INIT_VALUES:
            line += f" | {stats[s]['means'][i]:8.4f}"
        print(line)
    print("=" * 80)
    
    # Create plots
    print("\nGenerating plots...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#8E24AA', '#1E88E5', '#43A047', '#FB8C00', '#E53935']
    markers = ['v', 'D', '^', 's', 'o']
    
    for idx, sigma_init in enumerate(SIGMA_INIT_VALUES):
        s_label = "∞ (Random)" if math.isinf(sigma_init) else f"{sigma_init}" if sigma_init > 0 else "0 (Teacher)"
        
        ax.errorbar(alphas_list, stats[sigma_init]['means'], 
                    yerr=stats[sigma_init]['sems'],
                    fmt=f'{markers[idx]}-', color=colors[idx], 
                    markersize=8, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=f'σ_init = {s_label}')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'AGD Continuation Method: Multiple Initialization Noise Levels\n'
                 f'({N1}×{N2}, M={M}, max {MAX_STEPS_PER_ALPHA} steps/α, {NUM_REPLICAS} replicas)', 
                 fontsize=16)
    ax.set_xlim(ALPHA_MIN - 0.1, ALPHA_MAX + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plot_path = plots_dir / "qy_vs_alpha_sigma_init.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        header = "alpha"
        for s in SIGMA_INIT_VALUES:
            s_str = "inf" if math.isinf(s) else f"{s}"
            header += f",mean_s{s_str},std_s{s_str}"
        f.write(header + "\n")
        
        for i, alpha in enumerate(alphas_list):
            line = f"{alpha}"
            for s in SIGMA_INIT_VALUES:
                line += f",{stats[s]['means'][i]},{stats[s]['stds'][i]}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")

# %%

