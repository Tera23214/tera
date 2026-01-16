#!/usr/bin/env python
"""
G-AMP Warm Start vs Cold Start Comparison for Phase Transition Analysis.

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

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph
from terao_gamp_gaussian.F_random.F_random_core.core import gamp_step_with_F, f_input, g_out
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns  
M = 10      # Rank (hidden dimension)

ALPHA_START = 1.5
ALPHA_STOP = 3.5
ALPHA_STEP = 0.25

MAX_STEPS = 500
DAMPING = 0.5
NOISE_VAR = 1e-10
SEED = 42
NUM_REPLICAS = 5

# Epsilon values to compare
# inf = Cold Start (random init), 0 = Perfect Warm Start (start at teacher)
EPSILON_VALUES = [float('inf'), 1.0, 0.5, 0.1, 0.0]

CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# G-AMP Warm Start Training
# ============================================================================

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
    
    # Generate graph (observed entries)
    graph = RandomGraph()
    i_idx, j_idx, E = graph.generate(N1, N2, M, alpha, device, seed)
    
    if E == 0:
        return 0.0, 0.0, 0
    
    # Generate spreading matrix F: (E, M) with F[c,μ] ~ N(0,1)
    torch.manual_seed(seed + 500)
    F = torch.randn(E, M, device=device, dtype=torch.float32)
    F_sq = F ** 2
    
    # Generate observations with F: Y = (1/√M) Σ_μ F[c,μ] W X
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = (1.0 / math.sqrt(M)) * (F * W_sel * X_sel).sum(dim=1)
    
    # Add small noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y) * math.sqrt(NOISE_VAR)
    Y_noisy = Y + noise
    
    # Initialize messages based on epsilon
    torch.manual_seed(seed + 2000)
    
    if epsilon == float('inf'):
        # Cold Start: random initialization (small values)
        m_W = torch.randn(N1, M, device=device) * 0.1
        m_X = torch.randn(M, N2, device=device) * 0.1
        # For cold start, v = E[m²] + variance, with m small, v ≈ 1
        v_W = torch.ones(N1, M, device=device)
        v_X = torch.ones(M, N2, device=device)
    elif epsilon == 0.0:
        # Perfect Warm Start: start exactly at teacher
        m_W = W_teacher.clone()
        m_X = X_teacher.clone()
        # v = E[x²] = m² + variance. For perfect start, set small variance
        v_W = m_W ** 2 + 0.01
        v_X = m_X ** 2 + 0.01
    else:
        # Warm Start: teacher + perturbation
        m_W = W_teacher + epsilon * torch.randn(N1, M, device=device)
        m_X = X_teacher + epsilon * torch.randn(M, N2, device=device)
        m_W = normalize_to_unit_variance(m_W)
        m_X = normalize_to_unit_variance(m_X)
        # v = m² + epsilon² (variance from perturbation)
        v_W = m_W ** 2 + epsilon ** 2
        v_X = m_X ** 2 + epsilon ** 2
    
    g_prev = torch.zeros(E, device=device)
    
    # G-AMP iterations
    final_loss = 0.0
    steps_taken = MAX_STEPS
    prev_loss = float('inf')
    
    for step in range(MAX_STEPS):
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_F(
            m_W, v_W, m_X, v_X,
            Y_noisy, F, F_sq, i_idx, j_idx, g_prev,
            NOISE_VAR, DAMPING, N1, N2, M
        )
        
        # Check convergence
        if step % 50 == 0 or step == MAX_STEPS - 1:
            W_sel_pred = m_W[i_idx.long(), :]
            X_sel_pred = m_X[:, j_idx.long()].T
            Y_pred = (1.0 / math.sqrt(M)) * (F * W_sel_pred * X_sel_pred).sum(dim=1)
            
            loss = ((Y_noisy - Y_pred) ** 2).mean().item()
            final_loss = loss
            
            if abs(prev_loss - loss) < CONVERGENCE_THRESHOLD:
                steps_taken = step + 1
                break
            prev_loss = loss
    
    # Normalize student matrices for Q_Y computation
    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)
    
    # Compute Q_Y
    qy = compute_qy(m_W, m_X, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("G-AMP Warm Start vs Cold Start Comparison")
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
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_gamp_warm_start"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'gamp_warm_start',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'epsilon_values': [str(e) for e in EPSILON_VALUES],
        'num_replicas': NUM_REPLICAS,
        'max_steps': MAX_STEPS,
        'damping': DAMPING,
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
    ax.set_title(f'G-AMP Warm Start Analysis: Phase Transition Type\n({N1}×{N2}, M={M}, {NUM_REPLICAS} replicas)', fontsize=16)
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
