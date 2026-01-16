#!/usr/bin/env python
"""
G-AMP Alpha Continuation Method for Metastable State Analysis.

This script implements the α-continuation method:
1. Start from teacher matrices at low α
2. Gradually increase α (add more observations)
3. Use previous solution as initial condition

This allows observation of metastable states and hysteresis behavior.
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
from terao_gamp_gaussian.F_random.F_random_core.core import gamp_step_with_F
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000
N2 = 1000
M = 10

ALPHA_MIN = 0.5
ALPHA_MAX = 5.0
ALPHA_STEPS = 19  # Number of alpha points

STEPS_PER_ALPHA = 500  # Steps at each alpha level - needs enough for convergence
DAMPING = 0.5
NOISE_VAR = 1e-10
SEED = 42
NUM_REPLICAS = 5

# ============================================================================
# Alpha Continuation Training
# ============================================================================

def train_continuation(
    direction: str,  # "ascending" or "descending"
    device: torch.device,
    seed: int,
):
    """
    Train using α-continuation method.
    
    Args:
        direction: "ascending" (low→high α) or "descending" (high→low α)
        device: torch device
        seed: Random seed
    
    Returns:
        dict: {alpha: qy} results
    """
    # Generate alpha sequence
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_STEPS)
    if direction == "descending":
        alphas = alphas[::-1]
    
    # Generate teacher matrices
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate full graph at maximum alpha
    # We'll use RandomGraph with max alpha and subsample
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
    
    # Add small noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y_full) * math.sqrt(NOISE_VAR)
    Y_full = Y_full + noise
    
    # Initialize messages
    torch.manual_seed(seed + 2000)
    
    if direction == "ascending":
        # Start from teacher (warm start)
        m_W = W_teacher.clone()
        m_X = X_teacher.clone()
    else:
        # Start with random initialization (cold start) for descending
        m_W = torch.randn(N1, M, device=device) * 0.1
        m_X = torch.randn(M, N2, device=device) * 0.1
    
    # Initialize v properly - always start fresh
    v_W = torch.ones(N1, M, device=device)
    v_X = torch.ones(M, N2, device=device)
    
    results = {}
    
    for alpha in alphas:
        # Compute number of observations for this alpha
        # C ∝ alpha, so C = (alpha / ALPHA_MAX) * C_max
        C = max(1, int((alpha / ALPHA_MAX) * C_max))
        
        # Use subset of observations
        i_idx = i_idx_full[:C]
        j_idx = j_idx_full[:C]
        F = F_full[:C]
        F_sq = F ** 2
        Y = Y_full[:C]
        g_prev = torch.zeros(C, device=device)
        
        # Reset v for new alpha (keep m as continuation)
        v_W = torch.ones(N1, M, device=device)
        v_X = torch.ones(M, N2, device=device)
        
        # Run G-AMP for this alpha
        for step in range(STEPS_PER_ALPHA):
            m_W, v_W, m_X, v_X, g_prev = gamp_step_with_F(
                m_W, v_W, m_X, v_X,
                Y, F, F_sq, i_idx, j_idx, g_prev,
                NOISE_VAR, DAMPING, N1, N2, M
            )
            
            # Clamp values to prevent NaN propagation
            m_W = torch.clamp(m_W, min=-100, max=100)
            m_X = torch.clamp(m_X, min=-100, max=100)
            v_W = torch.clamp(v_W, min=1e-8, max=100)
            v_X = torch.clamp(v_X, min=1e-8, max=100)
        
        # Handle NaN - reset if needed
        if torch.isnan(m_W).any() or torch.isnan(m_X).any():
            m_W = torch.randn(N1, M, device=device) * 0.1
            m_X = torch.randn(M, N2, device=device) * 0.1
            v_W = torch.ones(N1, M, device=device)
            v_X = torch.ones(M, N2, device=device)
            qy = 0.0
        else:
            # Compute Q_Y at this alpha
            m_W_norm = normalize_to_unit_variance(m_W)
            m_X_norm = normalize_to_unit_variance(m_X)
            qy = compute_qy(m_W_norm, m_X_norm, W_teacher, X_teacher)
        
        results[alpha] = qy
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("G-AMP Alpha Continuation Method")
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
    print(f"Steps per alpha: {STEPS_PER_ALPHA}")
    print(f"Replicas: {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_gamp_continuation"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'gamp_continuation',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_min': ALPHA_MIN,
        'alpha_max': ALPHA_MAX,
        'alpha_steps': ALPHA_STEPS,
        'steps_per_alpha': STEPS_PER_ALPHA,
        'damping': DAMPING,
        'num_replicas': NUM_REPLICAS,
        'device': str(device),
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    start_time = time.time()
    
    # Results storage
    ascending_results = {}   # alpha -> list of Q_Y
    descending_results = {}  # alpha -> list of Q_Y
    
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_STEPS)
    for alpha in alphas:
        ascending_results[alpha] = []
        descending_results[alpha] = []
    
    # Run ascending (teacher → high α)
    print("\n--- Ascending Path (Teacher → High α) ---")
    for rep in range(NUM_REPLICAS):
        seed = SEED + rep * 1000
        results = train_continuation("ascending", device, seed)
        for alpha, qy in results.items():
            ascending_results[alpha].append(qy)
        print(f"Replica {rep+1}/{NUM_REPLICAS} done")
    
    # Run descending (random → low α)
    print("\n--- Descending Path (Random → Low α) ---")
    for rep in range(NUM_REPLICAS):
        seed = SEED + rep * 1000
        results = train_continuation("descending", device, seed)
        for alpha, qy in results.items():
            descending_results[alpha].append(qy)
        print(f"Replica {rep+1}/{NUM_REPLICAS} done")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Compute statistics
    alphas_list = sorted(ascending_results.keys())
    
    asc_means = [np.mean(ascending_results[a]) for a in alphas_list]
    asc_stds = [np.std(ascending_results[a]) for a in alphas_list]
    asc_sems = [s / np.sqrt(NUM_REPLICAS) for s in asc_stds]
    
    desc_means = [np.mean(descending_results[a]) for a in alphas_list]
    desc_stds = [np.std(descending_results[a]) for a in alphas_list]
    desc_sems = [s / np.sqrt(NUM_REPLICAS) for s in desc_stds]
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results (mean ± SEM)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'Ascending':^20} | {'Descending':^20}")
    print("-" * 60)
    for i, alpha in enumerate(alphas_list):
        print(f"{alpha:6.2f} | {asc_means[i]:8.4f} ± {asc_sems[i]:<8.4f} | "
              f"{desc_means[i]:8.4f} ± {desc_sems[i]:<8.4f}")
    print("=" * 60)
    
    # Create plots
    print("\nGenerating plots...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Ascending path (teacher start)
    ax.errorbar(alphas_list, asc_means, yerr=asc_sems,
                fmt='o-', color='#1E88E5', markersize=8, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='Ascending (Teacher→High α)')
    
    # Descending path (random start)
    ax.errorbar(alphas_list, desc_means, yerr=desc_sems,
                fmt='s-', color='#E53935', markersize=8, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='Descending (Random→Low α)')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'G-AMP Continuation Method: Hysteresis Analysis\n'
                 f'({N1}×{N2}, M={M}, {STEPS_PER_ALPHA} steps/α, {NUM_REPLICAS} replicas)', 
                 fontsize=16)
    ax.set_xlim(ALPHA_MIN - 0.1, ALPHA_MAX + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plot_path = plots_dir / "hysteresis_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        f.write("alpha,asc_mean,asc_std,desc_mean,desc_std\n")
        for i, alpha in enumerate(alphas_list):
            f.write(f"{alpha},{asc_means[i]},{asc_stds[i]},{desc_means[i]},{desc_stds[i]}\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")

# %%
