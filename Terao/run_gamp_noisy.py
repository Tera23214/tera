#!/usr/bin/env python
"""
Noisy Observation Simulation (Fixed α, Varying Noise)

This script adds Gaussian noise to the teacher's observations
and measures how the noise level affects student performance.

Model:
    Y_observed = Y_true + ε,  where ε ~ N(0, σ²)

Key concept:
- α (observation density) is FIXED
- Noise variance σ² is swept
- Measures Q_ab (replica similarity) and Q_Y (student-teacher overlap)

Expected results:
- Low noise: Q_ab ≈ 1, Q_Y ≈ 1 (good recovery)
- High noise: Q_ab and Q_Y degrade

Usage:
    cd /Users/password-is-0000/Projects/Sparse-Matrix-Factorization/Terao
    python run_gamp_noisy.py
"""
#%%

import sys
import math
import time
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from smf.modules.graphs.random import RandomGraph

# ============================================================================
# Configuration (Easily adjustable)
# ============================================================================

N1 = 3000   # Number of rows
N2 = 3000   # Number of columns  
M = 30      # Rank (hidden dimension)

# Fixed alpha
ALPHA = 5.0  # Observation density (fix this, vary noise instead)

# Noise variance sweep
NOISE_VAR_START = 0.0
NOISE_VAR_STOP = 0.0005
NOISE_VAR_STEP = 0.0001

MAX_STEPS = 300       # BiG-AMP iterations
DAMPING = 0.5
BIGAMP_NOISE_VAR = 1e-10  # BiG-AMP internal noise (separate from observation noise)
SEED = 42

NUM_REPLICAS = 5     # Number of student replicas per instance

# ============================================================================
# BiG-AMP Spreading Algorithm (2D Version)
# ============================================================================

def generate_F(C: int, M: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate F ~ N(0, 1) spreading coefficients."""
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(C, M, device=device, dtype=torch.float32, generator=gen)


def bigamp_step(
    W_hat: torch.Tensor,   # (N1, M)
    X_hat: torch.Tensor,   # (M, N2)
    W_var: torch.Tensor,   # (N1, M)
    X_var: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,       # (C,)
    F: torch.Tensor,       # (C, M)
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
    damping: float,
    noise_var: float,
):
    """Single BiG-AMP step with spreading (2D version)."""
    N1, M = W_hat.shape
    N2 = X_hat.shape[1]
    C = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # Forward pass
    W_sel = W_hat[i_idx.long(), :]
    X_sel = X_hat[:, j_idx.long()].T
    Z_hat = alpha_scale * (F * W_sel * X_sel).sum(dim=1)

    # Variance
    W_var_sel = W_var[i_idx.long(), :]
    X_var_sel = X_var[:, j_idx.long()].T
    F_sq = F.pow(2)
    V = alpha_scale_sq * (F_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=1)
    V = V + 1e-10

    # Residuals
    denom = torch.clamp(V + noise_var, min=1e-6)
    s = (Y - Z_hat) / denom
    s = torch.clamp(s, min=-1e6, max=1e6)

    # Update W
    s_exp = s.unsqueeze(1)
    inv_V = (1.0 / denom).unsqueeze(1)

    r_W_contrib = alpha_scale * F * X_sel * s_exp
    r_W = torch.zeros(N1, M, device=W_hat.device, dtype=W_hat.dtype)
    r_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(C, M), r_W_contrib)

    tau_W_contrib = alpha_scale_sq * F_sq * X_sel.pow(2) * inv_V
    tau_W = torch.zeros(N1, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(C, M), tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)

    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_hat + W_var_new * r_W

    # Update X
    r_X_contrib = alpha_scale * F * W_sel * s_exp
    r_X = torch.zeros(M, N2, device=X_hat.device, dtype=X_hat.dtype)
    r_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, C), r_X_contrib.T)

    tau_X_contrib = alpha_scale_sq * F_sq * W_sel.pow(2) * inv_V
    tau_X = torch.zeros(M, N2, device=X_hat.device, dtype=X_hat.dtype)
    tau_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, C), tau_X_contrib.T)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_hat + X_var_new * r_X

    # Damping
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)

    # NaN protection
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out


def train_single_student(
    Y: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    N1: int,
    M: int,
    N2: int,
    device: torch.device,
    seed: int,
):
    """Train a single student with BiG-AMP."""
    torch.manual_seed(seed)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.1
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.1
    W_var = torch.ones(N1, M, device=device, dtype=torch.float32)
    X_var = torch.ones(M, N2, device=device, dtype=torch.float32)
    
    for step in range(MAX_STEPS):
        W_hat, X_hat, W_var, X_var = bigamp_step(
            W_hat, X_hat, W_var, X_var,
            Y, F, i_idx, j_idx,
            DAMPING, BIGAMP_NOISE_VAR
        )
    
    return W_hat, X_hat


# ============================================================================
# Similarity Metrics
# ============================================================================

def compute_pairwise_overlap(W_list, X_list):
    """Compute average pairwise overlap Q_ab for Y = W @ X."""
    R = len(W_list)
    if R < 2:
        return 1.0
    
    overlaps = []
    for a, b in combinations(range(R), 2):
        Y_a = W_list[a] @ X_list[a]
        Y_b = W_list[b] @ X_list[b]
        
        num = (Y_a * Y_b).sum()
        denom = torch.sqrt((Y_a ** 2).sum() * (Y_b ** 2).sum())
        q_ab = (num / (denom + 1e-10)).item()
        overlaps.append(q_ab)
    
    return sum(overlaps) / len(overlaps)


def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """Compute Q_Y overlap using full dense matrices."""
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student
    
    num = (Y_teacher * Y_student).sum()
    denom = torch.sqrt((Y_teacher ** 2).sum() * (Y_student ** 2).sum())
    
    return (num / (denom + 1e-10)).item()


def train_replicas_single_noise(
    noise_var: float,
    alpha: float,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    device: torch.device,
    seed: int,
    num_replicas: int,
) -> dict:
    """Train replicas with noisy observations."""
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    # Generate graph
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return {'Q_ab': 0.0, 'Q_Y': 0.0}
    
    # Generate F
    F = generate_F(C, M, seed + 1000, device)
    
    # Compute TRUE teacher Y (without noise)
    alpha_scale = 1.0 / math.sqrt(M)
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y_true = alpha_scale * (F * W_sel * X_sel).sum(dim=1)
    
    # Add Gaussian noise: Y_noisy = Y_true + ε, ε ~ N(0, noise_var)
    torch.manual_seed(seed + 500)  # Fixed noise seed for reproducibility
    noise = torch.randn_like(Y_true) * math.sqrt(noise_var)
    Y_noisy = Y_true + noise
    
    # Train replicas with NOISY observations
    W_list = []
    X_list = []
    qy_values = []
    
    for r in range(num_replicas):
        replica_seed = seed + 2000 + r * 1000
        W_hat, X_hat = train_single_student(
            Y_noisy, F, i_idx, j_idx,  # Use noisy Y
            N1, M, N2, device, replica_seed
        )
        W_list.append(W_hat)
        X_list.append(X_hat)
        
        qy = compute_qy(W_hat, X_hat, W_teacher, X_teacher)
        qy_values.append(qy)
    
    Q_ab = compute_pairwise_overlap(W_list, X_list)
    Q_Y = sum(qy_values) / len(qy_values)
    
    return {'Q_ab': Q_ab, 'Q_Y': Q_Y}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Noisy Observation Simulation (Fixed α, Varying Noise)")
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
    print(f"Steps: {MAX_STEPS}")
    print(f"Replicas per instance: {NUM_REPLICAS}")
    print()
    
    # Generate teacher
    torch.manual_seed(SEED)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32) / math.sqrt(M)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32) / math.sqrt(M)
    
    # Run for each noise level
    noise_vars = np.arange(NOISE_VAR_START, NOISE_VAR_STOP + NOISE_VAR_STEP/2, NOISE_VAR_STEP)
    results = {}
    
    start_time = time.time()
    
    for nv in noise_vars:
        t0 = time.time()
        result = train_replicas_single_noise(
            nv, ALPHA, W_teacher, X_teacher, device, SEED, NUM_REPLICAS
        )
        dt = time.time() - t0
        results[nv] = result
        print(f"σ²={nv:.2e}: Q_ab={result['Q_ab']:.4f}, Q_Y={result['Q_Y']:.4f}  ({dt:.1f}s)")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)
    
    # Plot
    print("\nGenerating plot...")
    
    noise_list = sorted(results.keys())
    qab_values = [results[n]['Q_ab'] for n in noise_list]
    qy_values = [results[n]['Q_Y'] for n in noise_list]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(noise_list, qab_values, 'o-', color='#1E88E5', markersize=8, linewidth=2,
            label=r'$\bar{Q}_{ab}$ (replica-replica overlap)')
    ax.plot(noise_list, qy_values, 's--', color='#E53935', markersize=6, linewidth=1.5,
            alpha=0.7, label=r'$Q_Y$ (student-teacher overlap)')
    
    ax.set_xlabel(r'Noise variance $\sigma^2$', fontsize=14)
    ax.set_ylabel('Overlap', fontsize=14)
    ax.set_title(f'Effect of Observation Noise\n({N1}×{N2}, M={M}, α={ALPHA}, {NUM_REPLICAS} replicas)', fontsize=16)
    
    # Dynamic x-axis limits based on noise range
    x_margin = max(NOISE_VAR_STEP * 0.5, (NOISE_VAR_STOP - NOISE_VAR_START) * 0.05)
    ax.set_xlim(NOISE_VAR_START - x_margin, NOISE_VAR_STOP + x_margin)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=12)
    
    # Use scientific notation for small values
    if NOISE_VAR_STOP < 0.1:
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f"qab_vs_noise({N1}x{N2},M{M},alpha{ALPHA}).png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    print("Done!")

# %%
