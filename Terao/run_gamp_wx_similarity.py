#!/usr/bin/env python
"""
W, X Similarity Measurement Between Replicas (Q_ab^W, Q_ab^X)

This script trains multiple student replicas on the SAME problem instance
and measures the pairwise overlap of their W and X factors separately.

Key concept:
- All replicas see the same (Teacher, Graph, F coefficients)
- Only the initial conditions differ
- Q_ab^W measures whether W factors converge to the same solution
- Q_ab^X measures whether X factors converge to the same solution

Note: Gauge freedom is ignored in this version.
      W and X are measured with simple Frobenius inner products.

Expected results:
- Low alpha: Q_ab^W ≈ 0, Q_ab^X ≈ 0 (different local minima)
- High alpha: Q_ab^W ≈ 1, Q_ab^X ≈ 1 (unique solution)

Usage:
    cd /Users/password-is-0000/Projects/Sparse-Matrix-Factorization/Terao
    python run_gamp_wx_similarity.py
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

ALPHA_START = 1.0
ALPHA_STOP = 7.0
ALPHA_STEP = 0.5

MAX_STEPS = 300       # BiG-AMP iterations
DAMPING = 0.5
NOISE_VAR = 1e-10
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
            DAMPING, NOISE_VAR
        )
    
    return W_hat, X_hat


# ============================================================================
# Similarity Metrics
# ============================================================================

def compute_pairwise_overlap_W(W_list):
    """
    Compute average pairwise overlap between all replica W factors.
    
    Q_ab^W = <W_a, W_b> / (||W_a|| ||W_b||)
    Using Frobenius inner product (ignoring gauge freedom).
    """
    R = len(W_list)
    if R < 2:
        return 1.0
    
    overlaps = []
    for a, b in combinations(range(R), 2):
        W_a = W_list[a]
        W_b = W_list[b]
        
        num = (W_a * W_b).sum()
        denom = torch.sqrt((W_a ** 2).sum() * (W_b ** 2).sum())
        q_ab = (num / (denom + 1e-10)).item()
        overlaps.append(q_ab)
    
    return sum(overlaps) / len(overlaps)


def compute_pairwise_overlap_X(X_list):
    """
    Compute average pairwise overlap between all replica X factors.
    
    Q_ab^X = <X_a, X_b> / (||X_a|| ||X_b||)
    Using Frobenius inner product (ignoring gauge freedom).
    """
    R = len(X_list)
    if R < 2:
        return 1.0
    
    overlaps = []
    for a, b in combinations(range(R), 2):
        X_a = X_list[a]
        X_b = X_list[b]
        
        num = (X_a * X_b).sum()
        denom = torch.sqrt((X_a ** 2).sum() * (X_b ** 2).sum())
        q_ab = (num / (denom + 1e-10)).item()
        overlaps.append(q_ab)
    
    return sum(overlaps) / len(overlaps)


def compute_pairwise_overlap_Y(W_list, X_list):
    """
    Compute average pairwise overlap for Y = W @ X.
    (For comparison with W and X overlaps)
    """
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


def train_replicas_single_alpha(
    alpha: float,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    device: torch.device,
    seed: int,
    num_replicas: int,
) -> dict:
    """
    Train multiple replicas SEQUENTIALLY for a single alpha value.
    
    Returns:
        dict with Q_ab^W, Q_ab^X, and Q_ab^Y
    """
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    # Generate graph (SAME for all replicas)
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return {'Q_ab_W': 0.0, 'Q_ab_X': 0.0, 'Q_ab_Y': 0.0}
    
    # Generate F (SAME for all replicas)
    F = generate_F(C, M, seed + 1000, device)
    
    # Compute teacher Y (SAME for all replicas)
    alpha_scale = 1.0 / math.sqrt(M)
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y_teacher_values = alpha_scale * (F * W_sel * X_sel).sum(dim=1)
    
    # Train replicas sequentially
    W_list = []
    X_list = []
    
    for r in range(num_replicas):
        replica_seed = seed + 2000 + r * 1000
        W_hat, X_hat = train_single_student(
            Y_teacher_values, F, i_idx, j_idx,
            N1, M, N2, device, replica_seed
        )
        W_list.append(W_hat)
        X_list.append(X_hat)
    
    # Compute all overlap metrics
    Q_ab_W = compute_pairwise_overlap_W(W_list)
    Q_ab_X = compute_pairwise_overlap_X(X_list)
    Q_ab_Y = compute_pairwise_overlap_Y(W_list, X_list)
    
    return {'Q_ab_W': Q_ab_W, 'Q_ab_X': Q_ab_X, 'Q_ab_Y': Q_ab_Y}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("W, X Similarity Measurement (Q_ab^W, Q_ab^X)")
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
    print(f"Steps: {MAX_STEPS}")
    print(f"Replicas per instance: {NUM_REPLICAS}")
    print()
    
    # Generate teacher (SAME for all alphas)
    torch.manual_seed(SEED)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32) / math.sqrt(M)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32) / math.sqrt(M)
    
    # Run for each alpha
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {}
    
    start_time = time.time()
    
    for alpha in alphas:
        t0 = time.time()
        result = train_replicas_single_alpha(
            alpha, W_teacher, X_teacher, device, SEED, NUM_REPLICAS
        )
        dt = time.time() - t0
        results[alpha] = result
        print(f"α={alpha:.2f}: Q_ab^W={result['Q_ab_W']:.4f}, Q_ab^X={result['Q_ab_X']:.4f}, Q_ab^Y={result['Q_ab_Y']:.4f}  ({dt:.1f}s)")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)
    
    # Plot
    print("\nGenerating plot...")
    
    alphas_list = sorted(results.keys())
    qab_w_values = [results[a]['Q_ab_W'] for a in alphas_list]
    qab_x_values = [results[a]['Q_ab_X'] for a in alphas_list]
    qab_y_values = [results[a]['Q_ab_Y'] for a in alphas_list]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(alphas_list, qab_w_values, 'o-', color='#1E88E5', markersize=8, linewidth=2,
            label=r'$\bar{Q}_{ab}^W$ (W overlap)')
    ax.plot(alphas_list, qab_x_values, 's-', color='#43A047', markersize=8, linewidth=2,
            label=r'$\bar{Q}_{ab}^X$ (X overlap)')
    ax.plot(alphas_list, qab_y_values, '^--', color='#E53935', markersize=6, linewidth=1.5,
            alpha=0.7, label=r'$\bar{Q}_{ab}^Y$ (Y=WX overlap)')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel('Overlap', fontsize=14)
    ax.set_title(f'W, X Similarity vs Observation Density\n({N1}×{N2}, M={M}, {NUM_REPLICAS} replicas, {MAX_STEPS} steps)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / f"qab_wx_vs_alpha({N1}x{N2},M{M}).png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    print("Done!")

# %%
