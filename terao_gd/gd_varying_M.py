#!/usr/bin/env python
"""
AGD with Varying Rank M: Alternating Gradient Descent with Fixed Alpha.

This script keeps alpha (observation density) fixed and varies the rank M.
Since C = alpha * M * N, changing M also changes the number of observations.

Key concept:
- α (observation density) is FIXED
- M (rank / hidden dimension) is SWEPT
- Measures Q_Y (student-teacher overlap) as M changes

Expected results:
- Small M: Easier to recover (fewer parameters)
- Large M: Harder to recover (more parameters, but also more observations)
- There may be a phase transition or optimal M region

Loss function: L = sum((Y - Y_pred)^2)
Optimization: Alternating updates of W and X using gradient descent.
"""

#%%

import sys
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path (to get smf modules)
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from smf.modules.graphs.random import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns  

# Fixed alpha
ALPHA = 3.0  # Observation density (fixed)

# Rank M sweep
M_START = 5
M_STOP = 50
M_STEP = 5

MAX_STEPS = 3000
LR_BASE = 0.01     # Base learning rate (calibrated for N=1000)
SEED = 42
NUM_REPLICAS = 10  # Number of replicas per M value
CONVERGENCE_THRESHOLD = 1e-6  # Early stopping threshold for loss

# ============================================================================
# AGD Helper Functions
# ============================================================================

def compute_predictions(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
) -> torch.Tensor:
    """
    Compute predictions Y_pred for observed entries.
    
    Y_pred[c] = W[i_c,:] @ X[:,j_c] = sum_mu W[i_c, mu] * X[mu, j_c]
    """
    W_sel = W[i_idx.long(), :]       # (C, M)
    X_sel = X[:, j_idx.long()].T     # (C, M)
    
    Y_pred = (W_sel * X_sel).sum(dim=1)  # (C,)
    return Y_pred


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss: L = sum((Y - Y_pred)^2)"""
    return ((Y - Y_pred) ** 2).sum()


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
    
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y
    
    X_sel = X[:, j_idx.long()].T
    grad_contrib = 2.0 * residual.unsqueeze(1) * X_sel
    
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    
    W_new = W - lr * grad_W
    return W_new


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
    
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y
    
    W_sel = W[i_idx.long(), :]
    grad_contrib = 2.0 * residual.unsqueeze(1) * W_sel
    
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


def train_single_replica_varying_M(
    M: int,
    alpha: float,
    device: torch.device,
    seed: int,
):
    """
    Train a single replica for a given M (rank) with fixed alpha.
    
    Returns:
        tuple: (qy, final_loss, steps_taken, C)
    """
    # Compute learning rate scaled for matrix size
    lr = LR_BASE * (1e6 / (N1 * N2))
    
    # Generate teacher for this replica
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph (observed entries)
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0, 0.0, 0, 0
    
    # Generate Y (observations)
    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx)
    
    # Initialize student randomly
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    
    # Alternating Gradient Descent loop with early stopping
    final_loss = 0.0
    steps_taken = MAX_STEPS
    
    for step in range(MAX_STEPS):
        W_hat = agd_step_W(W_hat, X_hat, Y, i_idx, j_idx, lr)
        X_hat = agd_step_X(W_hat, X_hat, Y, i_idx, j_idx, lr)
        
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
        
        # Check for convergence every 100 steps
        if step % 100 == 0 or step == MAX_STEPS - 1:
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx)
            loss = compute_loss(Y, Y_pred).item()
            final_loss = loss
            
            if loss < CONVERGENCE_THRESHOLD:
                steps_taken = step + 1
                break
    
    # Compute Q_Y
    qy = compute_qy(W_hat, X_hat, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken, C


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGD with Varying Rank M (Fixed Alpha)")
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
    print(f"Alpha (fixed): {ALPHA}")
    print(f"M: {M_START} ~ {M_STOP} (step {M_STEP})")
    print(f"Steps: {MAX_STEPS}, LR_BASE={LR_BASE}")
    print(f"Replicas per M: {NUM_REPLICAS}")
    print()
    
    # Run simulations
    M_values = list(range(M_START, M_STOP + 1, M_STEP))
    results = {}
    
    start_time = time.time()
    total_tasks = len(M_values) * NUM_REPLICAS
    completed = 0
    
    for M in M_values:
        qy_values = []
        loss_values = []
        steps_values = []
        C_values = []
        
        for replica_id in range(NUM_REPLICAS):
            seed = SEED + replica_id * 1000
            t0 = time.time()
            qy, final_loss, steps_taken, C = train_single_replica_varying_M(M, ALPHA, device, seed)
            dt = time.time() - t0
            qy_values.append(qy)
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            C_values.append(C)
            completed += 1
            print(f"M={M:3d}, replica {replica_id+1}/{NUM_REPLICAS}: Q_Y={qy:.4f}, C={C:6d}, Loss={final_loss:.2e}, Steps={steps_taken} ({dt:.1f}s) [{completed}/{total_tasks}]")
        
        results[M] = {
            'qy_mean': np.mean(qy_values),
            'qy_std': np.std(qy_values),
            'qy_values': qy_values,
            'loss_mean': np.mean(loss_values),
            'loss_std': np.std(loss_values),
            'loss_values': loss_values,
            'steps_mean': np.mean(steps_values),
            'C_mean': np.mean(C_values),
            'C_values': C_values,
        }
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("Results (mean ± std)")
    print("=" * 70)
    print(f"{'M':>5} | {'C':>10} | {'Q_Y':^20} | {'Loss':^20} | {'Steps':>8}")
    print("-" * 70)
    for M in sorted(results.keys()):
        r = results[M]
        print(f"{M:5d} | {r['C_mean']:10.0f} | {r['qy_mean']:8.4f} ± {r['qy_std']:<8.4f} | {r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | {r['steps_mean']:8.0f}")
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 70)
    
    # Plot Q_Y vs M with error bars
    print("\nGenerating plots...")
    
    M_list = sorted(results.keys())
    qy_means = [results[m]['qy_mean'] for m in M_list]
    qy_stds = [results[m]['qy_std'] for m in M_list]
    qy_sems = [std / math.sqrt(NUM_REPLICAS) for std in qy_stds]
    
    C_means = [results[m]['C_mean'] for m in M_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Q_Y vs M
    ax1 = axes[0]
    ax1.errorbar(M_list, qy_means, yerr=qy_sems, 
                fmt='o-', color='#1E88E5', markersize=8, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5)
    ax1.set_xlabel(r'Rank $M$', fontsize=14)
    ax1.set_ylabel(r'$Q_Y$ (student-teacher overlap)', fontsize=14)
    ax1.set_title(f'Effect of Rank M on Recovery (α={ALPHA})\n({N1}×{N2}, {NUM_REPLICAS} replicas)', fontsize=14)
    ax1.set_xlim(M_START - M_STEP * 0.5, M_STOP + M_STEP * 0.5)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: C vs M (to show the relationship)
    ax2 = axes[1]
    ax2.plot(M_list, C_means, 'o-', color='#43A047', markersize=8, linewidth=2)
    ax2.set_xlabel(r'Rank $M$', fontsize=14)
    ax2.set_ylabel(r'Number of observations $C$', fontsize=14)
    ax2.set_title(f'Observations C = α·M·N (α={ALPHA})', fontsize=14)
    ax2.set_xlim(M_START - M_STEP * 0.5, M_STOP + M_STEP * 0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add theoretical line for C = alpha * M * sqrt(N1*N2)
    theoretical_C = [ALPHA * m * math.sqrt(N1 * N2) for m in M_list]
    ax2.plot(M_list, theoretical_C, '--', color='#FFA726', linewidth=2, label=r'$C = \alpha \cdot M \cdot \sqrt{N_1 N_2}$')
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save with parameters in filename
    sample_size = len(M_list)
    base_name = f"qy_vs_M_agd_N1{N1}_N2{N2}_alpha{ALPHA}_samples{sample_size}_replicas{NUM_REPLICAS}"
    
    output_path = Path(__file__).parent / f"{base_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    # Save results to CSV
    csv_path = Path(__file__).parent / f"{base_name}.csv"
    with open(csv_path, 'w') as f:
        header = "M,C_mean,Q_Y_mean,Q_Y_std,Loss_mean,Loss_std,Steps_mean"
        for i in range(NUM_REPLICAS):
            header += f",qy_replica_{i},loss_replica_{i},C_replica_{i}"
        f.write(header + "\n")
        
        for M in M_list:
            r = results[M]
            line = f"{M},{r['C_mean']},{r['qy_mean']},{r['qy_std']},{r['loss_mean']},{r['loss_std']},{r['steps_mean']}"
            for qy_v, loss_v, c_v in zip(r['qy_values'], r['loss_values'], r['C_values']):
                line += f",{qy_v},{loss_v},{c_v}"
            f.write(line + "\n")
    
    print(f"CSV saved to: {csv_path}")
    
    print("Done!")

# %%
