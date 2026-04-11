#!/usr/bin/env python
"""
Alternating Gradient Descent (AGD) for sparse matrix factorization with
cosine-similarity evaluation.

This script implements alternating gradient descent for W, X optimization.
Student estimates teacher's matrices through partially observed entries.

Loss function: L = sum((Y - Y_pred)^2)
where Y[i,j] = W_teacher[i,:] @ X_teacher[:,j] (observed entries only)
      Y_pred[i,j] = W_hat[i,:] @ X_hat[:,j]

Optimization: Alternating updates of W and X using gradient descent.
Supports multiple replicas for statistical averaging (GPU accelerated).
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

# Add project root to path(to get smf modules)
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 100   # Number of rows
N2 = 100   # Number of columns  
M = 10     # Rank (hidden dimension)

ALPHA_START = 0
ALPHA_STOP = 5
ALPHA_STEP = 0.2

MAX_STEPS = 3000
LR_BASE = 0.3   # Base learning rate (calibrated for N=1000)
LR = LR_BASE / math.sqrt(N1 * N2 * M)  # Auto-scale: 0.01 for N=1000, ~0.001 for N=3000
SEED = 42
NUM_REPLICAS = 10   # Number of replicas per alpha
CONVERGENCE_THRESHOLD = 1e-6  # Early stopping threshold for loss

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
    
    The 1/√M scaling ensures proper normalization: E[Y²] ~ O(1).
    """
    W_sel = W[i_idx.long(), :]       # (C, M)観測された行列の抽出
    X_sel = X[:, j_idx.long()].T     # (C, M)観測された行列のを抽出してから転置
    
    Y_pred = (W_sel * X_sel).sum(dim=1) / math.sqrt(M)  # (C,)
    return Y_pred


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """
    Compute the optimization loss: L = M * sum((Y - Y_pred)^2)
    
    The M factor compensates for 1/√M scaling in Y, keeping gradient scale unchanged.
    """
    return M * ((Y - Y_pred) ** 2).sum()


def compute_loss_per_edge(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """
    Compute reported loss normalized by the number of observed edges.

    This keeps the optimization loss unchanged while reporting a per-edge value:
    L_report = (M * sum((Y - Y_pred)^2)) / C
    """
    num_edges = max(Y.numel(), 1)
    return compute_loss(Y, Y_pred, M) / num_edges


@torch.compile(mode="reduce-overhead")
def agd_step_W(
    W: torch.Tensor,   # (N1, M)
    X: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,   # (C,)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Gradient descent step for W (fixing X).
    
    Gradient: dL/dW[i,mu] = 2 * sum_{c: i_c=i} (Y_pred[c] - Y[c]) * X[mu, j_c]
    """
    N1, M = W.shape
    
    # Compute predictions and residuals
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y  # (C,)
    
    # Compute gradient contributions: 2 * residual * X[mu, j_c]
    X_sel = X[:, j_idx.long()].T     # (C, M)
    # Gradient includes M factor from loss and 1/√M from Y, net effect: √M factor
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * X_sel  # (C, M)
    
    # Scatter-add gradients to W
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    
    # Update W
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
    """
    Gradient descent step for X (fixing W).
    
    Gradient: dL/dX[mu,j] = 2 * sum_{c: j_c=j} (Y_pred[c] - Y[c]) * W[i_c, mu]
    """
    M, N2 = X.shape
    
    # Compute predictions and residuals
    N1 = W.shape[0]  # Get N1 for M parameter
    Y_pred = compute_predictions(W, X, i_idx, j_idx, M)
    residual = Y_pred - Y  # (C,)
    
    # Compute gradient contributions: 2 * residual * W[i_c, mu]
    W_sel = W[i_idx.long(), :]       # (C, M)
    # Gradient includes M factor from loss and 1/√M from Y, net effect: √M factor
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * W_sel  # (C, M)
    
    # Scatter-add gradients to X
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    
    # Update X
    X_new = X - lr * grad_X
    return X_new


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor so that mean square equals 1.
    
    E[x^2] = 1  =>  x_new = x / sqrt(mean(x^2))
    """
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq)


def compute_y_cosine_similarity(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> float:
    """
    Compute cosine similarity between Y_teacher = W_teacher X_teacher and
    Y_student = W_student X_student without materializing the dense Y matrices.
    """
    cross_w = W_teacher.T @ W_student
    cross_x = X_student @ X_teacher.T
    inner = torch.trace(cross_w @ cross_x)

    teacher_norm_sq = torch.trace((W_teacher.T @ W_teacher) @ (X_teacher @ X_teacher.T))
    student_norm_sq = torch.trace((W_student.T @ W_student) @ (X_student @ X_student.T))
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))

    return (inner / denom).item()


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
):
    """
    Train a single replica for a given alpha using GPU.
    
    Returns:
        tuple: (cosine_similarity, reported_loss_per_edge, steps_taken)
    """
    # Generate teacher for this replica
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph (observed entries)
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0, 0.0, 0
    
    # Generate Y (observations)
    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx, M)
    
    # Initialize student randomly
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    
    # Alternating Gradient Descent loop with early stopping
    final_loss = 0.0
    steps_taken = MAX_STEPS
    
    for step in range(MAX_STEPS):
        # Update W (fix X)
        W_hat = agd_step_W(W_hat, X_hat, Y, i_idx, j_idx, LR)
        
        # Update X (fix W)
        X_hat = agd_step_X(W_hat, X_hat, Y, i_idx, j_idx, LR)
        
        # Apply constraint: normalize so that mean square = 1
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
        
        # Check for convergence every 100 steps
        if step % 100 == 0 or step == MAX_STEPS - 1:
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx, M)
            raw_loss = compute_loss(Y, Y_pred, M).item()
            final_loss = compute_loss_per_edge(Y, Y_pred, M).item()
            
            if raw_loss < CONVERGENCE_THRESHOLD:
                steps_taken = step + 1
                break
    
    # Compute cosine similarity in Y-space
    cosine_similarity = compute_y_cosine_similarity(
        W_hat, X_hat, W_teacher, X_teacher
    )
    
    return cosine_similarity, final_loss, steps_taken


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alternating Gradient Descent (AGD) - Matrix Factorization")
    print("GPU Accelerated with Multiple Replicas")
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
    print(f"Steps: {MAX_STEPS}, LR={LR}")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = (
        f"{timestamp}_agd_cosine_{N1}x{M}_alpha{ALPHA_START}-{ALPHA_STOP}"
    )
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'agd_cosine',
        'N1': N1,
        'N2': N2,
        'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'max_steps': MAX_STEPS,
        'lr': LR,
        'lr_base': LR_BASE,
        'seed': SEED,
        'num_replicas': NUM_REPLICAS,
        'convergence_threshold': CONVERGENCE_THRESHOLD,
        'device': str(device),
        'evaluation_metric': 'cosine_similarity_in_Y_space',
    }
    config_path = results_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved: {config_path}")
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {}
    
    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0
    
    for alpha in alphas:
        cosine_similarity_values = []
        loss_values = []
        steps_values = []
        for replica_id in range(NUM_REPLICAS):
            seed = SEED + replica_id * 1000
            t0 = time.time()
            cosine_similarity, final_loss, steps_taken = train_single_replica(
                alpha, device, seed
            )
            dt = time.time() - t0
            cosine_similarity_values.append(cosine_similarity)
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            completed += 1
            print(
                f"α={alpha:.2f}, replica {replica_id+1}/{NUM_REPLICAS}: "
                f"CosSim={cosine_similarity:.4f}, Loss/edge={final_loss:.2e}, "
                f"Steps={steps_taken} ({dt:.1f}s) [{completed}/{total_tasks}]"
            )
        
        results[alpha] = {
            'cosine_similarity_mean': np.mean(cosine_similarity_values),
            'cosine_similarity_std': np.std(cosine_similarity_values),
            'cosine_similarity_values': cosine_similarity_values,
            'loss_mean': np.mean(loss_values),
            'loss_std': np.std(loss_values),
            'loss_values': loss_values,
            'steps_mean': np.mean(steps_values),
        }
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results (mean ± std)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'CosSim':^20} | {'Loss/edge':^20} | {'Steps':>8}")
    print("-" * 60)
    for alpha in sorted(results.keys()):
        r = results[alpha]
        print(
            f"{alpha:6.2f} | "
            f"{r['cosine_similarity_mean']:8.4f} ± {r['cosine_similarity_std']:<8.4f} | "
            f"{r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | "
            f"{r['steps_mean']:8.0f}"
        )
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 60)
    
    # Plot cosine similarity vs alpha with error bars
    print("\nGenerating plots...")
    
    alphas_list = sorted(results.keys())
    cosine_similarity_means = [
        results[a]['cosine_similarity_mean'] for a in alphas_list
    ]
    cosine_similarity_stds = [
        results[a]['cosine_similarity_std'] for a in alphas_list
    ]
    # Standard Error of Mean (SEM) = std / sqrt(N)
    cosine_similarity_sems = [
        std / math.sqrt(NUM_REPLICAS) for std in cosine_similarity_stds
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(alphas_list, cosine_similarity_means, yerr=cosine_similarity_sems, 
                fmt='o-', color='#1E88E5', markersize=6, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5)
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel("Cosine Similarity", fontsize=14)
    ax.set_title(f'Phase Transition (AGD)\n({N1}×{N2}, M={M}, {MAX_STEPS} steps, {NUM_REPLICAS} replicas)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create plots subdirectory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Save plot
    plot_path = plots_dir / "cosine_similarity_vs_alpha.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save results to CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        # Header
        header = (
            "alpha,cosine_similarity_mean,cosine_similarity_std,"
            "loss_per_edge_mean,loss_per_edge_std,Steps_mean"
        )
        for i in range(NUM_REPLICAS):
            header += f",cosine_similarity_replica_{i},loss_per_edge_replica_{i}"
        f.write(header + "\n")
        
        # Data
        for alpha in alphas_list:
            r = results[alpha]
            line = (
                f"{alpha},{r['cosine_similarity_mean']},"
                f"{r['cosine_similarity_std']},{r['loss_mean']},"
                f"{r['loss_std']},{r['steps_mean']}"
            )
            for cosine_similarity_value, loss_v in zip(
                r['cosine_similarity_values'], r['loss_values']
            ):
                line += f",{cosine_similarity_value},{loss_v}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"\nResults saved to: {results_dir}")
    
    print("Done!")


# %%
