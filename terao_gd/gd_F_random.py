#!/usr/bin/env python
"""
Alternating Gradient Descent (AGD) with Random F Matrix.

This script implements AGD with the same observation model as G-AMP F_random:
    Y_c = (1/√M) Σ_μ F[c,μ] W[i_c,μ] X[μ,j_c]

where F[c,μ] ~ N(0,1) i.i.d.

This allows direct comparison between AGD and G-AMP under identical conditions.
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

ALPHA_START = 0.5
ALPHA_STOP = 5.0
ALPHA_STEP = 0.5

MAX_STEPS = 3000
LR = 0.001  # Fixed LR works better with F_random model
SEED = 42
NUM_REPLICAS = 10
CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# AGD Helper Functions with F
# ============================================================================

def compute_predictions_with_F(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    F: torch.Tensor,       # (C, M) - Random F factors
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
    M: int,
) -> torch.Tensor:
    """
    Compute predictions with F factors:
    Y_pred[c] = (1/√M) * Σ_μ F[c,μ] W[i_c,μ] X[μ,j_c]
    """
    W_sel = W[i_idx.long(), :]       # (C, M)
    X_sel = X[:, j_idx.long()].T     # (C, M)
    
    Y_pred = (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)  # (C,)
    return Y_pred


def compute_loss(Y: torch.Tensor, Y_pred: torch.Tensor, M: int) -> torch.Tensor:
    """Compute MSE loss: L = M * sum((Y - Y_pred)^2)"""
    return M * ((Y - Y_pred) ** 2).sum()


@torch.compile(mode="reduce-overhead")
def agd_step_W_with_F(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    Y: torch.Tensor,       # (C,)
    F: torch.Tensor,       # (C, M)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Gradient descent step for W (fixing X) with F.
    
    Gradient: dL/dW[i,μ] = 2 * Σ_{c:i_c=i} (Y_pred - Y) × (1/√M) F[c,μ] X[μ,j_c]
    """
    N1, M = W.shape
    
    # Compute predictions and residuals
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    Y_pred = (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)
    residual = Y_pred - Y  # (C,)
    
    # Gradient includes F factor
    # Scale: M from loss, 1/√M from Y_pred → net √M
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * F * X_sel  # (C, M)
    
    # Scatter-add gradients to W
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    
    # Update W
    return W - lr * grad_W


@torch.compile(mode="reduce-overhead")
def agd_step_X_with_F(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    Y: torch.Tensor,       # (C,)
    F: torch.Tensor,       # (C, M)
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Gradient descent step for X (fixing W) with F.
    
    Gradient: dL/dX[μ,j] = 2 * Σ_{c:j_c=j} (Y_pred - Y) × (1/√M) F[c,μ] W[i_c,μ]
    """
    M, N2 = X.shape
    
    # Compute predictions and residuals
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    Y_pred = (F * W_sel * X_sel).sum(dim=1) / math.sqrt(M)
    residual = Y_pred - Y  # (C,)
    
    # Gradient includes F factor
    grad_contrib = 2.0 * math.sqrt(M) * residual.unsqueeze(1) * F * W_sel  # (C, M)
    
    # Scatter-add gradients to X
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), grad_contrib.T)
    
    # Update X
    return X - lr * grad_X


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor so that mean square equals 1."""
    mean_sq = (tensor ** 2).mean()
    return tensor / torch.sqrt(mean_sq)


def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """Compute Q_Y overlap using theoretical normalization."""
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    Y_teacher = W_teacher @ X_teacher
    Y_student = W_student @ X_student
    
    inner_product = (Y_teacher * Y_student).sum()
    return (inner_product / (N1 * N2 * M)).item()


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
):
    """
    Train a single replica using AGD with random F.
    
    Returns:
        tuple: (qy, final_loss, steps_taken)
    """
    # Generate teacher
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph (observed entries)
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0, 0.0, 0
    
    # Generate random F matrix: (C, M) with F[c,μ] ~ N(0,1)
    torch.manual_seed(seed + 500)
    F = torch.randn(C, M, device=device, dtype=torch.float32)
    
    # Generate Y with F: Y = (1/√M) Σ_μ F[c,μ] W X
    Y = compute_predictions_with_F(W_teacher, X_teacher, F, i_idx, j_idx, M)
    
    # Initialize student randomly
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.01
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.01
    
    # AGD loop
    final_loss = 0.0
    steps_taken = MAX_STEPS
    
    for step in range(MAX_STEPS):
        # Update W (fix X)
        W_hat = agd_step_W_with_F(W_hat, X_hat, Y, F, i_idx, j_idx, LR)
        
        # Update X (fix W)
        X_hat = agd_step_X_with_F(W_hat, X_hat, Y, F, i_idx, j_idx, LR)
        
        # Normalize
        W_hat = normalize_to_unit_variance(W_hat)
        X_hat = normalize_to_unit_variance(X_hat)
        
        # Check for convergence every 100 steps
        if step % 100 == 0 or step == MAX_STEPS - 1:
            Y_pred = compute_predictions_with_F(W_hat, X_hat, F, i_idx, j_idx, M)
            loss = compute_loss(Y, Y_pred, M).item()
            final_loss = loss
            
            if loss < CONVERGENCE_THRESHOLD:
                steps_taken = step + 1
                break
    
    # Compute Q_Y
    qy = compute_qy(W_hat, X_hat, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alternating Gradient Descent (AGD) with Random F")
    print("Observation model: Y = (1/√M) Σ_μ F_cμ W_iμ X_μj")
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
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = f"{timestamp}_agd_F_random_{N1}x{M}_alpha{ALPHA_START}-{ALPHA_STOP}"
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'agd_F_random',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'max_steps': MAX_STEPS,
        'lr': LR,
        'seed': SEED,
        'num_replicas': NUM_REPLICAS,
        'convergence_threshold': CONVERGENCE_THRESHOLD,
        'device': str(device),
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
        qy_values = []
        loss_values = []
        steps_values = []
        
        for replica_id in range(NUM_REPLICAS):
            seed = SEED + replica_id * 1000
            t0 = time.time()
            qy, final_loss, steps_taken = train_single_replica(alpha, device, seed)
            dt = time.time() - t0
            qy_values.append(qy)
            loss_values.append(final_loss)
            steps_values.append(steps_taken)
            completed += 1
            print(f"α={alpha:.2f}, replica {replica_id+1}/{NUM_REPLICAS}: "
                  f"Q_Y={qy:.4f}, Loss={final_loss:.2e}, "
                  f"Steps={steps_taken} ({dt:.1f}s) [{completed}/{total_tasks}]")
        
        results[alpha] = {
            'qy_mean': np.mean(qy_values),
            'qy_std': np.std(qy_values),
            'qy_values': qy_values,
            'loss_mean': np.mean(loss_values),
            'loss_std': np.std(loss_values),
            'steps_mean': np.mean(steps_values),
        }
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results (mean ± std)")
    print("=" * 60)
    print(f"{'Alpha':>6} | {'Q_Y':^20} | {'Loss':^20} | {'Steps':>8}")
    print("-" * 60)
    for alpha in sorted(results.keys()):
        r = results[alpha]
        print(f"{alpha:6.2f} | {r['qy_mean']:8.4f} ± {r['qy_std']:<8.4f} | "
              f"{r['loss_mean']:8.2e} ± {r['loss_std']:<8.2e} | {r['steps_mean']:8.0f}")
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 60)
    
    # Create plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    alphas_list = sorted(results.keys())
    qy_means = [results[a]['qy_mean'] for a in alphas_list]
    qy_stds = [results[a]['qy_std'] for a in alphas_list]
    qy_sems = [std / math.sqrt(NUM_REPLICAS) for std in qy_stds]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(alphas_list, qy_means, yerr=qy_sems, 
                fmt='o-', color='#E53935', markersize=6, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='AGD with F~N(0,1)')
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(f'Phase Transition (AGD with Random F)\n({N1}×{N2}, M={M}, {MAX_STEPS} steps)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    plot_path = plots_dir / "qy_vs_alpha.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save results to CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        header = "alpha,Q_Y_mean,Q_Y_std,Loss_mean,Loss_std,Steps_mean"
        for i in range(NUM_REPLICAS):
            header += f",qy_replica_{i}"
        f.write(header + "\n")
        
        for alpha in alphas_list:
            r = results[alpha]
            line = f"{alpha},{r['qy_mean']},{r['qy_std']},{r['loss_mean']},{r['loss_std']},{r['steps_mean']}"
            for qy_v in r['qy_values']:
                line += f",{qy_v}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"\nResults saved to: {results_dir}")
    print("Done!")

# %%
