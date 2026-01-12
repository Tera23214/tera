#!/usr/bin/env python
"""
Alternating Gradient Descent (AGD) for Sparse Matrix Factorization.

Extended version with 6 order parameters:
- Magnetization (Student-Teacher overlap): m_W, m_X, m_Y
- Overlap (Student-Student similarity): q_W, q_X, q_Y

This script trains multiple students on the same observation pattern to compute
both student-teacher overlaps and student-student overlaps.
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

# Add parent directory to path (to get smf modules)
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns  
M = 10      # Rank (hidden dimension)

ALPHA_START = 0.1
ALPHA_STOP = 3.5
ALPHA_STEP = 0.5

MAX_STEPS = 3000
LR_BASE = 0.01     # Base learning rate (calibrated for N=1000)
LR = LR_BASE * (1e6 / (N1 * N2))  # Auto-scale
SEED = 42
NUM_REPLICAS = 10    # Number of replicas per alpha
NUM_STUDENTS = 2     # Number of students per replica (for q overlap computation)
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
    """
    Gradient descent step for W (fixing X).
    
    Gradient: dL/dW[i,mu] = 2 * sum_{c: i_c=i} (Y_pred[c] - Y[c]) * X[mu, j_c]
    """
    N1, M = W.shape
    
    # Compute predictions and residuals
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y  # (C,)
    
    # Compute gradient contributions: 2 * residual * X[mu, j_c]
    X_sel = X[:, j_idx.long()].T     # (C, M)
    grad_contrib = 2.0 * residual.unsqueeze(1) * X_sel  # (C, M)
    
    # Scatter-add gradients to W
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), grad_contrib)
    
    # Update W
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
    """
    Gradient descent step for X (fixing W).
    
    Gradient: dL/dX[mu,j] = 2 * sum_{c: j_c=j} (Y_pred[c] - Y[c]) * W[i_c, mu]
    """
    M, N2 = X.shape
    
    # Compute predictions and residuals
    Y_pred = compute_predictions(W, X, i_idx, j_idx)
    residual = Y_pred - Y  # (C,)
    
    # Compute gradient contributions: 2 * residual * W[i_c, mu]
    W_sel = W[i_idx.long(), :]       # (C, M)
    grad_contrib = 2.0 * residual.unsqueeze(1) * W_sel  # (C, M)
    
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


# ============================================================================
# Order Parameter Functions
# ============================================================================

def compute_m_W(W_student: torch.Tensor, W_teacher: torch.Tensor) -> float:
    """
    Compute magnetization m_W (student-teacher overlap for W).
    
    m_W = <W_student, W_teacher> / (N1 * M)
    """
    N1, M = W_teacher.shape
    inner_product = (W_student * W_teacher).sum()
    return (inner_product / (N1 * M)).item()


def compute_m_X(X_student: torch.Tensor, X_teacher: torch.Tensor) -> float:
    """
    Compute magnetization m_X (student-teacher overlap for X).
    
    m_X = <X_student, X_teacher> / (M * N2)
    """
    M, N2 = X_teacher.shape
    inner_product = (X_student * X_teacher).sum()
    return (inner_product / (M * N2)).item()


def compute_m_Y(W_student: torch.Tensor, X_student: torch.Tensor, 
                W_teacher: torch.Tensor, X_teacher: torch.Tensor) -> float:
    """
    Compute magnetization m_Y (student-teacher overlap for Y).
    
    m_Y = <Y_student, Y_teacher> / (N1 * N2 * M)
    """
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    Y_teacher = W_teacher @ X_teacher  # (N1, N2)
    Y_student = W_student @ X_student  # (N1, N2)
    
    inner_product = (Y_teacher * Y_student).sum()
    return (inner_product / (N1 * N2 * M)).item()


def compute_q_W(W_a: torch.Tensor, W_b: torch.Tensor) -> float:
    """
    Compute overlap q_W (student-student overlap for W).
    
    q_W = <W_a, W_b> / (N1 * M)
    """
    N1, M = W_a.shape
    inner_product = (W_a * W_b).sum()
    return (inner_product / (N1 * M)).item()


def compute_q_X(X_a: torch.Tensor, X_b: torch.Tensor) -> float:
    """
    Compute overlap q_X (student-student overlap for X).
    
    q_X = <X_a, X_b> / (M * N2)
    """
    M, N2 = X_a.shape
    inner_product = (X_a * X_b).sum()
    return (inner_product / (M * N2)).item()


def compute_q_Y(W_a: torch.Tensor, X_a: torch.Tensor, 
                W_b: torch.Tensor, X_b: torch.Tensor) -> float:
    """
    Compute overlap q_Y (student-student overlap for Y).
    
    q_Y = <Y_a, Y_b> / (N1 * N2 * M)
    """
    N1, M = W_a.shape
    N2 = X_a.shape[1]
    
    Y_a = W_a @ X_a  # (N1, N2)
    Y_b = W_b @ X_b  # (N1, N2)
    
    inner_product = (Y_a * Y_b).sum()
    return (inner_product / (N1 * N2 * M)).item()


# ============================================================================
# Training Functions
# ============================================================================

def train_single_student(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple:
    """
    Train a single student on given observation pattern.
    
    Returns:
        tuple: (W_hat, X_hat, final_loss, steps_taken)
    """
    # Initialize student randomly
    torch.manual_seed(seed)
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
            Y_pred = compute_predictions(W_hat, X_hat, i_idx, j_idx)
            loss = compute_loss(Y, Y_pred).item()
            final_loss = loss
            
            if loss < CONVERGENCE_THRESHOLD:
                steps_taken = step + 1
                break
    
    return W_hat, X_hat, final_loss, steps_taken


def train_multi_students(
    alpha: float,
    device: torch.device,
    seed: int,
    num_students: int = 2,
) -> dict:
    """
    Train multiple students on the same observation pattern.
    
    Returns:
        dict containing all 6 order parameters
    """
    # Generate teacher for this replica
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph (observed entries) - same for all students
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return {
            'm_W': 0.0, 'm_X': 0.0, 'm_Y': 0.0,
            'q_W': 0.0, 'q_X': 0.0, 'q_Y': 0.0,
            'final_loss': 0.0, 'steps': 0
        }
    
    # Generate Y (observations)
    Y = compute_predictions(W_teacher, X_teacher, i_idx, j_idx)
    
    # Train multiple students
    students = []
    losses = []
    steps_list = []
    
    for s in range(num_students):
        student_seed = seed + 1000 * (s + 1)  # Different seed for each student
        W_hat, X_hat, loss, steps = train_single_student(
            W_teacher, X_teacher, i_idx, j_idx, Y, device, student_seed
        )
        students.append((W_hat, X_hat))
        losses.append(loss)
        steps_list.append(steps)
    
    # Compute magnetization (using first student)
    W_0, X_0 = students[0]
    m_W = compute_m_W(W_0, W_teacher)
    m_X = compute_m_X(X_0, X_teacher)
    m_Y = compute_m_Y(W_0, X_0, W_teacher, X_teacher)
    
    # Compute overlaps between students (average over all pairs)
    q_W_values = []
    q_X_values = []
    q_Y_values = []
    
    for i in range(num_students):
        for j in range(i + 1, num_students):
            W_a, X_a = students[i]
            W_b, X_b = students[j]
            q_W_values.append(compute_q_W(W_a, W_b))
            q_X_values.append(compute_q_X(X_a, X_b))
            q_Y_values.append(compute_q_Y(W_a, X_a, W_b, X_b))
    
    q_W = np.mean(q_W_values) if q_W_values else 0.0
    q_X = np.mean(q_X_values) if q_X_values else 0.0
    q_Y = np.mean(q_Y_values) if q_Y_values else 0.0
    
    return {
        'm_W': m_W, 'm_X': m_X, 'm_Y': m_Y,
        'q_W': q_W, 'q_X': q_X, 'q_Y': q_Y,
        'final_loss': np.mean(losses),
        'steps': np.mean(steps_list)
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alternating Gradient Descent (AGD) - 6 Order Parameters")
    print("GPU Accelerated with Multiple Students per Replica")
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
    print(f"Students per replica: {NUM_STUDENTS}")
    print()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_name = f"{timestamp}_agd_order_params_{N1}x{M}_alpha{ALPHA_START}-{ALPHA_STOP}"
    results_dir = Path(__file__).parent / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'agd_order_params',
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
        'num_students': NUM_STUDENTS,
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
    
    param_names = ['m_W', 'm_X', 'm_Y', 'q_W', 'q_X', 'q_Y']
    
    start_time = time.time()
    total_tasks = len(alphas) * NUM_REPLICAS
    completed = 0
    
    for alpha in alphas:
        # Initialize storage for this alpha
        params_storage = {name: [] for name in param_names}
        loss_values = []
        steps_values = []
        
        for replica_id in range(NUM_REPLICAS):
            seed = SEED + replica_id * 10000
            t0 = time.time()
            
            result = train_multi_students(alpha, device, seed, NUM_STUDENTS)
            
            dt = time.time() - t0
            
            # Store results
            for name in param_names:
                params_storage[name].append(result[name])
            loss_values.append(result['final_loss'])
            steps_values.append(result['steps'])
            
            completed += 1
            print(f"α={alpha:.2f}, replica {replica_id+1}/{NUM_REPLICAS}: "
                  f"m_Y={result['m_Y']:.4f}, q_Y={result['q_Y']:.4f}, "
                  f"Loss={result['final_loss']:.2e} ({dt:.1f}s) [{completed}/{total_tasks}]")
        
        # Compute statistics for this alpha
        results[alpha] = {
            'loss_mean': np.mean(loss_values),
            'loss_std': np.std(loss_values),
            'steps_mean': np.mean(steps_values),
        }
        for name in param_names:
            results[alpha][f'{name}_mean'] = np.mean(params_storage[name])
            results[alpha][f'{name}_std'] = np.std(params_storage[name])
            results[alpha][f'{name}_values'] = params_storage[name]
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("Results (mean ± std)")
    print("=" * 80)
    print(f"{'Alpha':>6} | {'m_W':^15} | {'m_X':^15} | {'m_Y':^15} | {'q_W':^15} | {'q_X':^15} | {'q_Y':^15}")
    print("-" * 80)
    for alpha in sorted(results.keys()):
        r = results[alpha]
        print(f"{alpha:6.2f} | "
              f"{r['m_W_mean']:6.4f}±{r['m_W_std']:.4f} | "
              f"{r['m_X_mean']:6.4f}±{r['m_X_std']:.4f} | "
              f"{r['m_Y_mean']:6.4f}±{r['m_Y_std']:.4f} | "
              f"{r['q_W_mean']:6.4f}±{r['q_W_std']:.4f} | "
              f"{r['q_X_mean']:6.4f}±{r['q_X_std']:.4f} | "
              f"{r['q_Y_mean']:6.4f}±{r['q_Y_std']:.4f}")
    
    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 80)
    
    # Plot all 6 order parameters as separate files
    print("\nGenerating plots...")
    
    alphas_list = sorted(results.keys())
    
    # Create plots subdirectory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plot_params = [
        ('m_W', r'$m_W$', 'Student-Teacher W Overlap', '#1E88E5'),
        ('m_X', r'$m_X$', 'Student-Teacher X Overlap', '#43A047'),
        ('m_Y', r'$m_Y$', 'Student-Teacher Y Overlap', '#E53935'),
        ('q_W', r'$q_W$', 'Student-Student W Overlap', '#7B1FA2'),
        ('q_X', r'$q_X$', 'Student-Student X Overlap', '#FB8C00'),
        ('q_Y', r'$q_Y$', 'Student-Student Y Overlap', '#00ACC1'),
    ]
    
    sample_size = len(alphas_list)
    base_name = f"N1{N1}_N2{N2}_M{M}_samples{sample_size}_replicas{NUM_REPLICAS}"
    
    # Generate 6 separate plots
    for param_name, ylabel, title, color in plot_params:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        means = [results[a][f'{param_name}_mean'] for a in alphas_list]
        stds = [results[a][f'{param_name}_std'] for a in alphas_list]
        sems = [std / math.sqrt(NUM_REPLICAS) for std in stds]
        
        ax.errorbar(alphas_list, means, yerr=sems, 
                    fmt='o-', color=color, markersize=6, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5)
        ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'{title} (AGD)\n({N1}×{N2}, M={M}, {NUM_REPLICAS} replicas)', fontsize=14)
        ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = plots_dir / f"{param_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
        plt.close(fig)
    
    # Also save a combined plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (param_name, ylabel, title, color) in enumerate(plot_params):
        ax = axes[idx // 3, idx % 3]
        
        means = [results[a][f'{param_name}_mean'] for a in alphas_list]
        stds = [results[a][f'{param_name}_std'] for a in alphas_list]
        sems = [std / math.sqrt(NUM_REPLICAS) for std in stds]
        
        ax.errorbar(alphas_list, means, yerr=sems, 
                    fmt='o-', color=color, markersize=5, linewidth=2,
                    capsize=3, capthick=1.5, elinewidth=1.5)
        ax.set_xlabel(r'$\alpha$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'6 Order Parameters (AGD)\n({N1}×{N2}, M={M}, {NUM_REPLICAS} replicas, {NUM_STUDENTS} students)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    combined_path = plots_dir / "all_params.png"
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Combined plot saved: {combined_path}")
    plt.close(fig)
    
    # Save results to CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        # Header
        header = "alpha"
        for name in param_names:
            header += f",{name}_mean,{name}_std"
        header += ",Loss_mean,Loss_std,Steps_mean"
        f.write(header + "\n")
        
        # Data
        for alpha in alphas_list:
            r = results[alpha]
            line = f"{alpha}"
            for name in param_names:
                line += f",{r[f'{name}_mean']},{r[f'{name}_std']}"
            line += f",{r['loss_mean']},{r['loss_std']},{r['steps_mean']}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"\nResults saved to: {results_dir}")
    
    print("Done!")


# %%
