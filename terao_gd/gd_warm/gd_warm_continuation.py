#!/usr/bin/env python
"""
AGD Alpha Continuation Method for Hysteresis Analysis.
(Standard observation model without random F)

TRUE CONTINUATION with Forward/Backward sweeps:
1. Forward sweep: α increases from α_min to α_max
2. Backward sweep: α decreases from α_max to α_min

Uses consistent graph and observations across all alpha values.
Previous solution is used as initial condition for each step.

Observation model: Y_c = (1/√M) Σ_μ W[i_c,μ] X[μ,j_c]
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

N1 = 1000
N2 = 1000
M = 10

ALPHA_MIN = 0.5
ALPHA_MAX = 4.0
ALPHA_STEP = 0.5

MAX_STEPS_PER_ALPHA = 2000
LR_BASE = 1.0  # Base learning rate (will be scaled by system size)
SEED = 42
NUM_REPLICAS = 5

# Compute effective LR scaled by system size: LR = LR_BASE / sqrt(N1 * N2)
# This ensures updates don't become too large as system size increases
import math as _math
LR = LR_BASE / _math.sqrt(N1 * N2)
print(f"Effective LR: {LR:.6f} (LR_BASE={LR_BASE} / sqrt({N1}*{N2}))")

# Early stopping parameters
CONVERGENCE_THRESHOLD = 1e-8
CONVERGENCE_CHECK_INTERVAL = 20

# Observation noise (standard deviation)
SIGMA_Y = 1  # Nearly noiseless for clean comparison

# Initialization epsilon values at start of forward sweep.
# Finite values use student = epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1).
# epsilon=1 is exact teacher, inf is random initialization.
SIGMA_INIT_VALUES = [0.0, 0.1, 0.5, 1.0, float('inf')]

# ============================================================================
# AGD Helper Functions (Standard model without F)
# ============================================================================

def compute_predictions(W, X, i_idx, j_idx, M):
    """Compute Y_pred = (1/√M) Σ_μ W[i,μ] X[μ,j]"""
    W_sel = W[i_idx.long(), :]
    X_sel = X[:, j_idx.long()].T
    return (W_sel * X_sel).sum(dim=1) / math.sqrt(M)


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


def _agd_step_W_impl(W, X, Y, i_idx_long, j_idx_long, i_idx_exp, lr, M_sqrt):
    """AGD step for W (internal implementation)."""
    W_sel = W[i_idx_long, :]
    X_sel = X[:, j_idx_long].T
    Y_pred = (W_sel * X_sel).sum(dim=1) / M_sqrt
    residual = Y_pred - Y
    grad_contrib = 2.0 * M_sqrt * residual.unsqueeze(1) * X_sel
    grad_W = torch.zeros_like(W)
    grad_W.scatter_add_(0, i_idx_exp, grad_contrib)
    return W - lr * grad_W


def _agd_step_X_impl(W, X, Y, i_idx_long, j_idx_long, j_idx_exp, lr, M_sqrt):
    """AGD step for X (internal implementation)."""
    W_sel = W[i_idx_long, :]
    X_sel = X[:, j_idx_long].T
    Y_pred = (W_sel * X_sel).sum(dim=1) / M_sqrt
    residual = Y_pred - Y
    grad_contrib = 2.0 * M_sqrt * residual.unsqueeze(1) * W_sel
    grad_X = torch.zeros_like(X)
    grad_X.scatter_add_(1, j_idx_exp, grad_contrib.T)
    return X - lr * grad_X


# Try to compile functions for speedup (PyTorch 2.0+)
USE_COMPILE = True
try:
    if USE_COMPILE and hasattr(torch, 'compile'):
        agd_step_W = torch.compile(_agd_step_W_impl)
        agd_step_X = torch.compile(_agd_step_X_impl)
        print("torch.compile enabled for AGD steps")
    else:
        agd_step_W = _agd_step_W_impl
        agd_step_X = _agd_step_X_impl
except Exception as e:
    print(f"torch.compile failed: {e}, using uncompiled functions")
    agd_step_W = _agd_step_W_impl
    agd_step_X = _agd_step_X_impl


# ============================================================================
# Problem Setup (consistent across all alpha)
# ============================================================================

def create_full_problem(device: torch.device, seed: int):
    """
    Create the full problem instance for alpha_max.
    All smaller alpha problems are subsets of this.
    """
    # Generate teacher matrices
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate full graph at maximum alpha
    graph = RandomGraph()
    i_idx_full, j_idx_full, E_max = graph.generate(N1, N2, M, ALPHA_MAX, device, seed)
    
    if E_max == 0:
        return None
    
    # Compute Y for full graph (no F matrix, standard model)
    W_sel_full = W_teacher[i_idx_full.long(), :]
    X_sel_full = X_teacher[:, j_idx_full.long()].T
    Y_clean = (W_sel_full * X_sel_full).sum(dim=1) / math.sqrt(M)
    
    # Add observation noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y_clean) * SIGMA_Y
    Y_full = Y_clean + noise
    
    return {
        'W_teacher': W_teacher,
        'X_teacher': X_teacher,
        'i_idx_full': i_idx_full,
        'j_idx_full': j_idx_full,
        'Y_full': Y_full,
        'E_max': E_max,
    }


def get_problem_at_alpha(full_problem: dict, alpha: float):
    """
    Get problem data for a specific alpha by subsetting the full problem.
    """
    E_max = full_problem['E_max']
    
    # Compute number of edges for this alpha
    E = max(1, int((alpha / ALPHA_MAX) * E_max))
    E = min(E, E_max)
    
    # Use first E edges (consistent ordering)
    i_idx = full_problem['i_idx_full'][:E]
    j_idx = full_problem['j_idx_full'][:E]
    Y = full_problem['Y_full'][:E]
    
    return {
        'i_idx': i_idx,
        'j_idx': j_idx,
        'Y': Y,
        'E': E,
        'W_teacher': full_problem['W_teacher'],
        'X_teacher': full_problem['X_teacher'],
    }


# ============================================================================
# Training Functions
# ============================================================================

def initialize_matrices(
    sigma_init: float,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    device: torch.device,
    seed: int,
):
    """Initialize W, X from epsilon_init."""
    torch.manual_seed(seed + 2000)
    
    if math.isinf(sigma_init):
        # Random initialization (cold start)
        W = torch.randn(N1, M, device=device) * 0.01
        X = torch.randn(M, N2, device=device) * 0.01
    else:
        if not 0.0 <= sigma_init <= 1.0:
            raise ValueError(f"epsilon_init must satisfy 0 <= epsilon <= 1, got {sigma_init}.")
        noise_scale = math.sqrt(sigma_init - sigma_init * sigma_init)
        W = sigma_init * W_teacher + noise_scale * torch.randn(N1, M, device=device)
        X = sigma_init * X_teacher + noise_scale * torch.randn(M, N2, device=device)
    
    return W, X


def train_at_alpha(problem: dict, W: torch.Tensor, X: torch.Tensor):
    """
    Train at a specific alpha using given initial W, X.
    
    Returns:
        tuple: (qy, W, X)
    """
    i_idx = problem['i_idx']
    j_idx = problem['j_idx']
    Y = problem['Y']
    W_teacher = problem['W_teacher']
    X_teacher = problem['X_teacher']
    
    # Pre-compute indices
    i_idx_long = i_idx.long()
    j_idx_long = j_idx.long()
    i_idx_exp = i_idx_long.unsqueeze(1).expand(-1, M).contiguous()
    j_idx_exp = j_idx_long.unsqueeze(0).expand(M, -1).contiguous()
    
    # Pre-compute sqrt(M) for torch.compile compatibility
    M_sqrt = math.sqrt(M)
    
    # Run AGD with early stopping
    prev_loss = float('inf')
    for step in range(MAX_STEPS_PER_ALPHA):
        W = agd_step_W(W, X, Y, i_idx_long, j_idx_long, i_idx_exp, LR, M_sqrt)
        X = agd_step_X(W, X, Y, i_idx_long, j_idx_long, j_idx_exp, LR, M_sqrt)
        W = normalize_to_unit_variance(W)
        X = normalize_to_unit_variance(X)
        
        # Early stopping check
        if step % CONVERGENCE_CHECK_INTERVAL == 0:
            W_sel = W[i_idx_long, :]
            X_sel = X[:, j_idx_long].T
            Y_pred = (W_sel * X_sel).sum(dim=1) / M_sqrt
            loss = ((Y - Y_pred) ** 2).mean().item()
            if abs(prev_loss - loss) < CONVERGENCE_THRESHOLD:
                break
            prev_loss = loss
    
    # Compute Q_Y
    qy = compute_qy(W, X, W_teacher, X_teacher)
    
    return qy, W, X


def run_continuation(
    sigma_init: float,
    device: torch.device,
    seed: int,
):
    """
    Run forward and backward continuation sweeps.
    
    Returns:
        dict with 'forward' and 'backward' results
    """
    alphas_forward = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP/2, ALPHA_STEP)
    alphas_backward = alphas_forward[::-1].copy()
    
    results = {
        'forward': {'alphas': [], 'qy': []},
        'backward': {'alphas': [], 'qy': []},
    }
    
    # Create FULL problem at max alpha (all edges, Y are fixed)
    full_problem = create_full_problem(device, seed)
    if full_problem is None:
        return results
    
    # ========================================================================
    # Forward sweep (α increasing) - edges are ADDED
    # ========================================================================
    print(f"  Forward sweep: α = {ALPHA_MIN} → {ALPHA_MAX}")
    
    W, X = None, None
    
    for alpha in alphas_forward:
        problem = get_problem_at_alpha(full_problem, alpha)
        
        # Initialize for first alpha, or use previous solution
        if W is None:
            W, X = initialize_matrices(
                sigma_init, full_problem['W_teacher'], full_problem['X_teacher'],
                device, seed
            )
        
        qy, W, X = train_at_alpha(problem, W, X)
        
        results['forward']['alphas'].append(alpha)
        results['forward']['qy'].append(qy)
        print(f"    α={alpha:.2f}: Q_Y={qy:.4f} (E={problem['E']})")
    
    # ========================================================================
    # Backward sweep (α decreasing) - edges are REMOVED
    # ========================================================================
    print(f"  Backward sweep: α = {ALPHA_MAX} → {ALPHA_MIN}")
    
    # W, X are already set from forward sweep
    
    for alpha in alphas_backward:
        problem = get_problem_at_alpha(full_problem, alpha)
        
        qy, W, X = train_at_alpha(problem, W, X)
        
        results['backward']['alphas'].append(alpha)
        results['backward']['qy'].append(qy)
        print(f"    α={alpha:.2f}: Q_Y={qy:.4f} (E={problem['E']})")
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGD Continuation Method - Hysteresis Analysis")
    print("(TRUE CONTINUATION: Standard Model, Forward/Backward)")
    print("Observation: Y = (1/√M) Σ_μ W_iμ X_μj")
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
    print(f"Alpha: {ALPHA_MIN} → {ALPHA_MAX} (step {ALPHA_STEP})")
    print(f"Epsilon_init values: {SIGMA_INIT_VALUES}")
    print(f"Replicas: {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_agd_continuation_hysteresis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'algorithm': 'agd_continuation_hysteresis',
        'observation_model': 'Y = (1/sqrt(M)) * sum_mu W_imu X_muj',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_min': ALPHA_MIN,
        'alpha_max': ALPHA_MAX,
        'alpha_step': ALPHA_STEP,
        'epsilon_init_values': [str(s) for s in SIGMA_INIT_VALUES],
        'student_init_formula': 'epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1)',
        'num_replicas': NUM_REPLICAS,
        'max_steps_per_alpha': MAX_STEPS_PER_ALPHA,
        'lr': LR,
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run for each epsilon_init
    all_results = {}
    start_time = time.time()
    
    for sigma_init in SIGMA_INIT_VALUES:
        s_label = "inf" if math.isinf(sigma_init) else f"{sigma_init}"
        print(f"\n{'='*50}")
        print(f"epsilon_init = {s_label}")
        print('='*50)
        
        # Aggregate over replicas
        forward_qy_all = []
        backward_qy_all = []
        alphas_list = None
        
        for rep in range(NUM_REPLICAS):
            seed = SEED + rep * 1000
            print(f"\n  Replica {rep+1}/{NUM_REPLICAS}")
            results = run_continuation(sigma_init, device, seed)
            
            forward_qy_all.append(results['forward']['qy'])
            backward_qy_all.append(results['backward']['qy'])
            if alphas_list is None:
                alphas_list = results['forward']['alphas']
        
        # Compute mean and std
        forward_qy_all = np.array(forward_qy_all)
        backward_qy_all = np.array(backward_qy_all)
        
        all_results[sigma_init] = {
            'forward': {
                'alphas': alphas_list,
                'mean': forward_qy_all.mean(axis=0),
                'std': forward_qy_all.std(axis=0),
            },
            'backward': {
                'alphas': list(reversed(alphas_list)),
                'mean': backward_qy_all.mean(axis=0),
                'std': backward_qy_all.std(axis=0),
            },
        }
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # ========================================================================
    # Plotting
    # ========================================================================
    print("\nGenerating plots...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    colors = ['#8E24AA', '#1E88E5', '#43A047', '#FB8C00', '#E53935', '#00ACC1', '#7CB342', '#757575']
    
    # Individual plots
    for idx, sigma_init in enumerate(SIGMA_INIT_VALUES):
        s_label = "inf" if math.isinf(sigma_init) else f"{sigma_init}"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r = all_results[sigma_init]
        
        # Forward (solid line)
        ax.errorbar(
            r['forward']['alphas'], r['forward']['mean'],
            yerr=r['forward']['std'] / np.sqrt(NUM_REPLICAS),
            fmt='o-', color=colors[idx % len(colors)], markersize=6, linewidth=2,
            label='Forward (α↑)', capsize=3
        )
        
        # Backward (dashed line)
        ax.errorbar(
            r['backward']['alphas'], r['backward']['mean'],
            yerr=r['backward']['std'] / np.sqrt(NUM_REPLICAS),
            fmt='s--', color=colors[idx % len(colors)], markersize=6, linewidth=2,
            alpha=0.7, label='Backward (α↓)', capsize=3
        )
        
        ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
        ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
        ax.set_title(f'AGD TRUE Continuation: σ_init={s_label}\n({N1}×{N2}, M={M})', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=12)
        
        plot_path = plots_dir / f"hysteresis_sigmaInit{s_label}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()
    
    # Combined plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, sigma_init in enumerate(SIGMA_INIT_VALUES):
        s_label = "∞ (Cold)" if math.isinf(sigma_init) else f"{sigma_init}" if sigma_init < 1 else "1 (Teacher)"
        r = all_results[sigma_init]
        
        ax.plot(r['forward']['alphas'], r['forward']['mean'],
                'o-', color=colors[idx % len(colors)], markersize=6, linewidth=2,
                label=f'σ_init={s_label} (→)')
        ax.plot(r['backward']['alphas'], r['backward']['mean'],
                's--', color=colors[idx % len(colors)], markersize=5, linewidth=1.5,
                alpha=0.6, label=f'σ_init={s_label} (←)')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'AGD TRUE Continuation: Hysteresis Analysis\n({N1}×{N2}, M={M})', fontsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    
    plot_path = plots_dir / "hysteresis_combined.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.show()
    
    print(f"\nResults saved to: {results_dir}")
    print("Done!")

# %%
