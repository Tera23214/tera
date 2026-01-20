#!/usr/bin/env python
"""
G-AMP Continuation Method for Hysteresis Analysis.

TRUE CONTINUATION: Uses consistent F matrix, incremental graph, and shared observations.

- Forward sweep: α increases, edges are ADDED to the existing graph
- Backward sweep: α decreases, edges are REMOVED from the graph

For each step, the solution from the previous α is used as the initial value.
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
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.F_random.F_random_core.core import gamp_step_with_F
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy

# ============================================================================
# Configuration
# ============================================================================

N1 = 3000
N2 = 3000
M = 30

ALPHA_START = 0.5
ALPHA_STOP = 5.0
ALPHA_STEP = 0.5

MAX_STEPS = 500
DAMPING = 0.5
SEED = 42
NUM_REPLICAS = 10

# Observation noise (standard deviation)
SIGMA_Y = 1e-10

# Initialization noise for the first alpha (start of forward sweep)
# inf = Cold Start, 0 = Perfect Warm Start
SIGMA_INIT_VALUES = [float('inf'), 6.0, 3.0, 0.0]

CONVERGENCE_THRESHOLD = 1e-6


# ============================================================================
# Graph Generation with Consistent Edge Ordering
# ============================================================================

def generate_max_graph(N1: int, N2: int, M: int, alpha_max: float, device: torch.device, seed: int):
    """
    Generate the maximum alpha graph. All smaller alpha graphs are subsets.
    
    Returns:
        i_idx, j_idx: Full edge lists for alpha_max
        E_max: Total number of edges at alpha_max
    """
    C1_max = int(round(alpha_max * M))
    C1_max = max(1, min(C1_max, N2))
    
    np.random.seed(seed)
    
    i_list = []
    j_list = []
    
    for i in range(N1):
        # Randomly select C1_max columns for this row
        selected = np.random.choice(N2, size=C1_max, replace=False)
        for j in selected:
            i_list.append(i)
            j_list.append(j)
    
    i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
    j_idx = torch.tensor(j_list, dtype=torch.long, device=device)
    E_max = len(i_idx)
    
    return i_idx, j_idx, E_max


def get_edge_subset(i_idx_full: torch.Tensor, j_idx_full: torch.Tensor, 
                    N1: int, M: int, alpha: float, alpha_max: float):
    """
    Get edge subset for a specific alpha (alpha <= alpha_max).
    
    Edges are selected consistently: first C1 edges per row where C1 = alpha * M.
    """
    C1 = int(round(alpha * M))
    C1 = max(1, C1)
    
    # C1_max is computed from actual edge count to handle clamping
    C1_max = int(round(alpha_max * M))
    C1_max = max(1, min(C1_max, N2, len(j_idx_full) // N1))
    
    # C1 cannot exceed C1_max
    C1 = min(C1, C1_max)
    
    # Select first C1 edges per row (out of C1_max)
    mask = []
    for i in range(N1):
        start_idx = i * C1_max
        end_idx = start_idx + min(C1, C1_max)
        mask.extend(range(start_idx, end_idx))
    
    mask = torch.tensor(mask, dtype=torch.long, device=i_idx_full.device)
    
    return i_idx_full[mask], j_idx_full[mask], len(mask)


# ============================================================================
# Problem Setup (consistent across all alpha)
# ============================================================================

def create_full_problem(alpha_max: float, device: torch.device, seed: int):
    """
    Create the full problem instance for alpha_max.
    All smaller alpha problems are subsets of this.
    
    Returns:
        dict with all problem data
    """
    noise_var = SIGMA_Y ** 2
    
    # Generate teacher (fixed for all alpha)
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate max graph (all edges)
    i_idx_full, j_idx_full, E_max = generate_max_graph(N1, N2, M, alpha_max, device, seed)
    
    # Generate F matrix for ALL edges (fixed)
    torch.manual_seed(seed + 500)
    F_full = torch.randn(E_max, M, device=device, dtype=torch.float32)
    F_sq_full = F_full ** 2
    
    # Generate clean observations for ALL edges
    W_sel = W_teacher[i_idx_full.long(), :]
    X_sel = X_teacher[:, j_idx_full.long()].T
    Y_clean_full = (1.0 / math.sqrt(M)) * (F_full * W_sel * X_sel).sum(dim=1)
    
    # Add observation noise (fixed)
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y_clean_full) * SIGMA_Y
    Y_noisy_full = Y_clean_full + noise
    
    return {
        'W_teacher': W_teacher,
        'X_teacher': X_teacher,
        'i_idx_full': i_idx_full,
        'j_idx_full': j_idx_full,
        'F_full': F_full,
        'F_sq_full': F_sq_full,
        'Y_noisy_full': Y_noisy_full,
        'noise_var': noise_var,
        'E_max': E_max,
        'alpha_max': alpha_max,
    }


def get_problem_at_alpha(full_problem: dict, alpha: float):
    """
    Get problem data for a specific alpha by subsetting the full problem.
    """
    alpha_max = full_problem['alpha_max']
    
    i_idx, j_idx, E = get_edge_subset(
        full_problem['i_idx_full'], full_problem['j_idx_full'],
        N1, M, alpha, alpha_max
    )
    
    # Get corresponding F, Y subsets
    # IMPORTANT: C1_max must match generate_max_graph's clamping
    C1_max = int(round(alpha_max * M))
    C1_max = max(1, min(C1_max, N2))  # Same clamping as generate_max_graph
    C1 = int(round(alpha * M))
    C1 = max(1, min(C1, C1_max))  # C1 cannot exceed C1_max
    
    # Build mask for edge selection
    mask = []
    for i in range(N1):
        start_idx = i * C1_max
        end_idx = start_idx + min(C1, C1_max)
        mask.extend(range(start_idx, end_idx))
    
    mask = torch.tensor(mask, dtype=torch.long, device=i_idx.device)
    
    return {
        'i_idx': i_idx,
        'j_idx': j_idx,
        'F': full_problem['F_full'][mask],
        'F_sq': full_problem['F_sq_full'][mask],
        'Y_noisy': full_problem['Y_noisy_full'][mask],
        'noise_var': full_problem['noise_var'],
        'W_teacher': full_problem['W_teacher'],
        'X_teacher': full_problem['X_teacher'],
        'E': E,
    }


# ============================================================================
# Training Functions
# ============================================================================

def initialize_messages(
    sigma_init: float,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    device: torch.device,
    seed: int,
):
    """Initialize messages based on sigma_init."""
    torch.manual_seed(seed + 2000)
    
    if math.isinf(sigma_init):
        # Cold Start
        m_W = torch.randn(N1, M, device=device) * 0.1
        m_X = torch.randn(M, N2, device=device) * 0.1
        v_W = torch.ones(N1, M, device=device)
        v_X = torch.ones(M, N2, device=device)
    elif sigma_init == 0.0:
        # Perfect Warm Start
        m_W = W_teacher.clone()
        m_X = X_teacher.clone()
        v_W = m_W ** 2 + 0.01
        v_X = m_X ** 2 + 0.01
    else:
        # Warm Start with perturbation
        m_W = W_teacher + sigma_init * torch.randn(N1, M, device=device)
        m_X = X_teacher + sigma_init * torch.randn(M, N2, device=device)
        m_W = normalize_to_unit_variance(m_W)
        m_X = normalize_to_unit_variance(m_X)
        v_W = m_W ** 2 + sigma_init ** 2
        v_X = m_X ** 2 + sigma_init ** 2
    
    return m_W, v_W, m_X, v_X


def train_at_alpha(
    problem: dict,
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    g_prev: torch.Tensor = None,
):
    """
    Train at a specific alpha using given initial messages.
    """
    F = problem['F']
    F_sq = problem['F_sq']
    i_idx = problem['i_idx']
    j_idx = problem['j_idx']
    Y_noisy = problem['Y_noisy']
    noise_var = problem['noise_var']
    W_teacher = problem['W_teacher']
    X_teacher = problem['X_teacher']
    E = problem['E']
    
    if g_prev is None:
        g_prev = torch.zeros(E, device=m_W.device)
    elif len(g_prev) != E:
        # Properly inherit g_prev:
        # - If E > len(g_prev): Forward sweep, extend with zeros for new edges
        # - If E < len(g_prev): Backward sweep, truncate (keep first E values)
        # Edge ordering is consistent (first C1 edges per row), so this works correctly
        if E > len(g_prev):
            # Extend: new edges get g=0
            g_prev_new = torch.zeros(E, device=m_W.device)
            g_prev_new[:len(g_prev)] = g_prev
            g_prev = g_prev_new
        else:
            # Truncate: keep first E values
            g_prev = g_prev[:E].clone()
    
    # G-AMP iterations
    prev_loss = float('inf')
    
    for step in range(MAX_STEPS):
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_F(
            m_W, v_W, m_X, v_X,
            Y_noisy, F, F_sq, i_idx, j_idx, g_prev,
            noise_var, DAMPING, N1, N2, M
        )
        
        # Check convergence
        if step % 50 == 0 or step == MAX_STEPS - 1:
            W_sel_pred = m_W[i_idx.long(), :]
            X_sel_pred = m_X[:, j_idx.long()].T
            Y_pred = (1.0 / math.sqrt(M)) * (F * W_sel_pred * X_sel_pred).sum(dim=1)
            loss = ((Y_noisy - Y_pred) ** 2).mean().item()
            
            if abs(prev_loss - loss) < CONVERGENCE_THRESHOLD:
                break
            prev_loss = loss
    
    # Compute Q_Y
    m_W_norm = normalize_to_unit_variance(m_W.clone())
    m_X_norm = normalize_to_unit_variance(m_X.clone())
    qy = compute_qy(m_W_norm, m_X_norm, W_teacher, X_teacher)
    
    return qy, m_W, v_W, m_X, v_X, g_prev


def run_continuation(
    sigma_init: float,
    device: torch.device,
    seed: int,
):
    """
    Run forward and backward continuation sweeps with CONSISTENT problem.
    """
    alphas_forward = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    alphas_backward = alphas_forward[::-1].copy()
    
    results = {
        'forward': {'alphas': [], 'qy': []},
        'backward': {'alphas': [], 'qy': []},
    }
    
    # Create FULL problem at max alpha (all edges, F, Y are fixed)
    full_problem = create_full_problem(ALPHA_STOP, device, seed)
    
    # ========================================================================
    # Forward sweep (α increasing) - edges are ADDED
    # ========================================================================
    print(f"  Forward sweep: α = {ALPHA_START} → {ALPHA_STOP}")
    
    m_W, v_W, m_X, v_X = None, None, None, None
    g_prev = None  # Track g_prev across alpha steps
    
    for alpha in alphas_forward:
        problem = get_problem_at_alpha(full_problem, alpha)
        
        # Initialize for first alpha, or use previous solution
        if m_W is None:
            m_W, v_W, m_X, v_X = initialize_messages(
                sigma_init, full_problem['W_teacher'], full_problem['X_teacher'],
                device, seed
            )
        
        qy, m_W, v_W, m_X, v_X, g_prev = train_at_alpha(
            problem, m_W, v_W, m_X, v_X, g_prev
        )
        
        results['forward']['alphas'].append(alpha)
        results['forward']['qy'].append(qy)
        print(f"    α={alpha:.2f}: Q_Y={qy:.4f} (E={problem['E']})")
    
    # ========================================================================
    # Backward sweep (α decreasing) - edges are REMOVED
    # ========================================================================
    print(f"  Backward sweep: α = {ALPHA_STOP} → {ALPHA_START}")
    
    # m_W, v_W, m_X, v_X, g_prev are already set from forward sweep
    
    for alpha in alphas_backward:
        problem = get_problem_at_alpha(full_problem, alpha)
        
        qy, m_W, v_W, m_X, v_X, g_prev = train_at_alpha(
            problem, m_W, v_W, m_X, v_X, g_prev
        )
        
        results['backward']['alphas'].append(alpha)
        results['backward']['qy'].append(qy)
        print(f"    α={alpha:.2f}: Q_Y={qy:.4f} (E={problem['E']})")
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("G-AMP Continuation Method - Hysteresis Analysis")
    print("(TRUE CONTINUATION: Consistent F, Incremental Graph)")
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
    print(f"Alpha: {ALPHA_START} → {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Sigma_Y: {SIGMA_Y}")
    print(f"Sigma_init values: {SIGMA_INIT_VALUES}")
    print(f"Replicas: {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_continuation_true_sigmaY{SIGMA_Y}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'algorithm': 'gamp_continuation_true',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'sigma_y': SIGMA_Y,
        'sigma_init_values': [str(s) for s in SIGMA_INIT_VALUES],
        'num_replicas': NUM_REPLICAS,
        'max_steps': MAX_STEPS,
        'damping': DAMPING,
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run for each sigma_init
    all_results = {}
    start_time = time.time()
    
    for sigma_init in SIGMA_INIT_VALUES:
        s_label = "inf" if math.isinf(sigma_init) else f"{sigma_init}"
        print(f"\n{'='*50}")
        print(f"sigma_init = {s_label}")
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
    
    colors = ['#E53935', '#FB8C00', '#FDD835', '#43A047', '#1E88E5', '#8E24AA', '#795548', '#00BCD4', '#607D8B', '#FF5722']
    
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
        ax.set_title(f'G-AMP TRUE Continuation: σ_init={s_label}\n({N1}×{N2}, M={M}, σ_Y={SIGMA_Y})', fontsize=14)
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
        s_label = "∞ (Cold)" if math.isinf(sigma_init) else f"{sigma_init}" if sigma_init > 0 else "0 (Teacher)"
        r = all_results[sigma_init]
        
        ax.plot(r['forward']['alphas'], r['forward']['mean'],
                'o-', color=colors[idx % len(colors)], markersize=6, linewidth=2,
                label=f'σ_init={s_label} (→)')
        ax.plot(r['backward']['alphas'], r['backward']['mean'],
                's--', color=colors[idx % len(colors)], markersize=5, linewidth=1.5,
                alpha=0.6, label=f'σ_init={s_label} (←)')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'G-AMP TRUE Continuation: Hysteresis Analysis\n({N1}×{N2}, M={M}, σ_Y={SIGMA_Y})', fontsize=16)
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
