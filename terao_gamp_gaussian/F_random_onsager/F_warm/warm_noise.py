#!/usr/bin/env python
"""
G-AMP Warm Start with Noise (Onsager Correction Version).

Two noise sources:
1) Observation noise on Y (sigma_y) - fixed
2) Initialization noise on W/X (sigma_init) - variable

Uses gamp_step_with_F_onsager for proper Onsager correction.
Q_Y is computed against the noise-free teacher matrices.
"""

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

from terao_gamp_gaussian.graph import BiregularGraph
from terao_gamp_gaussian.F_random_onsager.core import gamp_step_with_F_onsager
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy

# ============================================================================
# Configuration (Dense Limit: N >> M >> 1)
# ============================================================================

N1 = 3000
N2 = 3000
M = 30

ALPHA_START = 0.5
ALPHA_STOP = 5.0
ALPHA_STEP = 0.5

MAX_STEPS = 5000
DAMPING = 0.5
SEED = 42
NUM_REPLICAS = 1  # For quick verification

# Observation noise (standard deviation) - FIXED
SIGMA_Y = 0.1

# Initialization noise (standard deviation) - VARIABLE
# inf = Cold Start (random init), 0 = Perfect Warm Start
SIGMA_INIT_VALUES = [1.0, 0.5, 0.1, 0.0]

CONVERGENCE_THRESHOLD = 1e-10

# ============================================================================
# G-AMP Training with Warm Start and Onsager Correction
# ============================================================================

def train_single_replica(
    alpha: float,
    sigma_init: float,
    sigma_y: float,
    device: torch.device,
    seed: int,
):
    """
    Train a single replica with observation noise and warm-start noise.
    Uses Onsager correction for proper G-AMP convergence.
    
    Args:
        alpha: Observation density
        sigma_init: Initialization noise strength
            - inf: Cold Start (random initialization)
            - 0: Start exactly at teacher
            - >0: Teacher + sigma_init * noise
        sigma_y: Observation noise standard deviation
        device: torch device
        seed: Random seed
    
    Returns:
        tuple: (qy, final_loss, steps_taken)
    """
    # Convert sigma_y to variance for G-AMP
    noise_var = sigma_y ** 2
    
    # Generate teacher matrices
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate graph (BiregularGraph for Dense Limit)
    graph = BiregularGraph()
    i_idx, j_idx, E, C1, C2, alpha2 = graph.generate(N1, N2, M, alpha, device, seed)
    
    if E == 0:
        return 0.0, 0.0, 0
    
    # Generate spreading matrix F: (E, M) with F[c,μ] ~ N(0,1)
    torch.manual_seed(seed + 500)
    F = torch.randn(E, M, device=device, dtype=torch.float32)
    
    # Generate clean observations: Y = (1/√M) Σ_μ F[c,μ] W X
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y_clean = (1.0 / math.sqrt(M)) * (F * W_sel * X_sel).sum(dim=1)
    
    # Add observation noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y_clean) * sigma_y
    Y_noisy = Y_clean + noise
    
    # Initialize messages based on sigma_init
    torch.manual_seed(seed + 2000)
    
    if math.isinf(sigma_init):
        # Cold Start: random initialization (small values)
        m_W = torch.randn(N1, M, device=device) * 0.1
        m_X = torch.randn(M, N2, device=device) * 0.1
        v_W = torch.ones(N1, M, device=device)
        v_X = torch.ones(M, N2, device=device)
    elif sigma_init == 0.0:
        # Perfect Warm Start: start exactly at teacher
        m_W = W_teacher.clone()
        m_X = X_teacher.clone()
        v_W = m_W ** 2 + 0.01  # Small variance for stability
        v_X = m_X ** 2 + 0.01
    else:
        # Warm Start with Noise: teacher + perturbation
        m_W = W_teacher + sigma_init * torch.randn(N1, M, device=device)
        m_X = X_teacher + sigma_init * torch.randn(M, N2, device=device)
        m_W = normalize_to_unit_variance(m_W)
        m_X = normalize_to_unit_variance(m_X)
        v_W = m_W ** 2 + sigma_init ** 2
        v_X = m_X ** 2 + sigma_init ** 2
    
    g_prev = torch.zeros(E, device=device)
    m_W_prev = m_W.clone()
    m_X_prev = m_X.clone()
    
    # G-AMP iterations with Onsager correction
    final_loss = 0.0
    steps_taken = MAX_STEPS
    prev_loss = float('inf')
    
    for step in range(MAX_STEPS):
        m_W_old = m_W.clone()
        m_X_old = m_X.clone()
        
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_F_onsager(
            m_W, v_W, m_X, v_X, m_W_prev, m_X_prev,
            Y_noisy, F, i_idx, j_idx, g_prev,
            noise_var, DAMPING, N1, N2, M
        )
        
        m_W_prev = m_W_old
        m_X_prev = m_X_old
        
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
    
    # Normalize and compute Q_Y against TRUE teacher
    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)
    qy = compute_qy(m_W, m_X, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("G-AMP Warm Start with Noise (Onsager Correction)")
    print("Phase Transition Analysis")
    print("=" * 70)
    
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
    
    print(f"Matrix: {N1}×{N2}, M={M} (Dense Limit: N >> M >> 1)")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Sigma_Y (observation noise): {SIGMA_Y}")
    print(f"Sigma_init values: {SIGMA_INIT_VALUES}")
    print(f"Replicas per (alpha, sigma_init): {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_warm_noise_onsager_sigmaY{SIGMA_Y}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Save configuration
    config = {
        'algorithm': 'gamp_warm_noise_onsager',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_start': ALPHA_START,
        'alpha_stop': ALPHA_STOP,
        'alpha_step': ALPHA_STEP,
        'sigma_y': SIGMA_Y,
        'sigma_init_values': [str(s) for s in SIGMA_INIT_VALUES],
        'num_replicas': NUM_REPLICAS,
        'max_steps': MAX_STEPS,
        'damping': DAMPING,
        'convergence_threshold': CONVERGENCE_THRESHOLD,
        'device': str(device),
        'onsager_correction': True,
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {s: {} for s in SIGMA_INIT_VALUES}
    
    total_tasks = len(alphas) * len(SIGMA_INIT_VALUES) * NUM_REPLICAS
    completed = 0
    start_time = time.time()
    
    for sigma_init in SIGMA_INIT_VALUES:
        s_label = "inf" if math.isinf(sigma_init) else f"{sigma_init}"
        print(f"\n--- sigma_init = {s_label} ---")
        
        for alpha in alphas:
            qy_values = []
            loss_values = []
            for rep in range(NUM_REPLICAS):
                seed = SEED + rep * 1000
                t0 = time.time()
                qy, loss, steps = train_single_replica(alpha, sigma_init, SIGMA_Y, device, seed)
                dt = time.time() - t0
                qy_values.append(qy)
                loss_values.append(loss)
                completed += 1
                print(f"σ_init={s_label}, α={alpha:.2f}: Q_Y={qy:.4f}, Loss={loss:.2e}, "
                      f"Steps={steps} ({dt:.1f}s) [{completed}/{total_tasks}]")
            
            mean_qy = np.mean(qy_values)
            std_qy = np.std(qy_values)
            mean_loss = np.mean(loss_values)
            results[sigma_init][alpha] = {
                'mean': mean_qy, 'std': std_qy, 
                'values': qy_values, 'loss_mean': mean_loss
            }
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Create plots
    print("\nGenerating plots...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    colors = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, sigma_init in enumerate(SIGMA_INIT_VALUES):
        s_label = "∞ (Cold)" if math.isinf(sigma_init) else f"{sigma_init}" if sigma_init > 0 else "0 (Teacher)"
        
        alphas_list = sorted(results[sigma_init].keys())
        means = [results[sigma_init][a]['mean'] for a in alphas_list]
        stds = [results[sigma_init][a]['std'] for a in alphas_list]
        sems = [s / np.sqrt(NUM_REPLICAS) if NUM_REPLICAS > 1 else 0 for s in stds]
        
        ax.errorbar(alphas_list, means, yerr=sems,
                    fmt=f'{markers[idx]}-', color=colors[idx],
                    markersize=8, linewidth=2,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label=f'σ_init = {s_label}')
    
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$ (reconstruction quality)', fontsize=14)
    ax.set_title(f'G-AMP Warm Start with Onsager Correction\n({N1}×{N2}, M={M}, σ_Y={SIGMA_Y})', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plot_path = plots_dir / f"qy_vs_alpha_warm_onsager.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.show()
    
    # Save CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w') as f:
        header = "alpha"
        for s in SIGMA_INIT_VALUES:
            s_str = "inf" if math.isinf(s) else f"{s}"
            header += f",Q_Y_mean_sigmaInit{s_str},Q_Y_std_sigmaInit{s_str},Loss_mean_sigmaInit{s_str}"
        f.write(header + "\n")
        
        for alpha in sorted(alphas):
            line = f"{alpha}"
            for s in SIGMA_INIT_VALUES:
                r = results[s].get(alpha, {'mean': 0, 'std': 0, 'loss_mean': 0})
                line += f",{r['mean']},{r['std']},{r['loss_mean']}"
            f.write(line + "\n")
    
    print(f"Metrics saved: {csv_path}")
    print(f"Results saved to: {results_dir}")
    print("Done!")
