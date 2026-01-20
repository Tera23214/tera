#!/usr/bin/env python
"""
Comparison: F=1 vs F~N(0,1) with Onsager Correction.

Runs both G-AMP algorithms and compares the phase transition behavior.
Includes both Q_Y and Loss comparison plots.
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
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.F_1_onsager.core import train_single_replica as train_F1
from terao_gamp_gaussian.F_random_onsager.core import train_single_replica as train_Frandom

# ============================================================================
# Configuration
# ============================================================================

N1 = 8000
N2 = 8000
M = 50

ALPHA_START = 0.5
ALPHA_STOP = 10.0
ALPHA_STEP = 0.5

MAX_STEPS = 500
DAMPING = 0.5
NOISE_VAR = 1e-10
SEED = 42
NUM_REPLICAS = 5
CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Comparison: G-AMP with F=1 vs F~N(0,1) (both with Onsager)")
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
    
    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Replicas: {NUM_REPLICAS}")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"{timestamp}_compare_F1_vs_Frandom_{N1}x{M}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")
    
    # Save configuration
    config = {
        'comparison': 'F1_vs_Frandom_onsager',
        'N1': N1, 'N2': N2, 'M': M,
        'alpha_range': f"{ALPHA_START}-{ALPHA_STOP}",
        'alpha_step': ALPHA_STEP,
        'max_steps': MAX_STEPS,
        'damping': DAMPING,
        'noise_var': NOISE_VAR,
        'num_replicas': NUM_REPLICAS,
    }
    with open(results_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run simulations
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results_F1 = {}
    results_Frandom = {}
    
    start_time = time.time()
    
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"Alpha = {alpha:.2f}")
        print("-" * 70)
        
        # F=1 simulations
        qy_F1 = []
        loss_F1 = []
        print("  F=1 (constant):")
        for r in range(NUM_REPLICAS):
            seed = SEED + r * 1000
            qy, loss, steps = train_F1(
                alpha=alpha, device=device, seed=seed,
                N1=N1, N2=N2, M=M, max_steps=MAX_STEPS,
                damping=DAMPING, noise_var=NOISE_VAR,
                convergence_threshold=CONVERGENCE_THRESHOLD,
            )
            qy_F1.append(qy)
            loss_F1.append(loss)
            print(f"    Replica {r+1}: Q_Y={qy:.4f}, Loss={loss:.2e}, Steps={steps}")
        
        results_F1[alpha] = {
            'qy_mean': np.mean(qy_F1),
            'qy_std': np.std(qy_F1),
            'qy_values': qy_F1,
            'loss_mean': np.mean(loss_F1),
            'loss_std': np.std(loss_F1),
            'loss_values': loss_F1,
        }
        
        # F~N(0,1) simulations
        qy_Frandom = []
        loss_Frandom = []
        print("  F~N(0,1) (random):")
        for r in range(NUM_REPLICAS):
            seed = SEED + r * 1000
            qy, loss, steps = train_Frandom(
                alpha=alpha, device=device, seed=seed,
                N1=N1, N2=N2, M=M, max_steps=MAX_STEPS,
                damping=DAMPING, noise_var=NOISE_VAR,
                convergence_threshold=CONVERGENCE_THRESHOLD,
            )
            qy_Frandom.append(qy)
            loss_Frandom.append(loss)
            print(f"    Replica {r+1}: Q_Y={qy:.4f}, Loss={loss:.2e}, Steps={steps}")
        
        results_Frandom[alpha] = {
            'qy_mean': np.mean(qy_Frandom),
            'qy_std': np.std(qy_Frandom),
            'qy_values': qy_Frandom,
            'loss_mean': np.mean(loss_Frandom),
            'loss_std': np.std(loss_Frandom),
            'loss_values': loss_Frandom,
        }
        
        print(f"  Q_Y:  F=1: {results_F1[alpha]['qy_mean']:.4f}±{results_F1[alpha]['qy_std']:.4f} | "
              f"F~N(0,1): {results_Frandom[alpha]['qy_mean']:.4f}±{results_Frandom[alpha]['qy_std']:.4f}")
        print(f"  Loss: F=1: {results_F1[alpha]['loss_mean']:.2e} | "
              f"F~N(0,1): {results_Frandom[alpha]['loss_mean']:.2e}")
    
    total_time = time.time() - start_time
    
    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY: F=1 vs F~N(0,1) (both with Onsager)")
    print("=" * 90)
    print(f"{'Alpha':>6} | {'F=1 (Q_Y)':^16} | {'F~N(0,1) (Q_Y)':^16} | {'F=1 (Loss)':^12} | {'F~N(0,1) (Loss)':^12}")
    print("-" * 90)
    for alpha in sorted(results_F1.keys()):
        r1 = results_F1[alpha]
        r2 = results_Frandom[alpha]
        print(f"{alpha:6.2f} | {r1['qy_mean']:6.4f} ± {r1['qy_std']:<6.4f} | "
              f"{r2['qy_mean']:6.4f} ± {r2['qy_std']:<6.4f} | {r1['loss_mean']:10.2e} | {r2['loss_mean']:10.2e}")
    
    print(f"\nTotal time: {total_time/60:.1f} min")
    
    alphas_list = sorted(results_F1.keys())
    
    # Create comparison plots (2 subplots: Q_Y and Loss)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== Q_Y Plot ==========
    ax1 = axes[0]
    
    means_F1_qy = [results_F1[a]['qy_mean'] for a in alphas_list]
    sems_F1_qy = [results_F1[a]['qy_std'] / math.sqrt(NUM_REPLICAS) for a in alphas_list]
    means_Frandom_qy = [results_Frandom[a]['qy_mean'] for a in alphas_list]
    sems_Frandom_qy = [results_Frandom[a]['qy_std'] / math.sqrt(NUM_REPLICAS) for a in alphas_list]
    
    ax1.errorbar(alphas_list, means_F1_qy, yerr=sems_F1_qy,
                fmt='o-', color='#1976D2', markersize=6, linewidth=2,
                capsize=3, capthick=1, label='F=1 + Onsager')
    ax1.errorbar(alphas_list, means_Frandom_qy, yerr=sems_Frandom_qy,
                fmt='s--', color='#E53935', markersize=6, linewidth=2,
                capsize=3, capthick=1, label='F~N(0,1) + Onsager')
    
    ax1.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax1.set_ylabel(r'$Q_Y$', fontsize=14)
    ax1.set_title(f'Q_Y Comparison\n(N={N1}, M={M})', fontsize=14)
    ax1.set_xlim(ALPHA_START - 0.2, ALPHA_STOP + 0.2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=11)
    
    # ========== Loss Plot ==========
    ax2 = axes[1]
    
    means_F1_loss = [results_F1[a]['loss_mean'] for a in alphas_list]
    sems_F1_loss = [results_F1[a]['loss_std'] / math.sqrt(NUM_REPLICAS) for a in alphas_list]
    means_Frandom_loss = [results_Frandom[a]['loss_mean'] for a in alphas_list]
    sems_Frandom_loss = [results_Frandom[a]['loss_std'] / math.sqrt(NUM_REPLICAS) for a in alphas_list]
    
    ax2.errorbar(alphas_list, means_F1_loss, yerr=sems_F1_loss,
                fmt='o-', color='#1976D2', markersize=6, linewidth=2,
                capsize=3, capthick=1, label='F=1 + Onsager')
    ax2.errorbar(alphas_list, means_Frandom_loss, yerr=sems_Frandom_loss,
                fmt='s--', color='#E53935', markersize=6, linewidth=2,
                capsize=3, capthick=1, label='F~N(0,1) + Onsager')
    
    ax2.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax2.set_ylabel('Loss (MSE)', fontsize=14)
    ax2.set_title(f'Loss Comparison\n(N={N1}, M={M})', fontsize=14)
    ax2.set_xlim(ALPHA_START - 0.2, ALPHA_STOP + 0.2)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    plot_path = results_dir / "comparison_F1_vs_Frandom.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")
    plt.show()
    
    # Save CSV with Loss data
    csv_path = results_dir / "comparison_results.csv"
    with open(csv_path, 'w') as f:
        f.write("alpha,F1_qy_mean,F1_qy_std,Frandom_qy_mean,Frandom_qy_std,F1_loss_mean,F1_loss_std,Frandom_loss_mean,Frandom_loss_std,qy_diff\n")
        for alpha in alphas_list:
            r1, r2 = results_F1[alpha], results_Frandom[alpha]
            f.write(f"{alpha},{r1['qy_mean']},{r1['qy_std']},{r2['qy_mean']},{r2['qy_std']},"
                    f"{r1['loss_mean']},{r1['loss_std']},{r2['loss_mean']},{r2['loss_std']},"
                    f"{r1['qy_mean']-r2['qy_mean']}\n")
    
    print(f"CSV saved: {csv_path}")
    print("\nDone!")
