#!/usr/bin/env python
"""
G-AMP Simulation Runner.

Runs G-AMP algorithm for various alpha values and plots Q_Y vs alpha.
Based on the structure of terao_gd/gd.py.

Usage:
    python terao_gamp/run_gamp.py
"""

import sys
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp.core import train_single_replica

# ============================================================================
# Configuration
# ============================================================================

N1 = 1000   # Number of rows
N2 = 1000   # Number of columns  
M = 10      # Rank (hidden dimension)

ALPHA_START = 0.5
ALPHA_STOP = 5.0
ALPHA_STEP = 0.5

MAX_STEPS = 500         # G-AMP iterations
DAMPING = 0.5           # Message damping
NOISE_VAR = 1e-10       # Noise variance
SEED = 42
NUM_REPLICAS = 10       # Number of replicas per alpha
CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("G-AMP - Generalized Approximate Message Passing")
    print("Sparse Matrix Factorization")
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
    print(f"Steps: {MAX_STEPS}, Damping: {DAMPING}")
    print(f"Replicas per alpha: {NUM_REPLICAS}")
    print()
    
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
            
            qy, final_loss, steps_taken = train_single_replica(
                alpha=alpha,
                device=device,
                seed=seed,
                N1=N1,
                N2=N2,
                M=M,
                max_steps=MAX_STEPS,
                damping=DAMPING,
                noise_var=NOISE_VAR,
                convergence_threshold=CONVERGENCE_THRESHOLD,
            )
            
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
    
    # Plot Q_Y vs Alpha with error bars
    print("\nGenerating plots...")
    
    alphas_list = sorted(results.keys())
    qy_means = [results[a]['qy_mean'] for a in alphas_list]
    qy_stds = [results[a]['qy_std'] for a in alphas_list]
    qy_sems = [std / math.sqrt(NUM_REPLICAS) for std in qy_stds]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.errorbar(alphas_list, qy_means, yerr=qy_sems, 
                fmt='o-', color='#E53935', markersize=6, linewidth=2,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='G-AMP')
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(f'Phase Transition (G-AMP)\n({N1}×{N2}, M={M}, {MAX_STEPS} steps, {NUM_REPLICAS} replicas)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    # Save with parameters in filename
    sample_size = len(alphas_list)
    base_name = f"qy_vs_alpha_gamp_N1{N1}_N2{N2}_M{M}_samples{sample_size}_replicas{NUM_REPLICAS}"
    
    output_path = Path(__file__).parent / f"{base_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    # Save results to CSV
    csv_path = Path(__file__).parent / f"{base_name}.csv"
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
    
    print(f"CSV saved to: {csv_path}")
    print("Done!")
