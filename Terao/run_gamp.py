#!/usr/bin/env python
"""
Runner script for gamp_algo.py (BiG-AMP Spreading Parallel).

This script adds the parent directory to Python path so that
the smf package can be imported, then runs the algorithm.

Usage:
    cd /Users/password-is-0000/Projects/Sparse-Matrix-Factorization/Terao
    python run_gamp.py
"""
#%%

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory (Sparse-Matrix-Factorization) to path so smf can be imported
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Now we can import from smf
from smf.modules.algorithms.gamp_algo import (
    BiGAMPSpreadingParallel,
    run_spreading_parallel,
)
from smf.core.config import Config, MatrixConfig, AlphaConfig, TrainingConfig, AlgorithmConfig

# ============================================================================
# Configuration
# ============================================================================

# Matrix dimensions
N1 = 2000  # Number of rows
N2 = 2000  # Number of columns  
M = 20    # Rank (hidden dimension)

# Alpha (observation density) range
ALPHA_START = 0.1
ALPHA_STOP = 10.0
ALPHA_STEP = 0.5

# Training parameters
MAX_STEPS = 500       # BiG-AMP iterations (200-1000 typical)
SAMPLES_PER_ALPHA = 100  # Number of random initializations
SEED = 42

# Algorithm parameters
DAMPING = 0.5         # Message damping (0.5 is typical)
NOISE_VAR = 1e-10     # Assumed noise variance

# ============================================================================
# Run Experiment
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BiG-AMP Spreading Parallel - Terao Runner")
    print("=" * 60)
    
    # Create configuration
    config = Config(
        algorithm_key="bigamp_spreading_parallel",
        graph_key="random",
        teacher_key="standard",
        matrix=MatrixConfig(N1=N1, N2=N2, M=M),
        alpha=AlphaConfig(start=ALPHA_START, stop=ALPHA_STOP, step=ALPHA_STEP),
        training=TrainingConfig(
            max_steps=MAX_STEPS,
            samples_per_alpha=SAMPLES_PER_ALPHA,
            seed=SEED,
        ),
        algorithm=AlgorithmConfig(
            damping=DAMPING,
            noise_var=NOISE_VAR,
        ),
    )
    
    print(f"Matrix: {N1}×{N2}, M={M}")
    print(f"Alpha: {ALPHA_START} ~ {ALPHA_STOP} (step {ALPHA_STEP})")
    print(f"Steps: {MAX_STEPS}, Samples: {SAMPLES_PER_ALPHA}")
    print()
    
    # Run experiment
    result = run_spreading_parallel(config, verbose=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    results = result['results']
    for alpha in sorted(results.keys()):
        metrics = results[alpha]
        print(f"α={alpha:.2f}: Q_Y={metrics['Q_Y_mean']:.4f} ± {metrics['Q_Y_std']:.4f}")
    
    print(f"\nTotal time: {result['total_time']:.1f}s")
    
    # ============================================================================
    # Plot Q_Y vs Alpha
    # ============================================================================
    print("\nGenerating plot...")
    
    # Extract data for plotting
    alphas = sorted(results.keys())
    qy_means = [results[a]['Q_Y_mean'] for a in alphas]
    qy_stds = [results[a]['Q_Y_std'] for a in alphas]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot Q_Y with error bars
    ax.errorbar(
        alphas, qy_means, yerr=qy_stds,
        fmt='o-', color='#E53935', markersize=6,
        capsize=3, capthick=1, linewidth=2,
        label=r'$Q_Y$ (reconstruction overlap)'
    )
    
    # Style the plot
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(f'Phase Transition Curve\n({N1}×{N2}, M={M}, {MAX_STEPS} steps)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    # Add annotation for phase transition
    # Find approximate transition point (where Q_Y crosses 0.5)
    for i, (a, q) in enumerate(zip(alphas, qy_means)):
        if q > 0.5 and i > 0:
            alpha_c = (alphas[i-1] + a) / 2
            ax.axvline(x=alpha_c, color='blue', linestyle=':', alpha=0.5)
            ax.annotate(
                f'α_c ≈ {alpha_c:.2f}',
                xy=(alpha_c, 0.5),
                xytext=(alpha_c + 0.3, 0.6),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5)
            )
            break
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "qy_vs_alpha({N1}x{N2},{M}).png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show plot (if running interactively)
    plt.show()
    
    print("Done!")

# %%
