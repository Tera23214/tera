#!/usr/bin/env python3
"""
Baseline Runner with Smart Alphas

This script runs ORIGINAL Main_multi_alpha.py baseline (SGD) using smart alphas from Step3.
This ensures a fair 对拍 comparison where both methods train on the SAME alpha points.

Usage:
    python run_baseline_with_smart_alphas.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Import from the ORIGINAL Main_multi_alpha.py (SGD version) at repository root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Main_multi_alpha import (
    create_teacher_dense, train_all_alphas_parallel,
    set_seed, DEVICE, N1, N2, M, SAMPLES_PER_ALPHA, SEED, LEARNING_RATE
)


def main():
    """Main workflow: Load smart alphas and run original SGD baseline"""
    print("="*80)
    print("ORIGINAL SGD BASELINE WITH SMART ALPHAS")
    print("="*80)
    print("\nThis script runs ORIGINAL Main_multi_alpha.py (SGD, fixed LR) with")
    print("the smart alphas from Step3 to enable fair 对拍 validation.\n")

    # Paths
    base_dir = Path(__file__).parent
    smart_alphas_path = base_dir / "Result" / "200_200_50" / "smart_alphas.npy"
    output_path = base_dir / "Result" / "200_200_50" / "results_baseline_smart_alphas_epoch20000.json"

    # Load smart alphas
    if not smart_alphas_path.exists():
        print(f"❌ Error: Smart alphas not found at {smart_alphas_path}")
        print("\nYou need to run Step3 first to generate smart_alphas.npy")
        return

    smart_alphas = np.load(smart_alphas_path)
    print(f"[Smart Alphas Loaded] {len(smart_alphas)} points")
    print(f"  Range: [{smart_alphas.min():.2f}, {smart_alphas.max():.2f}]")
    print(f"  First 5: {smart_alphas[:5]}")
    print(f"  Last 5: {smart_alphas[-5:]}")

    # Training configuration
    epochs = 20000
    learning_rate = LEARNING_RATE  # Original Main_multi_alpha.py uses fixed LR=1e-2 (SGD)

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Optimizer: SGD (original Main_multi_alpha.py)")
    print(f"Matrix dimensions: {N1}×{N2}×{M}")
    print(f"Number of alphas: {len(smart_alphas)}")
    print(f"Training epochs: {epochs:,}")
    print(f"Samples per alpha: {SAMPLES_PER_ALPHA}")
    print(f"Learning rate: {learning_rate} (FIXED - no scheduler)")
    print(f"Device: {DEVICE}")

    # Initialize teacher model
    set_seed(SEED)
    Wt, Xt = create_teacher_dense(N1, N2, M, DEVICE, seed=SEED)
    print(f"\n[Teacher model initialized with seed={SEED}]")

    # Run training
    print(f"\n{'='*80}")
    print("STARTING ORIGINAL SGD BASELINE TRAINING")
    print("="*80)

    start_time = time.time()

    results = train_all_alphas_parallel(
        Wt, Xt,
        alpha_values=smart_alphas.tolist(),
        steps=epochs,
        S=SAMPLES_PER_ALPHA,
        seed_for_init=SEED + 10_000,
        lr=learning_rate,
        loss_squared_sum=True
    )

    elapsed_time = time.time() - start_time

    print(f"\n✓ Training completed in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")

    # Convert float keys to string keys for JSON compatibility
    results_str_keys = {str(k): v for k, v in results.items()}

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_str_keys, f, indent=2)

    print(f"\n[Results saved] {output_path}")
    print(f"  Total alphas: {len(results_str_keys)}")

    # Print sample results
    print(f"\n{'='*80}")
    print("SAMPLE RESULTS")
    print("="*80)

    sample_alphas = [0.0, 1.0, 1.8, 2.0, 3.0]
    print(f"\n{'Alpha':<10} {'Q_Y':<10} {'Q_W_prime':<10} {'Q_X_prime':<10}")
    print("-" * 45)

    for alpha in sample_alphas:
        alpha_str = str(alpha)
        if alpha_str in results_str_keys:
            data = results_str_keys[alpha_str]
            print(f"{alpha:<10.1f} {data['Q_Y_mean']:<10.4f} {data['Q_W_prime_mean']:<10.4f} {data['Q_X_prime_mean']:<10.4f}")

    print(f"\n{'='*80}")
    print("ORIGINAL SGD BASELINE TRAINING COMPLETED")
    print("="*80)
    print(f"\nYou can now run validate_step3.py to compare Step3 (Adam+LR) with this SGD baseline.")


if __name__ == "__main__":
    main()
