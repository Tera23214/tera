#!/usr/bin/env python
"""
Benchmark: F_random (G-AMP) vs gd.py (AGD) Simulation Time Comparison.

Compares execution time of:
- G-AMP (Generalized Approximate Message Passing) with Random F
- AGD (Alternating Gradient Descent)

Both algorithms solve sparse matrix factorization with the same matrix dimensions
and observation density.
"""

import sys
import time
import math
from pathlib import Path
import torch
import numpy as np

# Add paths
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

# Import both implementations
from terao_gamp_gaussian.F_random.F_random_core.core import train_single_replica as gamp_train
from terao_gd.gd import train_single_replica as agd_train, N1, N2, M

# ============================================================================
# Benchmark Configuration
# ============================================================================

# Match parameters between both algorithms
BENCHMARK_N1 = 1000
BENCHMARK_N2 = 1000
BENCHMARK_M = 10
BENCHMARK_ALPHA = 2.0  # Mid-range alpha for fair comparison
BENCHMARK_REPLICAS = 5  # Number of replicas for averaging
SEED = 42

# Individual algorithm configurations
AGD_MAX_STEPS = 500  # Reduced from 3000 for fair comparison
GAMP_MAX_STEPS = 500

# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_agd(device, num_replicas=BENCHMARK_REPLICAS):
    """Benchmark AGD algorithm."""
    times = []
    qy_values = []
    steps_list = []
    
    for i in range(num_replicas):
        seed = SEED + i * 1000
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        t0 = time.time()
        qy, loss, steps = agd_train(
            alpha=BENCHMARK_ALPHA,
            device=device,
            seed=seed,
        )
        dt = time.time() - t0
        
        times.append(dt)
        qy_values.append(qy)
        steps_list.append(steps)
        print(f"  AGD replica {i+1}/{num_replicas}: Q_Y={qy:.4f}, steps={steps}, time={dt:.2f}s")
    
    return {
        'name': 'AGD',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_time': sum(times),
        'mean_qy': np.mean(qy_values),
        'std_qy': np.std(qy_values),
        'mean_steps': np.mean(steps_list),
        'times': times,
        'qy_values': qy_values,
        'steps': steps_list,
    }


def benchmark_gamp(device, num_replicas=BENCHMARK_REPLICAS):
    """Benchmark G-AMP algorithm."""
    times = []
    qy_values = []
    steps_list = []
    
    for i in range(num_replicas):
        seed = SEED + i * 1000
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        t0 = time.time()
        qy, loss, steps = gamp_train(
            alpha=BENCHMARK_ALPHA,
            device=device,
            seed=seed,
            N1=BENCHMARK_N1,
            N2=BENCHMARK_N2,
            M=BENCHMARK_M,
            max_steps=GAMP_MAX_STEPS,
            damping=0.5,
            noise_var=1e-10,
            convergence_threshold=1e-6,
        )
        dt = time.time() - t0
        
        times.append(dt)
        qy_values.append(qy)
        steps_list.append(steps)
        print(f"  G-AMP replica {i+1}/{num_replicas}: Q_Y={qy:.4f}, steps={steps}, time={dt:.2f}s")
    
    return {
        'name': 'G-AMP (F_random)',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_time': sum(times),
        'mean_qy': np.mean(qy_values),
        'std_qy': np.std(qy_values),
        'mean_steps': np.mean(steps_list),
        'times': times,
        'qy_values': qy_values,
        'steps': steps_list,
    }


def print_comparison(agd_result, gamp_result):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("SIMULATION TIME COMPARISON: F_random (G-AMP) vs gd.py (AGD)")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Matrix size: {BENCHMARK_N1} × {BENCHMARK_N2}")
    print(f"  Rank (M): {BENCHMARK_M}")
    print(f"  Alpha: {BENCHMARK_ALPHA}")
    print(f"  Replicas: {BENCHMARK_REPLICAS}")
    print(f"  Max steps: AGD={agd_result['mean_steps']:.0f}, G-AMP={gamp_result['mean_steps']:.0f}")
    
    print("\n" + "-" * 70)
    print(f"{'Metric':<25} | {'AGD':^18} | {'G-AMP (F_random)':^18}")
    print("-" * 70)
    
    print(f"{'Mean time per replica':<25} | {agd_result['mean_time']:>8.3f}s ± {agd_result['std_time']:.3f}s | "
          f"{gamp_result['mean_time']:>8.3f}s ± {gamp_result['std_time']:.3f}s")
    
    print(f"{'Total time (all replicas)':<25} | {agd_result['total_time']:>14.2f}s | "
          f"{gamp_result['total_time']:>14.2f}s")
    
    print(f"{'Mean Q_Y':<25} | {agd_result['mean_qy']:>8.4f} ± {agd_result['std_qy']:.4f} | "
          f"{gamp_result['mean_qy']:>8.4f} ± {gamp_result['std_qy']:.4f}")
    
    print(f"{'Mean steps to converge':<25} | {agd_result['mean_steps']:>14.0f} | "
          f"{gamp_result['mean_steps']:>14.0f}")
    
    # Speed comparison
    speedup = agd_result['mean_time'] / gamp_result['mean_time'] if gamp_result['mean_time'] > 0 else float('inf')
    faster = "G-AMP" if speedup > 1 else "AGD"
    ratio = speedup if speedup > 1 else 1/speedup
    
    print("-" * 70)
    print(f"\nSpeed comparison: {faster} is {ratio:.2f}x faster")
    
    # Time per iteration
    agd_time_per_iter = agd_result['mean_time'] / agd_result['mean_steps'] if agd_result['mean_steps'] > 0 else 0
    gamp_time_per_iter = gamp_result['mean_time'] / gamp_result['mean_steps'] if gamp_result['mean_steps'] > 0 else 0
    
    print(f"\nTime per iteration:")
    print(f"  AGD:   {agd_time_per_iter*1000:.3f} ms/step")
    print(f"  G-AMP: {gamp_time_per_iter*1000:.3f} ms/step")
    
    iter_ratio = agd_time_per_iter / gamp_time_per_iter if gamp_time_per_iter > 0 else float('inf')
    print(f"  Ratio: AGD step takes {iter_ratio:.2f}x time of G-AMP step")
    
    print("=" * 70)
    
    return {
        'speedup': speedup,
        'faster_algorithm': faster,
        'agd_time_per_iter': agd_time_per_iter,
        'gamp_time_per_iter': gamp_time_per_iter,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Benchmark: F_random (G-AMP) vs gd.py (AGD)")
    print("Simulation Time Comparison")
    print("=" * 70)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using: CUDA ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        print("Using: CPU")
    
    print()
    
    # Warm-up runs (to avoid cold start effects)
    print("Warm-up runs...")
    _ = agd_train(alpha=BENCHMARK_ALPHA, device=device, seed=0)
    _ = gamp_train(alpha=BENCHMARK_ALPHA, device=device, seed=0, 
                   N1=BENCHMARK_N1, N2=BENCHMARK_N2, M=BENCHMARK_M,
                   max_steps=10, damping=0.5, noise_var=1e-10)
    print("Warm-up complete.\n")
    
    # Benchmark AGD
    print("Benchmarking AGD (gd.py)...")
    agd_result = benchmark_agd(device)
    
    print()
    
    # Benchmark G-AMP
    print("Benchmarking G-AMP (F_random)...")
    gamp_result = benchmark_gamp(device)
    
    # Print comparison
    comparison = print_comparison(agd_result, gamp_result)
