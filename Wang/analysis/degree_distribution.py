"""
Degree Distribution Analysis for Random Graph Generation

Analyzes the node degree distribution of randomly generated observation masks.
Copies the random graph generation method from Main.py and computes statistics
of left-node (row) degrees.

Key metrics: mean and variance of node degrees.
"""

import numpy as np
import torch
from collections import Counter


# =============================================================================
# Configuration
# =============================================================================

N1 = 200          # Number of rows
N2 = 200          # Number of columns
M = 50            # Hidden dimension
ALPHA = 2.0       # Observation density

NUM_SAMPLES = 50  # Number of repeated samplings
SEED = 42         # Random seed


# =============================================================================
# Random Graph Generation
# =============================================================================

def sample_pairs_random_gpu(N1: int, N2: int, C: int, device: torch.device,
                            seed: int = None) -> tuple:
    """Generate random mask using pure GPU operations.

    Args:
        N1: Number of rows
        N2: Number of columns
        C: Number of edges (observed entries)
        device: PyTorch device
        seed: Random seed

    Returns:
        Tuple of (row_indices, col_indices, edge_count)
    """
    if seed is not None:
        torch.manual_seed(seed)

    total = N1 * N2
    if C > total:
        raise RuntimeError(f"Requested edge count C={C} exceeds matrix total size {N1}×{N2}={total}")

    if C == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device), 0)

    # Randomly shuffle all position indices on GPU
    perm = torch.randperm(total, device=device)
    selected = perm[:C]

    # Convert 1D index to (i, j) coordinates
    i_idx = selected // N2
    j_idx = selected % N2

    return i_idx, j_idx, C


# =============================================================================
# Degree Distribution Analysis
# =============================================================================

def analyze_degree_distribution():
    """Analyze left-node degree distribution - focus on mean and variance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Compute edge count
    C = int(ALPHA * M * N1)
    expected_degree = C / N1  # Theoretical expected degree = α × M

    print(f"\n{'='*60}")
    print("Parameter Settings")
    print(f"{'='*60}")
    print(f"N1 = {N1}, N2 = {N2}, M = {M}")
    print(f"Alpha = {ALPHA}")
    print(f"Edge count C = α × M × N1 = {C}")
    print(f"Theoretical expected degree E[d] = C / N1 = α × M = {expected_degree:.2f}")

    # Theoretical variance (Hypergeometric distribution)
    # Each left node has N2 possible positions, total N1*N2 positions, select C
    # Single node degree follows Hypergeometric(N1*N2, N2, C)
    N_total = N1 * N2
    K = N2    # Number of positions per left node
    n = C     # Total sample count
    # E[X] = n * K / N = C * N2 / (N1*N2) = C/N1 ✓
    # Var[X] = n * K/N * (1 - K/N) * (N-n)/(N-1)
    theoretical_var = n * (K/N_total) * (1 - K/N_total) * (N_total - n) / (N_total - 1)
    theoretical_std = np.sqrt(theoretical_var)

    print(f"\nTheoretical Prediction (Hypergeometric Distribution):")
    print(f"  E[d] = {expected_degree:.4f}")
    print(f"  Var[d] = {theoretical_var:.4f}")
    print(f"  Std[d] = {theoretical_std:.4f}")

    # Multiple sampling statistics
    print(f"\n{'='*60}")
    print(f"Empirical Statistics ({NUM_SAMPLES} samples)")
    print(f"{'='*60}")

    all_means = []
    all_vars = []

    for sample_idx in range(NUM_SAMPLES):
        seed = SEED + sample_idx
        i_idx, _, _ = sample_pairs_random_gpu(N1, N2, C, device, seed=seed)

        # Count degree for each left node
        i_idx_cpu = i_idx.cpu().numpy()
        degree_counts = Counter(i_idx_cpu)
        degrees = np.array([degree_counts.get(i, 0) for i in range(N1)])

        all_means.append(np.mean(degrees))
        all_vars.append(np.var(degrees))

    print(f"\nWithin-sample statistics (degrees of {N1} nodes per sample):")
    print(f"  Mean degree (average): {np.mean(all_means):.4f} (theoretical: {expected_degree:.4f})")
    print(f"  Mean degree (std): {np.std(all_means):.4f}")
    print(f"  Variance (average): {np.mean(all_vars):.4f} (theoretical: {theoretical_var:.4f})")
    print(f"  Variance (std): {np.std(all_vars):.4f}")

    # Detailed single sample results
    print(f"\n{'='*60}")
    print("Single Sample Details (seed=42)")
    print(f"{'='*60}")

    i_idx, _, _ = sample_pairs_random_gpu(N1, N2, C, device, seed=SEED)
    i_idx_cpu = i_idx.cpu().numpy()
    degree_counts = Counter(i_idx_cpu)
    degrees = np.array([degree_counts.get(i, 0) for i in range(N1)])

    print(f"  Mean: {np.mean(degrees):.4f}")
    print(f"  Variance: {np.var(degrees):.4f}")
    print(f"  Std: {np.std(degrees):.4f}")
    print(f"  Min degree: {np.min(degrees)}")
    print(f"  Max degree: {np.max(degrees)}")
    print(f"  Degree range: [{np.min(degrees)}, {np.max(degrees)}]")

    # Conclusions
    print(f"\n{'='*60}")
    print("Conclusions")
    print(f"{'='*60}")
    print(f"1. Mean: E[d] = α × M = {expected_degree:.1f} (exact)")
    print(f"2. Variance: Var[d] ≈ {theoretical_var:.2f} (Hypergeometric distribution)")
    print(f"3. Std: Std[d] ≈ {theoretical_std:.2f}")
    print(f"4. Degree fluctuation range ≈ [{expected_degree - 3*theoretical_std:.0f}, "
          f"{expected_degree + 3*theoretical_std:.0f}] (3σ)")
    print(f"\nComparison with BiRegular graph: each node has fixed degree α × M = {expected_degree:.0f}, "
          f"variance = 0")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    analyze_degree_distribution()
