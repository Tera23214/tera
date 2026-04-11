#!/usr/bin/env python

#%%

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

from terao_gamp_gaussian.graph import RandomGraph

# ============================================================================
# Configuration
# ============================================================================

N1 = 3000   # Number of rows
N2 = 3000   # Number of columns  
M = 30      # Rank (hidden dimension)

ALPHA_START = 0.1
ALPHA_STOP = 5.0
ALPHA_STEP = 0.5

MAX_STEPS = 300
DAMPING = 0.5
NOISE_VAR = 1e-10
SEED = 42

# ============================================================================
# BiG-AMP Spreading Algorithm (Memory Efficient)
# ============================================================================

def generate_F(C: int, M: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate F ~ N(0, 1) spreading coefficients."""
    if C == 0:
        return torch.empty(0, M, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(C, M, device=device, dtype=torch.float32, generator=gen)


def bigamp_step(
    W_hat: torch.Tensor,   # (N1, M)
    X_hat: torch.Tensor,   # (M, N2)
    W_var: torch.Tensor,   # (N1, M)
    X_var: torch.Tensor,   # (M, N2)
    Y: torch.Tensor,       # (C,)
    F: torch.Tensor,       # (C, M)
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
    damping: float,
    noise_var: float,
):
    """Single BiG-AMP step with spreading."""
    N1, M = W_hat.shape
    N2 = X_hat.shape[1]
    C = F.shape[0]
    alpha_scale = 1.0 / math.sqrt(M)
    alpha_scale_sq = 1.0 / M

    # Forward pass: Z_hat[c] = (1/√M) Σ_μ F[c,μ] W[i,μ] X[μ,j]
    W_sel = W_hat[i_idx.long(), :]  # (C, M)
    X_sel = X_hat[:, j_idx.long()].T  # (C, M)
    Z_hat = alpha_scale * (F * W_sel * X_sel).sum(dim=1)  # (C,)

    # Variance
    W_var_sel = W_var[i_idx.long(), :]  # (C, M)
    X_var_sel = X_var[:, j_idx.long()].T  # (C, M)
    F_sq = F.pow(2)
    V = alpha_scale_sq * (F_sq * (W_var_sel * X_sel.pow(2) + W_sel.pow(2) * X_var_sel)).sum(dim=1)
    V = V + 1e-10

    # Residuals
    denom = torch.clamp(V + noise_var, min=1e-6)
    s = (Y - Z_hat) / denom  # (C,)
    s = torch.clamp(s, min=-1e6, max=1e6)

    # Update W: scatter contributions
    s_exp = s.unsqueeze(1)  # (C, 1)
    inv_V = (1.0 / denom).unsqueeze(1)  # (C, 1)

    r_W_contrib = alpha_scale * F * X_sel * s_exp  # (C, M)
    r_W = torch.zeros(N1, M, device=W_hat.device, dtype=W_hat.dtype)
    r_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(C, M), r_W_contrib)

    tau_W_contrib = alpha_scale_sq * F_sq * X_sel.pow(2) * inv_V
    tau_W = torch.zeros(N1, M, device=W_hat.device, dtype=W_hat.dtype)
    tau_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(C, M), tau_W_contrib)
    tau_W = tau_W.clamp(min=1e-10)

    W_var_new = 1.0 / (1.0 + tau_W)
    r_W = torch.clamp(r_W, min=-1e4, max=1e4)
    W_hat_new = W_hat + W_var_new * r_W

    # Update X
    r_X_contrib = alpha_scale * F * W_sel * s_exp  # (C, M)
    r_X = torch.zeros(M, N2, device=X_hat.device, dtype=X_hat.dtype)
    r_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, C), r_X_contrib.T)

    tau_X_contrib = alpha_scale_sq * F_sq * W_sel.pow(2) * inv_V
    tau_X = torch.zeros(M, N2, device=X_hat.device, dtype=X_hat.dtype)
    tau_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, C), tau_X_contrib.T)
    tau_X = tau_X.clamp(min=1e-10)

    X_var_new = 1.0 / (1.0 + tau_X)
    r_X = torch.clamp(r_X, min=-1e4, max=1e4)
    X_hat_new = X_hat + X_var_new * r_X

    # Damping
    W_hat_out = damping * W_hat_new + (1 - damping) * W_hat
    X_hat_out = damping * X_hat_new + (1 - damping) * X_hat
    W_var_out = torch.clamp(damping * W_var_new + (1 - damping) * W_var, min=1e-4, max=1.0)
    X_var_out = torch.clamp(damping * X_var_new + (1 - damping) * X_var, min=1e-4, max=1.0)

    # NaN protection
    W_hat_out = torch.nan_to_num(W_hat_out, nan=0.0)
    X_hat_out = torch.nan_to_num(X_hat_out, nan=0.0)

    return W_hat_out, X_hat_out, W_var_out, X_var_out


def compute_qy(W_student, X_student, W_teacher, X_teacher):
    """
    Compute Q_Y overlap using full dense matrices.
    
    Q_Y = <Y_teacher, Y_student> / (||Y_teacher|| * ||Y_student||)
    """
    # Full matrix products
    Y_teacher = W_teacher @ X_teacher  # (N1, N2)
    Y_student = W_student @ X_student  # (N1, N2)
    
    # Normalized overlap (Frobenius inner product)
    num = (Y_teacher * Y_student).sum()
    denom = torch.sqrt((Y_teacher ** 2).sum() * (Y_student ** 2).sum())
    
    return (num / (denom + 1e-10)).item()


def train_single_alpha(
    alpha: float,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    device: torch.device,
    seed: int,
):
    """Train BiG-AMP for a single alpha value."""
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    # Generate graph
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0
    
    # Generate F and Y
    F = generate_F(C, M, seed + 1000, device)
    alpha_scale = 1.0 / math.sqrt(M)
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = alpha_scale * (F * W_sel * X_sel).sum(dim=1)
    
    # Initialize student
    torch.manual_seed(seed + 2000)
    W_hat = torch.randn(N1, M, device=device, dtype=torch.float32) * 0.1
    X_hat = torch.randn(M, N2, device=device, dtype=torch.float32) * 0.1
    W_var = torch.ones(N1, M, device=device, dtype=torch.float32)
    X_var = torch.ones(M, N2, device=device, dtype=torch.float32)
    
    # Training loop
    for step in range(MAX_STEPS):
        W_hat, X_hat, W_var, X_var = bigamp_step(
            W_hat, X_hat, W_var, X_var,
            Y, F, i_idx, j_idx,
            DAMPING, NOISE_VAR
        )
    
    # Compute Q_Y using full dense matrices
    qy = compute_qy(W_hat, X_hat, W_teacher, X_teacher)
    
    return qy


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BiG-AMP Spreading - Memory Efficient (RandomGraph)")
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
    print(f"Steps: {MAX_STEPS}")
    print()
    
    # Generate teacher
    torch.manual_seed(SEED)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32) / math.sqrt(M)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32) / math.sqrt(M)
    
    # Run for each alpha
    alphas = np.arange(ALPHA_START, ALPHA_STOP + ALPHA_STEP/2, ALPHA_STEP)
    results = {}
    
    start_time = time.time()
    
    for alpha in alphas:
        t0 = time.time()
        qy = train_single_alpha(alpha, W_teacher, X_teacher, device, SEED)
        dt = time.time() - t0
        results[alpha] = qy
        print(f"α={alpha:.2f}: Q_Y={qy:.4f}  ({dt:.1f}s)")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)
    
    # Plot
    print("\nGenerating plot...")
    
    alphas_list = sorted(results.keys())
    qy_values = [results[a] for a in alphas_list]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(alphas_list, qy_values, 'o-', color='#E53935', markersize=6, linewidth=2)
    ax.set_xlabel(r'$\alpha$ (observation density)', fontsize=14)
    ax.set_ylabel(r'$Q_Y$', fontsize=14)
    ax.set_title(f'Phase Transition Curve\n({N1}×{N2}, M={M}, {MAX_STEPS} steps)', fontsize=16)
    ax.set_xlim(ALPHA_START - 0.1, ALPHA_STOP + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "qy_vs_alpha_efficient.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    
    print("Done!")

# %%
