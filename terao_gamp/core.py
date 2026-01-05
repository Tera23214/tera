#!/usr/bin/env python
"""
G-AMP (Generalized Approximate Message Passing) Core Module.

Implements the G-AMP algorithm for sparse matrix factorization
based on arXiv:2510.17886 Algorithm 2.

Key functions:
- f_input: Input denoiser (Gaussian prior)
- g_out: Output function (Gaussian noise)
- gamp_step: Single G-AMP iteration
- train_single_replica: Train one replica for given alpha
"""

import sys
from pathlib import Path
import torch
import math

# Add parent directory to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from smf.modules.graphs.random import RandomGraph
from .utils import normalize_to_unit_variance, compute_qy


# ============================================================================
# G-AMP Functions (Algorithm 2)
# ============================================================================

def f_input(Sigma: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input function for Gaussian prior (Alg2.4).
    
    For standard Gaussian prior N(0, 1):
        m = T / (1 + Sigma)
        v = Sigma / (1 + Sigma)
    
    Args:
        Sigma: Inverse variance parameter
        T: Mean parameter (scaled)
    
    Returns:
        m: Posterior mean
        v: Posterior variance
    """
    # Avoid division by zero
    denom = 1.0 + Sigma
    denom = torch.clamp(denom, min=1e-10)
    
    m = T / denom
    v = Sigma / denom
    
    return m, v


def g_out(
    omega: torch.Tensor,
    y: torch.Tensor,
    V: torch.Tensor,
    noise_var: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Output function for Gaussian noise channel (Alg2.2).
    
    For additive Gaussian noise with variance sigma^2:
        g = (y - omega) / (V + sigma^2)
        dg/d_omega = -1 / (V + sigma^2)
    
    Args:
        omega: Current prediction
        y: Observed value
        V: Variance estimate
        noise_var: Noise variance (sigma^2)
    
    Returns:
        g: Output function value
        dg: Derivative of g w.r.t. omega
    """
    denom = V + noise_var
    denom = torch.clamp(denom, min=1e-10)
    
    g = (y - omega) / denom
    dg = -1.0 / denom
    
    return g, dg


def gamp_step(
    m_W: torch.Tensor,      # (N1, M) - W messages (mean)
    v_W: torch.Tensor,      # (N1, M) - W messages (variance)
    m_X: torch.Tensor,      # (M, N2) - X messages (mean)
    v_X: torch.Tensor,      # (M, N2) - X messages (variance)
    Y: torch.Tensor,        # (C,) - Observed values
    i_idx: torch.Tensor,    # (C,) - Row indices
    j_idx: torch.Tensor,    # (C,) - Column indices
    g_prev: torch.Tensor,   # (C,) - Previous g values
    lam: float,             # Lambda = sqrt(alpha * M)
    noise_var: float,       # Noise variance
    damping: float,         # Damping factor
    N1: int,
    N2: int,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single G-AMP step (Algorithm 2).
    
    Updates messages for W and X based on observed values Y.
    
    Returns:
        m_W_new, v_W_new, m_X_new, v_X_new, g_new
    """
    C = Y.shape[0]
    device = Y.device
    
    # Scale factor
    scale = lam / math.sqrt(M)
    
    # ========================================================================
    # Step 3: Update omega and V (Alg2.1)
    # ========================================================================
    
    # Compute predictions at observed locations
    # omega_c = sum_mu (lambda/sqrt(M)) * prod_j m_W[i,mu] * prod_k m_X[mu,j]
    # Simplified: omega_c = W[i_c, :] @ X[:, j_c]
    
    W_sel = m_W[i_idx.long(), :]      # (C, M)
    X_sel = m_X[:, j_idx.long()].T    # (C, M)
    
    # Predicted values
    omega = (W_sel * X_sel).sum(dim=1)  # (C,)
    
    # Variance at observed locations
    vW_sel = v_W[i_idx.long(), :]     # (C, M)
    vX_sel = v_X[:, j_idx.long()].T   # (C, M)
    
    # V = sum_mu (var_W * m_X^2 + m_W^2 * var_X + var_W * var_X)
    V = (vW_sel * (X_sel ** 2) + (W_sel ** 2) * vX_sel + vW_sel * vX_sel).sum(dim=1)  # (C,)
    
    # ========================================================================
    # Step 4: Output function (Alg2.2)
    # ========================================================================
    
    g, dg = g_out(omega, Y, V, noise_var)  # (C,), (C,)
    
    # Apply damping to g
    g = damping * g + (1 - damping) * g_prev
    
    # ========================================================================
    # Step 5-6: Update Sigma, T and then m, v (Alg2.3, Alg2.4)
    # ========================================================================
    
    # Update W messages
    # Sigma_W[i,mu] = sum_{c: i_c=i} (-dg_c) * m_X[mu, j_c]^2
    # T_W[i,mu] = m_W[i,mu] + sum_{c: i_c=i} g_c * m_X[mu, j_c]
    
    Sigma_W = torch.zeros_like(m_W)
    T_W = m_W.clone()
    
    # Scatter add for efficiency
    dg_expanded = (-dg).unsqueeze(1) * (X_sel ** 2)  # (C, M)
    Sigma_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    g_expanded = g.unsqueeze(1) * X_sel  # (C, M)
    T_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(torch.clamp(Sigma_W, min=1e-10), T_W)
    
    # Update X messages
    # Sigma_X[mu,j] = sum_{c: j_c=j} (-dg_c) * m_W[i_c, mu]^2
    # T_X[mu,j] = m_X[mu,j] + sum_{c: j_c=j} g_c * m_W[i_c, mu]
    
    Sigma_X = torch.zeros_like(m_X)
    T_X = m_X.clone()
    
    dg_expanded_X = (-dg).unsqueeze(1) * (W_sel ** 2)  # (C, M)
    Sigma_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    g_expanded_X = g.unsqueeze(1) * W_sel  # (C, M)
    T_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    # Apply f_input for X
    m_X_new, v_X_new = f_input(torch.clamp(Sigma_X, min=1e-10), T_X)
    
    # Apply damping to messages
    m_W_new = damping * m_W_new + (1 - damping) * m_W
    v_W_new = damping * v_W_new + (1 - damping) * v_W
    m_X_new = damping * m_X_new + (1 - damping) * m_X
    v_X_new = damping * v_X_new + (1 - damping) * v_X
    
    return m_W_new, v_W_new, m_X_new, v_X_new, g


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    max_steps: int = 500,
    damping: float = 0.5,
    noise_var: float = 1e-10,
    convergence_threshold: float = 1e-6,
) -> tuple[float, float, int]:
    """
    Train a single replica using G-AMP.
    
    Args:
        alpha: Observation density
        device: torch device
        seed: Random seed
        N1, N2, M: Matrix dimensions
        max_steps: Maximum iterations
        damping: Message damping factor
        noise_var: Noise variance
        convergence_threshold: Convergence threshold
    
    Returns:
        qy: Q_Y overlap metric
        final_loss: Final MSE loss
        steps_taken: Number of iterations
    """
    # Lambda parameter
    lam = math.sqrt(alpha * M)
    
    # Generate teacher matrices
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate observation graph
    graph = RandomGraph()
    i_idx, j_idx, C = graph.generate(N1, N2, M, alpha, device, seed)
    
    if C == 0:
        return 0.0, 0.0, 0
    
    # Generate observed values
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = (W_sel * X_sel).sum(dim=1)  # (C,)
    
    # Initialize messages (small random values)
    torch.manual_seed(seed + 1000)
    m_W = torch.randn(N1, M, device=device) * 0.01
    v_W = torch.ones(N1, M, device=device)
    m_X = torch.randn(M, N2, device=device) * 0.01
    v_X = torch.ones(M, N2, device=device)
    g_prev = torch.zeros(C, device=device)
    
    # G-AMP iterations
    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float('inf')
    
    for step in range(max_steps):
        m_W, v_W, m_X, v_X, g_prev = gamp_step(
            m_W, v_W, m_X, v_X, Y, i_idx, j_idx, g_prev,
            lam, noise_var, damping, N1, N2, M
        )
        
        # Check convergence every 50 steps
        if step % 50 == 0 or step == max_steps - 1:
            # Compute predictions
            W_sel = m_W[i_idx.long(), :]
            X_sel = m_X[:, j_idx.long()].T
            Y_pred = (W_sel * X_sel).sum(dim=1)
            
            loss = ((Y - Y_pred) ** 2).mean().item()
            final_loss = loss
            
            # Check for convergence
            if abs(prev_loss - loss) < convergence_threshold:
                steps_taken = step + 1
                break
            prev_loss = loss
    
    # Normalize student matrices
    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)
    
    # Compute Q_Y
    qy = compute_qy(m_W, m_X, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken
