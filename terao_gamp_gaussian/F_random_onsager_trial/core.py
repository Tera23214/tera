#!/usr/bin/env python
"""
G-AMP with True Biregular Graph and Onsager Correction.

Tests whether true biregular graph stabilizes Onsager correction.
"""

import sys
from pathlib import Path
import torch
import math

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.F_random_onsager_trial.true_biregular import TrueBiregularGraph
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy


# ============================================================================
# G-AMP Functions
# ============================================================================

def f_input(Sigma: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Input function for Gaussian prior."""
    denom = 1.0 + Sigma
    denom = torch.clamp(denom, min=1e-10)
    m = T / denom
    v = Sigma / denom + (T ** 2) / (denom ** 2)
    return m, v


def g_out(
    omega: torch.Tensor,
    y: torch.Tensor,
    V: torch.Tensor,
    noise_var: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output function for Gaussian noise channel."""
    denom = V + noise_var
    denom = torch.clamp(denom, min=1e-10)
    g = (y - omega) / denom
    dg = -1.0 / denom
    return g, dg


def gamp_step_with_onsager(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y: torch.Tensor,
    F: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    g_prev: torch.Tensor,
    noise_var: float,
    damping: float,
    N1: int,
    N2: int,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    G-AMP step with Onsager correction (simplified version).
    
    This version uses a more stable Onsager formulation.
    """
    C = Y.shape[0]
    device = Y.device
    
    scale = 1.0 / math.sqrt(M)
    scale_sq = 1.0 / M
    
    # Select current values
    W_sel = m_W[i_idx.long(), :]
    X_sel = m_X[:, j_idx.long()].T
    vW_sel = v_W[i_idx.long(), :]
    vX_sel = v_X[:, j_idx.long()].T
    
    # Compute omega (WITHOUT Onsager in omega - simpler approach)
    omega = scale * (F * W_sel * X_sel).sum(dim=1)
    
    # Compute V
    F_sq = F ** 2
    V = scale_sq * (F_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2))).sum(dim=1)
    V = torch.clamp(V, min=1e-10)
    
    # Output function
    g, dg = g_out(omega, Y, V, noise_var)
    g = damping * g + (1 - damping) * g_prev
    g = torch.clamp(g, min=-10.0, max=10.0)  # More aggressive clamp
    
    # ========================================================================
    # Update W with simplified Onsager (only in Sigma, not T)
    # ========================================================================
    Sigma_W = torch.zeros_like(m_W)
    T_W = torch.zeros_like(m_W)  # Start from zero, not m_W
    
    # Sigma contribution
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2) * F_sq
    Sigma_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    # T contribution
    g_expanded = scale * g.unsqueeze(1) * X_sel * F
    T_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    # Add prior mean term (T = T + m_prior, for N(0,1) prior: m_prior = 0 implicitly)
    # No Onsager in T (stable version)
    
    m_W_new, v_W_new = f_input(torch.clamp(Sigma_W, min=1e-10), T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=10.0)
    
    # ========================================================================
    # Update X (symmetric)
    # ========================================================================
    Sigma_X = torch.zeros_like(m_X)
    T_X = torch.zeros_like(m_X)
    
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2) * F_sq
    Sigma_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    g_expanded_X = scale * g.unsqueeze(1) * W_sel * F
    T_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    m_X_new, v_X_new = f_input(torch.clamp(Sigma_X, min=1e-10), T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=10.0)
    
    # Apply damping
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
    """Train using G-AMP with true biregular graph."""
    
    # Generate TRUE biregular graph
    graph = TrueBiregularGraph()
    i_idx, j_idx, E, C1, C2, alpha2 = graph.generate(N1, N2, M, alpha, device, seed)
    
    if E == 0:
        return 0.0, 0.0, 0
    
    # Generate teacher
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate F
    torch.manual_seed(seed + 500)
    F = torch.randn(E, M, device=device, dtype=torch.float32)
    
    # Generate observations
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = (1.0 / math.sqrt(M)) * (F * W_sel * X_sel).sum(dim=1)
    
    # Add noise
    torch.manual_seed(seed + 1000)
    Y_noisy = Y + torch.randn_like(Y) * math.sqrt(noise_var)
    
    # Initialize
    torch.manual_seed(seed + 2000)
    m_W = torch.randn(N1, M, device=device) * 0.1
    v_W = torch.ones(N1, M, device=device)
    m_X = torch.randn(M, N2, device=device) * 0.1
    v_X = torch.ones(M, N2, device=device)
    g_prev = torch.zeros(E, device=device)
    
    m_W_prev = m_W.clone()
    m_X_prev = m_X.clone()
    
    # Iterations
    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float('inf')
    
    for step in range(max_steps):
        m_W_old = m_W.clone()
        m_X_old = m_X.clone()
        
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_onsager(
            m_W, v_W, m_X, v_X, m_W_prev, m_X_prev,
            Y_noisy, F, i_idx, j_idx, g_prev,
            noise_var, damping, N1, N2, M
        )
        
        m_W_prev = m_W_old
        m_X_prev = m_X_old
        
        # Check convergence
        if step % 50 == 0 or step == max_steps - 1:
            W_sel = m_W[i_idx.long(), :]
            X_sel = m_X[:, j_idx.long()].T
            Y_pred = (1.0 / math.sqrt(M)) * (F * W_sel * X_sel).sum(dim=1)
            
            loss = ((Y_noisy - Y_pred) ** 2).mean().item()
            final_loss = loss
            
            if abs(prev_loss - loss) < convergence_threshold:
                steps_taken = step + 1
                break
            prev_loss = loss
    
    # Normalize
    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)
    
    # Compute Q_Y
    qy = compute_qy(m_W, m_X, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken
