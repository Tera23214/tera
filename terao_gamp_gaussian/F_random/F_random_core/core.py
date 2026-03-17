#!/usr/bin/env python
"""
G-AMP (Generalized Approximate Message Passing) Core Module with Random F.

Implements the G-AMP algorithm for sparse matrix factorization
with F ~ N(0,1) instead of F=1.

Observation model:
    Y_obs = (1/√M) Σ_μ F[c,μ] W_{i_c,μ} X_{μ,j_c} + ΔZ

where F[c,μ] ~ N(0,1) i.i.d. for each observed edge c.
"""

import sys
from pathlib import Path
from typing import Tuple
import torch
import math

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import BiregularGraph
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy, f_input, g_out


# ============================================================================
# G-AMP Functions with Random F
# ============================================================================


def _gamp_step_with_F_impl(
    m_W: torch.Tensor,      # (N1, M) - W messages (mean)
    v_W: torch.Tensor,      # (N1, M) - W messages (second moment)
    m_X: torch.Tensor,      # (M, N2) - X messages (mean)
    v_X: torch.Tensor,      # (M, N2) - X messages (second moment)
    Y: torch.Tensor,        # (C,) - Observed values
    F: torch.Tensor,        # (C, M) - Random F factors per edge
    F_sq: torch.Tensor,     # (C, M) - Pre-computed F**2 for efficiency
    i_idx: torch.Tensor,    # (C,) - Row indices
    j_idx: torch.Tensor,    # (C,) - Column indices
    g_prev: torch.Tensor,   # (C,) - Previous g values
    noise_var: float,       # Noise variance
    damping: float,         # Damping factor
    N1: int,
    N2: int,
    M: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single G-AMP step with random F ~ N(0,1) per edge.
    
    The observation model is:
        Y_c = (1/√M) Σ_μ F[c,μ] W_{i_c,μ} X_{μ,j_c}
    
    F is now (C, M) - different for each observation.
    """
    C = Y.shape[0]
    device = Y.device
    
    scale = 1.0 / math.sqrt(M)
    scale_sq = 1.0 / M
    
    # ========================================================================
    # Step 1: Compute omega and V with F
    # ========================================================================
    
    W_sel = m_W[i_idx.long(), :]      # (C, M)
    X_sel = m_X[:, j_idx.long()].T    # (C, M)
    
    # ω = (1/√M) Σ_μ F[c,μ] m_W × m_X
    omega = scale * (F * W_sel * X_sel).sum(dim=1)  # (C,)
    
    # V = (1/M) Σ_μ F[c,μ]² (v_W × v_X - m_W² × m_X²)
    vW_sel = v_W[i_idx.long(), :]
    vX_sel = v_X[:, j_idx.long()].T
    # F_sq is now passed as pre-computed parameter
    
    V = scale_sq * (F_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2))).sum(dim=1)
    V = torch.clamp(V, min=1e-10)  # Ensure positive
    
    # ========================================================================
    # Step 2: Output function
    # ========================================================================
    
    g, dg = g_out(omega, Y, V, noise_var)
    
    # Apply damping to g (old-value biased)
    g = damping * g_prev + (1 - damping) * g
    
    # ========================================================================
    # Step 3: Update Sigma, T with F
    # Paper formula: Σ = -1 / (Σ_c (1/M) F²_cμ × ∂g/∂ω × m²)
    # Since ∂g/∂ω = -1/(V+σ²), we have: Σ = 1 / (Σ_c (1/M) F²_cμ × (1/(V+σ²)) × m²)
    #
    # Paper formula: T/Σ = m/Σ + sum
    # Therefore: T = m + Σ × sum
    # ========================================================================
    
    # Update W messages
    Sigma_W_denom = torch.zeros_like(m_W)
    
    # Denominator: (1/M) × F²_cμ × (-∂g_c) × m_X² = (1/M) × F² × (1/(V+σ²)) × m_X²
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2) * F_sq  # (C, M)
    Sigma_W_denom.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    # Apply reciprocal: Σ = 1 / denominator
    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    
    # Sum contribution: Σ_c (1/√M) F g m_X
    sum_W = torch.zeros_like(m_W)
    g_expanded = scale * g.unsqueeze(1) * X_sel * F  # (C, M)
    sum_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    # T = m + Σ × sum
    T_W = m_W + Sigma_W * sum_W
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    
    # Update X messages
    Sigma_X_denom = torch.zeros_like(m_X)
    
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2) * F_sq
    Sigma_X_denom.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    # Apply reciprocal for X
    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    
    # Sum contribution for X
    sum_X = torch.zeros_like(m_X)
    g_expanded_X = scale * g.unsqueeze(1) * W_sel * F
    sum_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    # T = m + Σ × sum
    T_X = m_X + Sigma_X * sum_X
    
    # Apply f_input for X
    m_X_new, v_X_new = f_input(Sigma_X, T_X)
    
    # Apply damping (old-value biased)
    m_W_new = damping * m_W + (1 - damping) * m_W_new
    v_W_new = damping * v_W + (1 - damping) * v_W_new
    m_X_new = damping * m_X + (1 - damping) * m_X_new
    v_X_new = damping * v_X + (1 - damping) * v_X_new
    
    return m_W_new, v_W_new, m_X_new, v_X_new, g


# Apply torch.compile only if PyTorch >= 2.0
if hasattr(torch, 'compile'):
    gamp_step_with_F = torch.compile(_gamp_step_with_F_impl, mode="reduce-overhead")
else:
    gamp_step_with_F = _gamp_step_with_F_impl


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
) -> Tuple[float, float, int]:
    """
    Train a single replica using G-AMP with F ~ N(0,1) for spreading.
    
    Observation model:
        Y_c = (1/√M) Σ_μ F[c,μ] W_{i_c,μ} X_{μ,j_c}
    
    where F[c,μ] ~ N(0, 1) i.i.d. for each edge c and component μ.
    """
    # Generate observation graph (BiregularGraph for Dense Limit)
    graph = BiregularGraph()
    i_idx, j_idx, E, C1, C2, alpha2 = graph.generate(N1, N2, M, alpha, device, seed)
    
    if E == 0:
        return 0.0, 0.0, 0
    
    # Generate teacher matrices W ~ N(0,1), X ~ N(0,1)
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate spreading matrix F: (E, M) with F[c,μ] ~ N(0,1) i.i.d.
    torch.manual_seed(seed + 500)
    F = torch.randn(E, M, device=device, dtype=torch.float32)
    F_sq = F ** 2  # Pre-compute for efficiency (used every step)
    
    # Generate observations with F: Y = (1/√M) Σ_μ F[c,μ] W X
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = (1.0 / math.sqrt(M)) * (F * W_sel * X_sel).sum(dim=1)
    
    # Add noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y) * math.sqrt(noise_var)
    Y_noisy = Y + noise
    
    # Initialize messages: m~N(0, 0.1) small to ensure V > 0, v=1.0
    torch.manual_seed(seed + 2000)
    m_W = torch.randn(N1, M, device=device) * 0.1  # Small init
    v_W = torch.ones(N1, M, device=device)
    m_X = torch.randn(M, N2, device=device) * 0.1  # Small init
    v_X = torch.ones(M, N2, device=device)
    g_prev = torch.zeros(E, device=device)
    
    # G-AMP iterations
    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float('inf')
    
    for step in range(max_steps):
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_F(
            m_W, v_W, m_X, v_X,
            Y_noisy, F, F_sq, i_idx, j_idx, g_prev,
            noise_var, damping, N1, N2, M
        )
        
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
    
    # Normalize student matrices
    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)
    
    # Compute Q_Y
    qy = compute_qy(m_W, m_X, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken
