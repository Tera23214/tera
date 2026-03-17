#!/usr/bin/env python
"""
G-AMP (Generalized Approximate Message Passing) Core Module with Random F and Onsager Term.

Implements the G-AMP algorithm for sparse matrix factorization
with F ~ N(0,1) and proper Onsager correction.

Observation model:
    Y_obs = (1/√M) Σ_μ F_μ W_{i,μ} X_{μ,j} + ΔZ

where F_μ ~ N(0,1) i.i.d.

The Onsager term corrects for the self-correlation in the iteration,
which is critical for proper convergence in message passing algorithms.
"""

import sys
from pathlib import Path
import torch
import math

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import BiregularGraph
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy, f_input, g_out


# ============================================================================
# G-AMP Functions with Random F and Onsager Correction
# ============================================================================


def gamp_step_with_F_onsager(
    m_W: torch.Tensor,      # (N1, M) - W messages (mean) at time t
    v_W: torch.Tensor,      # (N1, M) - W messages (second moment) at time t
    m_X: torch.Tensor,      # (M, N2) - X messages (mean) at time t
    v_X: torch.Tensor,      # (M, N2) - X messages (second moment) at time t
    m_W_prev: torch.Tensor, # (N1, M) - W messages from t-1 (for Onsager)
    m_X_prev: torch.Tensor, # (M, N2) - X messages from t-1 (for Onsager)
    Y: torch.Tensor,        # (C,) - Observed values
    F: torch.Tensor,        # (C, M) - Random F factors per edge
    i_idx: torch.Tensor,    # (C,) - Row indices
    j_idx: torch.Tensor,    # (C,) - Column indices
    g_prev: torch.Tensor,   # (C,) - Previous g values
    noise_var: float,       # Noise variance
    damping: float,         # Damping factor
    N1: int,
    N2: int,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single G-AMP step with random F ~ N(0,1) per edge and proper Onsager correction.
    
    The observation model is:
        Y_c = (1/√M) Σ_μ F[c,μ] W_{i_c,μ} X_{μ,j_c}
    
    F is (C, M) - different for each observation.
    
    Onsager Correction (following paper equations):
        ω includes both W-side and X-side Onsager terms with F (not F²)
        T Onsager uses g (not dg) with time-t (v - m²)
    """
    C = Y.shape[0]
    device = Y.device
    
    scale = 1.0 / math.sqrt(M)
    scale_sq = 1.0 / M

    
    # ========================================================================
    # Step 1: Compute omega with Onsager correction
    # ========================================================================
    
    W_sel = m_W[i_idx.long(), :]      # (C, M)
    X_sel = m_X[:, j_idx.long()].T    # (C, M)
    vW_sel = v_W[i_idx.long(), :]     # (C, M)
    vX_sel = v_X[:, j_idx.long()].T   # (C, M)
    W_prev_sel = m_W_prev[i_idx.long(), :]    # (C, M)
    X_prev_sel = m_X_prev[:, j_idx.long()].T  # (C, M)
    
    # Main term: ω_main = (1/√M) Σ_μ F[c,μ] m_W × m_X
    omega_main = scale * (F * W_sel * X_sel).sum(dim=1)  # (C,)
    
    # (v - m^2) at time t (clamp to keep variance non-negative)
    var_term_X = torch.clamp(vX_sel - X_sel ** 2, min=0.0)  # (C, M)
    var_term_W = torch.clamp(vW_sel - W_sel ** 2, min=0.0)  # (C, M)
    
    # Compute F² for Onsager terms
    F_sq = F ** 2  # (C, M)
    
    # W-side Onsager: (1/M) g^{t-1} Σ_μ F²[c,μ] (v_X - m_X²) m_W m_W^{t-1}
    onsager_W_side = scale_sq * (F_sq * var_term_X * W_sel * W_prev_sel).sum(dim=1)  # (C,)
    
    # X-side Onsager: (1/M) g^{t-1} Σ_μ F²[c,μ] (v_W - m_W²) m_X m_X^{t-1}
    onsager_X_side = scale_sq * (F_sq * var_term_W * X_sel * X_prev_sel).sum(dim=1)  # (C,)
    
    # Combined omega with Onsager correction
    omega = omega_main - g_prev * (onsager_W_side + onsager_X_side)  # (C,)
    
    # ========================================================================
    # Step 2: Compute V
    # ========================================================================
    
    # V = (1/M) Σ_μ F[c,μ]² (v_W × v_X - m_W² × m_X²)
    V = scale_sq * (F_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2))).sum(dim=1)
    
    # ========================================================================
    # Step 3: Output function
    # ========================================================================
    
    g, dg = g_out(omega, Y, V, noise_var)
    
    # Apply damping to g (old-value biased)
    g = damping * g_prev + (1 - damping) * g
    
    # ========================================================================
    # Step 4: Compute T Onsager correction coefficients
    # Onsager form: (1/M) g^t g^{t-1} F^2 (v^t - m^{t,2})
    # ========================================================================
    g_pair = (g * g_prev).unsqueeze(1)  # (C, 1)
    
    # Onsager for W: (1/M) g^t g^{t-1} F^2 (v_X^t - m_X^{t,2})
    # Note: m_W^{t-1} is multiplied later in T_W calculation
    onsager_W_contrib = scale_sq * g_pair * F_sq * var_term_X  # (C, M)
    onsager_W = torch.zeros_like(m_W)
    onsager_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), onsager_W_contrib)
    
    # Onsager for X: (1/M) g^t g^{t-1} F^2 (v_W^t - m_W^{t,2})
    # Note: m_X^{t-1} is multiplied later in T_X calculation
    onsager_X_contrib = scale_sq * g_pair * F_sq * var_term_W  # (C, M)
    onsager_X = torch.zeros_like(m_X)
    onsager_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), onsager_X_contrib.T)
    
    # ========================================================================
    # Step 5: Update Sigma, T with Onsager correction
    # Paper formula: Σ = 1 / (Σ_c (1/M) F²_cμ × (-∂g/∂ω) × m²)
    # Since ∂g/∂ω = -1/(V+σ²), we have: Σ = 1 / (Σ_c (1/M) F²_cμ × (1/(V+σ²)) × m²)
    # 
    # Paper formula: T/Σ = m/Σ + sum - Onsager
    # Therefore: T = m + Σ × (sum - Onsager)
    # ========================================================================
    
    # Update W messages
    Sigma_W_denom = torch.zeros_like(m_W)
    
    # Denominator: (1/M) × F²_cμ × (-∂g_c) × m_X²
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2) * F_sq  # (C, M)
    Sigma_W_denom.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    # Apply reciprocal: Σ = 1 / denominator
    Sigma_W = 1.0 / Sigma_W_denom
    
    # Sum contribution: Σ_c (1/√M) F g m_X
    sum_W = torch.zeros_like(m_W)
    g_expanded = scale * g.unsqueeze(1) * X_sel * F  # (C, M)
    sum_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    # T = m + Σ × g × (sum - Onsager × m_prev)
    # where Onsager = (1/M) g^{t-1} F**2 (v_X - m_X²)
    T_W = m_W + Sigma_W * (sum_W - onsager_W * m_W_prev)
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    
    # Update X messages (symmetric to W)
    Sigma_X_denom = torch.zeros_like(m_X)
    
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2) * F_sq
    Sigma_X_denom.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    # Apply reciprocal for X
    Sigma_X = 1.0 / Sigma_X_denom
    
    # Sum contribution for X
    sum_X = torch.zeros_like(m_X)
    g_expanded_X = scale * g.unsqueeze(1) * W_sel * F
    sum_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    # T = m + Σ × (sum - Onsager × m_prev)
    T_X = m_X + Sigma_X * (sum_X - onsager_X * m_X_prev)
    
    # Apply f_input for X
    m_X_new, v_X_new = f_input(Sigma_X, T_X)

    # Optional damping on message updates
    if damping > 0.0:
        m_W_new = damping * m_W + (1 - damping) * m_W_new
        v_W_new = damping * v_W + (1 - damping) * v_W_new
        m_X_new = damping * m_X + (1 - damping) * m_X_new
        v_X_new = damping * v_X + (1 - damping) * v_X_new

    return m_W_new, v_W_new, m_X_new, v_X_new, g


def compute_adaptive_damping(
    alpha: float,
    d_min: float = 0.1,
    d_max: float = 0.9,
    alpha_min: float = 1.0,
    k: float = 0.15,
) -> float:
    """
    Compute adaptive damping based on alpha.

    Higher alpha leads to larger update oscillations, so we increase damping
    to stabilize convergence.

    Formula: damping = clamp(d_min + k * (alpha - alpha_min), d_min, d_max)

    Example with defaults: d_min=0.1, d_max=0.9, alpha_min=1.0, k=0.15
        alpha=1.0 -> damping=0.1
        alpha=6.0 -> damping=0.85
    """
    damping = d_min + k * (alpha - alpha_min)
    return max(d_min, min(d_max, damping))


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    max_steps: int = 500,
    damping: float | None = None,  # None = use adaptive damping
    noise_var: float = 1e-10,
    convergence_threshold: float = 1e-6,
) -> tuple[float, float, int]:
    """
    Train a single replica using G-AMP with F ~ N(0,1) and Onsager correction.

    Observation model:
        Y_c = (1/√M) Σ_μ F[c,μ] W_{i_c,μ} X_{μ,j_c}

    where F[c,μ] ~ N(0, 1) i.i.d. for each edge c and component μ.

    This version includes proper Onsager correction using t-1 messages.

    Args:
        damping: If None, use adaptive damping based on alpha.
                 If float, use the specified fixed damping value.
    """
    # Use adaptive damping if not specified
    if damping is None:
        damping = compute_adaptive_damping(alpha)

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
    
    # Generate observations with F: Y = (1/√M) Σ_μ F[c,μ] W X
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = (1.0 / math.sqrt(M)) * (F * W_sel * X_sel).sum(dim=1)
    
    # Add noise
    torch.manual_seed(seed + 1000)
    noise = torch.randn_like(Y) * math.sqrt(noise_var)
    Y_noisy = Y + noise
    
    # Initialize messages: m ~ 0.1*N(0,1), v = 1.0
    torch.manual_seed(seed + 2000)
    m_W = 0.1 * torch.randn(N1, M, device=device)
    v_W = torch.ones(N1, M, device=device)
    m_X = 0.1 * torch.randn(M, N2, device=device)
    v_X = torch.ones(M, N2, device=device)
    g_prev = torch.zeros(E, device=device)
    
    m_W_prev = m_W.clone()
    m_X_prev = m_X.clone()
    
    # G-AMP iterations
    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float('inf')
    
    for step in range(max_steps):
        m_W_old = m_W.clone()
        m_X_old = m_X.clone()
        
        damping_step = 0.0 if step == 0 else damping
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_F_onsager(
            m_W, v_W, m_X, v_X, m_W_prev, m_X_prev,
            Y_noisy, F, i_idx, j_idx, g_prev,
            noise_var, damping_step, N1, N2, M
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
    
    # Normalize student matrices
    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)
    
    # Compute Q_Y
    qy = compute_qy(m_W, m_X, W_teacher, X_teacher)
    
    return qy, final_loss, steps_taken
