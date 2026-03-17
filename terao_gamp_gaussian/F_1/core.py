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

from terao_gamp_gaussian.graph import BiregularGraph
from terao_gamp_gaussian.utils import normalize_to_unit_variance, compute_qy, f_input, g_out


# ============================================================================
# G-AMP Functions (Algorithm 2)
# ============================================================================


def gamp_step(
    m_W: torch.Tensor,      # (N1, M) - W messages (mean)
    v_W: torch.Tensor,      # (N1, M) - W messages (variance)
    m_X: torch.Tensor,      # (M, N2) - X messages (mean)
    v_X: torch.Tensor,      # (M, N2) - X messages (variance)
    m_W_prev: torch.Tensor, # (N1, M) - W messages from t-1 (for Onsager)
    m_X_prev: torch.Tensor, # (M, N2) - X messages from t-1 (for Onsager)
    Y: torch.Tensor,        # (C,) - Observed values
    i_idx: torch.Tensor,    # (C,) - Row indices
    j_idx: torch.Tensor,    # (C,) - Column indices
    g_prev: torch.Tensor,   # (C,) - Previous g values
    lam: float,             # Lambda = signal strength
    noise_var: float,       # Noise variance
    damping: float,         # Damping factor
    N1: int,
    N2: int,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single G-AMP step (Algorithm 2) with λ signal strength and Onsager correction.
    
    The scaling factors are:
    - ω: (λ/√M) × Σ m_W × m_X - Onsager term
    - V: (λ/√M)² × Σ (v_j - m_j²)
    - Σ: (λ/√M)² coefficient
    - T: (λ/√M) coefficient
    
    Returns:
        m_W_new, v_W_new, m_X_new, v_X_new, g_new
    """
    C = Y.shape[0]
    device = Y.device
    
    # Scale factors
    scale = lam / math.sqrt(M)        # λ/√M
    scale_sq = (lam ** 2) / M         # (λ/√M)² = λ²/M
    
    # ========================================================================
    # Step 3: Update omega and V (Alg2.1) with Onsager correction
    # ========================================================================
    
    W_sel = m_W[i_idx.long(), :]      # (C, M) - m_W^t
    X_sel = m_X[:, j_idx.long()].T    # (C, M) - m_X^t
    vW_sel = v_W[i_idx.long(), :]     # (C, M) - v_W^t
    vX_sel = v_X[:, j_idx.long()].T   # (C, M) - v_X^t
    W_prev_sel = m_W_prev[i_idx.long(), :]    # (C, M) - m_W^{t-1}
    X_prev_sel = m_X_prev[:, j_idx.long()].T  # (C, M) - m_X^{t-1}
    
    # Main term: (λ/√M) Σ_μ m_W^t × m_X^t
    omega_main = scale * (W_sel * X_sel).sum(dim=1)  # (C,)
    
    # Onsager correction (both W-side and X-side) with λ²/M scaling
    # Formula: ω = main - (λ²/M) g^{t-1} Σ_j (v_j^t - (m_j^t)²) × Π_k m_k^t × m_k^{t-1}
    
    # (v - m²) variance terms at time t - clamp to ensure non-negative
    var_term_X = torch.clamp(vX_sel - X_sel ** 2, min=0.0)  # (C, M)
    var_term_W = torch.clamp(vW_sel - W_sel ** 2, min=0.0)  # (C, M)
    
    # W-side Onsager: -(λ²/M) g^{t-1} Σ_μ (v_X^t - (m_X^t)²) × m_W^t × m_W^{t-1}
    onsager_W_side = scale_sq * (var_term_X * W_sel * W_prev_sel).sum(dim=1)  # (C,)
    
    # X-side Onsager: -(λ²/M) g^{t-1} Σ_μ (v_W^t - (m_W^t)²) × m_X^t × m_X^{t-1}
    onsager_X_side = scale_sq * (var_term_W * X_sel * X_prev_sel).sum(dim=1)  # (C,)
    
    # Combined omega with both Onsager terms
    omega = omega_main - g_prev * (onsager_W_side + onsager_X_side)  # (C,)
    
    # V = (λ²/M) * Σ_μ (v_W × v_X - m_W² × m_X²)
    # v_W, v_X are second moments E[x²], so this gives variance of WX
    V = scale_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2)).sum(dim=1)  # (C,)
    
    # ========================================================================
    # Step 4: Output function (Alg2.2)
    # ========================================================================

    
    g, dg = g_out(omega, Y, V, noise_var)  # (C,), (C,)
    
    # Apply damping to g and clamp for stability (old-value biased)
    g = damping * g_prev + (1 - damping) * g
    g = torch.clamp(g, min=-100.0, max=100.0)  # Prevent extreme g values
    
    # ========================================================================
    # Step 5-6: Update Sigma, T and then m, v (Alg2.3, Alg2.4)
    # ========================================================================
    
    # Update W messages
    # Σ_W⁻¹[i,μ] = Σ_c (λ²/M) × (-∂g_c) × m_X[μ, j_c]²
    # Σ_W[i,μ] = 1 / Σ_W⁻¹[i,μ]
    # sum_W[i,μ] = Σ_c (λ/√M) × g_c × m_X[μ, j_c]
    # T_W[i,μ] = m_W[i,μ] + Σ_W[i,μ] × sum_W[i,μ]
    
    Sigma_W_denom = torch.zeros_like(m_W)
    
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2)  # (C, M)
    Sigma_W_denom.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    
    sum_W = torch.zeros_like(m_W)
    g_expanded = scale * g.unsqueeze(1) * X_sel  # (C, M)
    sum_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    T_W = m_W + Sigma_W * sum_W
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=10.0)  # Stability clamp
    
    # Update X messages
    # Σ_X⁻¹[μ,j] = Σ_c (λ²/M) × (-∂g_c) × m_W[i_c, μ]²
    # Σ_X[μ,j] = 1 / Σ_X⁻¹[μ,j]
    # sum_X[μ,j] = Σ_c (λ/√M) × g_c × m_W[i_c, μ]
    # T_X[μ,j] = m_X[μ,j] + Σ_X[μ,j] × sum_X[μ,j]
    
    Sigma_X_denom = torch.zeros_like(m_X)
    
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2)  # (C, M)
    Sigma_X_denom.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    
    sum_X = torch.zeros_like(m_X)
    g_expanded_X = scale * g.unsqueeze(1) * W_sel  # (C, M)
    sum_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    T_X = m_X + Sigma_X * sum_X
    
    # Apply f_input for X
    m_X_new, v_X_new = f_input(Sigma_X, T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=10.0)  # Stability clamp
    
    # Apply damping to messages (old-value biased)
    m_W_new = damping * m_W + (1 - damping) * m_W_new
    v_W_new = damping * v_W + (1 - damping) * v_W_new
    m_X_new = damping * m_X + (1 - damping) * m_X_new
    v_X_new = damping * v_X + (1 - damping) * v_X_new
    
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
    lam: float = 1.0,  # Signal strength λ
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
        noise_var: Noise variance (ΔZ ~ N(0, noise_var))
        convergence_threshold: Convergence threshold
        lam: Signal strength λ (Y_obs = (λ/√M) WX + ΔZ)
    
    Returns:
        qy: Q_Y overlap metric
        final_loss: Final MSE loss
        steps_taken: Number of iterations
    """
    # Generate teacher matrices
    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    
    # Generate observation graph (biregular)
    graph = BiregularGraph()
    i_idx, j_idx, E, C1, C2, alpha2 = graph.generate(N1, N2, M, alpha, device, seed)
    
    if E == 0:
        return 0.0, 0.0, 0
    
    # Generate true observation (for Q_Y evaluation): Y_true = (1/√M) WX
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y_true = (W_sel * X_sel).sum(dim=1) / math.sqrt(M)  # (C,) - clean, unscaled
    
    # Generate noisy observation for student: Y_noisy = (λ/√M) WX + ΔZ
    scale = lam / math.sqrt(M)
    Y_scaled = (W_sel * X_sel).sum(dim=1) * scale  # (C,) - with λ scaling
    torch.manual_seed(seed + 500)
    noise = torch.randn_like(Y_scaled) * math.sqrt(noise_var)
    Y_noisy = Y_scaled + noise  # This is what the student observes
    
    # Initialize messages (random values from N(0,1) like teacher)
    torch.manual_seed(seed + 1000)
    m_W = torch.randn(N1, M, device=device)
    v_W = torch.ones(N1, M, device=device)
    m_X = torch.randn(M, N2, device=device)
    v_X = torch.ones(M, N2, device=device)
    g_prev = torch.zeros(E, device=device)
    
    # Previous messages for Onsager correction (start with same as current)
    m_W_prev = m_W.clone()
    m_X_prev = m_X.clone()
    
    # G-AMP iterations (using noisy observations)
    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float('inf')
    
    for step in range(max_steps):
        # Store current m for next iteration's Onsager term
        m_W_old = m_W.clone()
        m_X_old = m_X.clone()
        
        m_W, v_W, m_X, v_X, g_prev = gamp_step(
            m_W, v_W, m_X, v_X, m_W_prev, m_X_prev,
            Y_noisy, i_idx, j_idx, g_prev,
            lam, noise_var, damping, N1, N2, M
        )
        
        # Update prev for next iteration
        m_W_prev = m_W_old
        m_X_prev = m_X_old
        
        # Check convergence every 50 steps
        if step % 50 == 0 or step == max_steps - 1:
            # Compute predictions
            W_sel = m_W[i_idx.long(), :]
            X_sel = m_X[:, j_idx.long()].T
            Y_pred = (W_sel * X_sel).sum(dim=1) / math.sqrt(M)
            
            loss = ((Y_noisy - Y_pred) ** 2).mean().item()
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
