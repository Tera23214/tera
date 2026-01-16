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

from .graph import BiregularGraph
from .utils import normalize_to_unit_variance, compute_qy


# ============================================================================
# G-AMP Functions (Algorithm 2)
# ============================================================================

def f_input(Sigma: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Input function for Gaussian prior (Eq. 178 in paper).
    
    For standard Gaussian prior N(0, 1):
        f_input(Σ, T) = T / (Σ + 1)
        f_input,II(Σ, T) = Σ / (Σ + 1) + T² / (Σ + 1)²
    
    Args:
        Sigma: Inverse variance parameter (Σ)
        T: Mean parameter (scaled)
    
    Returns:
        m: Posterior mean = f_input(Σ, T)
        v: Posterior second moment = f_input,II(Σ, T)
    """
    # Avoid division by zero
    denom = 1.0 + Sigma
    denom = torch.clamp(denom, min=1e-10)
    
    m = T / denom                           # f_input = T / (Σ + 1)
    v = Sigma / denom + (T ** 2) / (denom ** 2)  # f_input,II = Σ/(Σ+1) + T²/(Σ+1)²
    
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
    
    W_sel = m_W[i_idx.long(), :]      # (C, M)
    X_sel = m_X[:, j_idx.long()].T    # (C, M)
    
    # Main term: (λ/√M) Σ_μ m_W × m_X
    omega_main = scale * (W_sel * X_sel).sum(dim=1)  # (C,)
    
    # Onsager correction term: (λ/√M) g^{t-1} Σ (v - m²) × m^{t-1}
    # For F=1: -(λ/√M) g^{t-1} Σ (v_jν - m_jν²) × m_kν × m_kν^{t-1}
    vW_sel = v_W[i_idx.long(), :]     # (C, M)
    vX_sel = v_X[:, j_idx.long()].T   # (C, M)
    W_prev_sel = m_W_prev[i_idx.long(), :]  # (C, M)
    X_prev_sel = m_X_prev[:, j_idx.long()].T  # (C, M)
    
    # (v - m²) term for X side
    var_term_X = vX_sel - X_sel ** 2  # (C, M)
    # Onsager: (λ/√M) g^{t-1} × Σ_μ (v_X - m_X²) × m_W × m_W^{t-1}
    onsager = scale * g_prev.unsqueeze(1) * (var_term_X * W_sel * W_prev_sel).sum(dim=1, keepdim=True)
    onsager = onsager.squeeze(1)  # (C,)
    
    omega = omega_main - onsager  # (C,)
    
    # V = (λ²/M) * Σ_μ (v_W × v_X - m_W² × m_X²)
    # v_W, v_X are second moments E[x²], so this gives variance of WX
    V = scale_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2)).sum(dim=1)  # (C,)
    
    # ========================================================================
    # Step 4: Output function (Alg2.2)
    # ========================================================================

    
    g, dg = g_out(omega, Y, V, noise_var)  # (C,), (C,)
    
    # Apply damping to g and clamp for stability
    g = damping * g + (1 - damping) * g_prev
    g = torch.clamp(g, min=-100.0, max=100.0)  # Prevent extreme g values
    
    # ========================================================================
    # Step 5-6: Update Sigma, T and then m, v (Alg2.3, Alg2.4)
    # ========================================================================
    
    # Update W messages
    # Σ_W[i,μ] = Σ_c (λ/√M)² × (-∂g_c) × m_X[μ, j_c]²
    # T_W[i,μ] = m_W[i,μ] + Σ_c (λ/√M) × g_c × m_X[μ, j_c]
    
    Sigma_W = torch.zeros_like(m_W)
    T_W = m_W.clone()
    
    # Scatter add with λ scaling
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2)  # (C, M)
    Sigma_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    g_expanded = scale * g.unsqueeze(1) * X_sel  # (C, M)
    T_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(torch.clamp(Sigma_W, min=1e-10), T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=10.0)  # Stability clamp
    
    # Update X messages
    # Σ_X[μ,j] = Σ_c (λ/√M)² × (-∂g_c) × m_W[i_c, μ]²
    # T_X[μ,j] = m_X[μ,j] + Σ_c (λ/√M) × g_c × m_W[i_c, μ]
    
    Sigma_X = torch.zeros_like(m_X)
    T_X = m_X.clone()
    
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2)  # (C, M)
    Sigma_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    g_expanded_X = scale * g.unsqueeze(1) * W_sel  # (C, M)
    T_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    # Apply f_input for X
    m_X_new, v_X_new = f_input(torch.clamp(Sigma_X, min=1e-10), T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=10.0)  # Stability clamp
    
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
