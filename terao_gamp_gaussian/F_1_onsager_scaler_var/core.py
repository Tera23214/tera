#!/usr/bin/env python
"""
G-AMP core module with F=1 and scalar-variance Onsager terms.

Implements the G-AMP algorithm for sparse matrix factorization
with F = 1 (constant) and Onsager correction using the scalar averages

    chi_W = mean(v_W - m_W^2)
    chi_X = mean(v_X - m_X^2)

instead of node-dependent variances.
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
# G-AMP Functions with F=1 and Onsager Correction
# ============================================================================


def gamp_step_with_onsager(
    m_W: torch.Tensor,      # (N1, M) - W messages (mean) at time t
    v_W: torch.Tensor,      # (N1, M) - W messages (second moment) at time t
    m_X: torch.Tensor,      # (M, N2) - X messages (mean) at time t
    v_X: torch.Tensor,      # (M, N2) - X messages (second moment) at time t
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
    Single G-AMP step with F = 1 and scalar-variance Onsager correction.
    
    The observation model is:
        Y_c = (λ/√M) Σ_μ W_{i_c,μ} X_{μ,j_c}
    
    F = 1 for all edges and components (no random spreading).
    λ is the signal strength parameter.
    
    Onsager correction uses the scalar averages

        chi_W = mean(v_W - m_W^2)
        chi_X = mean(v_X - m_X^2)

    and applies the T-side correction through g(t) g(t-1).
    """
    device = Y.device
    i_idx_long = i_idx.long()
    j_idx_long = j_idx.long()

    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    # ========================================================================
    # Step 1: Compute omega with scalar chi Onsager correction
    # ========================================================================

    W_sel = m_W[i_idx_long, :]        # (C, M)
    X_sel = m_X[:, j_idx_long].T      # (C, M)

    # Scalar approximation to v - m^2. Clamp keeps the effective variance
    # non-negative, matching the stability treatment used in the base code.
    chi_W = torch.clamp(v_W - m_W ** 2, min=0.0).mean()
    chi_X = torch.clamp(v_X - m_X ** 2, min=0.0).mean()

    row_cross_W = (m_W * m_W_prev).sum(dim=1)  # (N1,)
    col_cross_X = (m_X * m_X_prev).sum(dim=0)  # (N2,)
    row_sq_W = (m_W ** 2).sum(dim=1)           # (N1,)
    col_sq_X = (m_X ** 2).sum(dim=0)           # (N2,)

    # ω_c = (λ/√M) Σ_μ m_W m_X
    #       - g_prev[c] (λ²/M) [chi_X Σ_μ m_W m_W_prev + chi_W Σ_μ m_X m_X_prev]
    omega = scale * (W_sel * X_sel).sum(dim=1) - g_prev * scale_sq * (
        chi_X * row_cross_W[i_idx_long] + chi_W * col_cross_X[j_idx_long]
    )

    # ========================================================================
    # Step 2: Compute V with scalar chi approximation
    # ========================================================================

    # Since v_W ≈ m_W^2 + chi_W and v_X ≈ m_X^2 + chi_X,
    # v_W v_X - m_W^2 m_X^2 ≈ chi_W chi_X + chi_X m_W^2 + chi_W m_X^2.
    V = scale_sq * (
        M * chi_W * chi_X
        + chi_X * row_sq_W[i_idx_long]
        + chi_W * col_sq_X[j_idx_long]
    )
    V = torch.clamp(V, min=1e-10)
    
    # ========================================================================
    # Step 3: Output function
    # ========================================================================
    
    g, dg = g_out(omega, Y, V, noise_var)
    
    # Apply damping to g and clamp for stability (old-value biased)
    g = damping * g_prev + (1 - damping) * g
    g = torch.clamp(g, min=-100.0, max=100.0)  # Prevent extreme g values
    
    # ========================================================================
    # Step 4: Compute T Onsager correction coefficients
    # Uses g(t) g(t-1) together with the scalar chi approximation.
    # ========================================================================

    g_pair = g * g_prev  # (C,)

    onsager_W = torch.zeros(N1, device=device, dtype=m_W.dtype)
    onsager_W.scatter_add_(0, i_idx_long, g_pair)
    onsager_W = scale_sq * chi_X * onsager_W  # (N1,)

    onsager_X = torch.zeros(N2, device=device, dtype=m_X.dtype)
    onsager_X.scatter_add_(0, j_idx_long, g_pair)
    onsager_X = scale_sq * chi_W * onsager_X  # (N2,)
    
    # ========================================================================
    # Step 5: Update Sigma, T with Onsager correction
    # Paper formula: Σ = -1 / (Σ_c (λ²/M) × ∂g/∂ω × m²)
    # Since ∂g/∂ω = -1/(V+σ²), we have: Σ = 1 / (Σ_c (λ²/M) × (1/(V+σ²)) × m²)
    # 
    # Paper formula: T/Σ = m/Σ + sum - Onsager
    # Therefore: T = m + Σ × (sum - Onsager)
    # Note: F=1, so no F² factor in Sigma and no F factor in T
    # ========================================================================
    
    # Update W messages
    Sigma_W_denom = torch.zeros_like(m_W)
    
    # Denominator: (λ²/M) × (-∂g_c) × m_X²
    # Note: F=1, so no F² factor
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2)  # (C, M)
    Sigma_W_denom.scatter_add_(0, i_idx_long.unsqueeze(1).expand(-1, M), dg_expanded)
    
    # Apply reciprocal: Σ = 1 / denominator
    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    
    # Sum contribution: Σ_c (λ/√M) g m_X
    # Note: F=1, so no F factor
    sum_W = torch.zeros_like(m_W)
    g_expanded = scale * g.unsqueeze(1) * X_sel  # (C, M)
    sum_W.scatter_add_(0, i_idx_long.unsqueeze(1).expand(-1, M), g_expanded)

    # T_W = m_W + Σ_W [sum_W - (λ²/M) chi_X m_W_prev Σ g(t) g(t-1)]
    T_W = m_W + Sigma_W * (sum_W - onsager_W.unsqueeze(1) * m_W_prev)
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=100.0)  # Stability clamp
    
    # Update X messages (symmetric to W)
    Sigma_X_denom = torch.zeros_like(m_X)
    
    # Note: F=1, so no F² factor
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2)
    Sigma_X_denom.scatter_add_(1, j_idx_long.unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    # Apply reciprocal for X
    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    
    # Sum contribution for X: Σ_c (λ/√M) g m_W
    # Note: F=1, so no F factor
    sum_X = torch.zeros_like(m_X)
    g_expanded_X = scale * g.unsqueeze(1) * W_sel
    sum_X.scatter_add_(1, j_idx_long.unsqueeze(0).expand(M, -1), g_expanded_X.T)

    # T_X = m_X + Σ_X [sum_X - (λ²/M) chi_W m_X_prev Σ g(t) g(t-1)]
    T_X = m_X + Sigma_X * (sum_X - onsager_X.unsqueeze(0) * m_X_prev)
    
    # Apply f_input for X
    m_X_new, v_X_new = f_input(Sigma_X, T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=100.0)  # Stability clamp
    
    # Apply damping (old-value biased)
    m_W_new = damping * m_W + (1 - damping) * m_W_new
    v_W_new = damping * v_W + (1 - damping) * v_W_new
    m_X_new = damping * m_X + (1 - damping) * m_X_new
    v_X_new = damping * v_X + (1 - damping) * v_X_new
    
    return m_W_new, v_W_new, m_X_new, v_X_new, g


def compute_observed_loss(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Compute MSE on observed entries.
    """
    W_sel = m_W[i_idx.long(), :]
    X_sel = m_X[:, j_idx.long()].T
    Y_pred = scale * (W_sel * X_sel).sum(dim=1)

    return ((Y_obs - Y_pred) ** 2).mean()


def compute_step_damping(
    step: int,
    base_damping: float,
    use_step_damping: bool,
    beta_scale: float,
    beta_max: float,
) -> float:
    """
    Compute damping factor for the current step.

    If use_step_damping is enabled:
        beta(step) = min(step * beta_scale, beta_max)
    Otherwise use the fixed base_damping.
    """
    if not use_step_damping:
        damping_t = base_damping
    else:
        damping_t = min(step * beta_scale, beta_max)

    # Keep damping in a numerically safe range.
    return float(max(0.0, min(1.0, damping_t)))


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    max_steps: int = 500,
    damping: float = 0.5,
    use_step_damping: bool = False,
    damping_beta_scale: float = 1e-3,
    damping_beta_max: float = 0.5,
    noise_var: float = 1e-10,
    convergence_threshold: float = 1e-6,
    lam: float = 1.0,  # Signal strength λ
    return_history: bool = False,
    loss_eval_interval: int = 50,
    early_stop: bool = True,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    """
    Train a single replica using G-AMP with F = 1 and Onsager correction.
    
    Observation model:
        Y_c = (λ/√M) Σ_μ W_{i_c,μ} X_{μ,j_c}
    
    where F = 1 (constant, no random spreading).
    
    This version includes proper Onsager correction using t-1 messages.
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
    
    # Generate observations: Y = (λ/√M) Σ_μ W X
    # Note: F=1, so no F factor
    scale = lam / math.sqrt(M)
    W_sel = W_teacher[i_idx.long(), :]
    X_sel = X_teacher[:, j_idx.long()].T
    Y = scale * (W_sel * X_sel).sum(dim=1)
    
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
    
    m_W_prev = m_W.clone()
    m_X_prev = m_X.clone()
    
    # G-AMP iterations
    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float('inf')
    history = {"steps": [], "loss": [], "qy": [], "damping": []}
    history_loss_tensors = []
    history_qy_values = []
    
    for step in range(max_steps):
        # Keep references to time-t messages for the next Onsager memory term.
        # This is safe because gamp_step_with_onsager returns new tensors and
        # does not in-place mutate m_W/m_X.
        m_W_old = m_W
        m_X_old = m_X

        damping_t = compute_step_damping(
            step=step,
            base_damping=damping,
            use_step_damping=use_step_damping,
            beta_scale=damping_beta_scale,
            beta_max=damping_beta_max,
        )
        
        m_W, v_W, m_X, v_X, g_prev = gamp_step_with_onsager(
            m_W, v_W, m_X, v_X, m_W_prev, m_X_prev,
            Y_noisy, i_idx, j_idx, g_prev,
            lam, noise_var, damping_t, N1, N2, M
        )
        
        m_W_prev = m_W_old
        m_X_prev = m_X_old
        
        # Check convergence and optionally record the full loss trace.
        if step % loss_eval_interval == 0 or step == max_steps - 1:
            loss_tensor = compute_observed_loss(
                m_W, m_X, Y_noisy, i_idx, j_idx, scale
            )

            if return_history:
                qy_step = compute_qy(
                    normalize_to_unit_variance(m_W),
                    normalize_to_unit_variance(m_X),
                    W_teacher,
                    X_teacher,
                )
                history["steps"].append(step + 1)
                history_loss_tensors.append(loss_tensor.detach())
                history_qy_values.append(qy_step)
                history["damping"].append(damping_t)

            loss = None
            if early_stop or not return_history or step == max_steps - 1:
                loss = float(loss_tensor.item())
                final_loss = loss

            if early_stop:
                if loss is None:
                    loss = float(loss_tensor.item())
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
    
    if return_history:
        if history_loss_tensors:
            history["loss"] = torch.stack(history_loss_tensors).cpu().tolist()
            history["qy"] = history_qy_values
            if not early_stop:
                final_loss = float(history["loss"][-1])
        return qy, final_loss, steps_taken, history

    return qy, final_loss, steps_taken
