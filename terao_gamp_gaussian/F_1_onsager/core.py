#!/usr/bin/env python
"""
G-AMP (Generalized Approximate Message Passing) Core Module with F=1 and Onsager Term.

Implements the G-AMP algorithm for sparse matrix factorization
with F = 1 (constant) and proper Onsager correction.

Observation model:
    Y_obs = (λ/√M) Σ_μ W_{i,μ} X_{μ,j} + ΔZ

where F = 1 (constant, no spreading) and λ is the signal strength.

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
    Single G-AMP step with F = 1 (constant) and proper Onsager correction.
    
    The observation model is:
        Y_c = (λ/√M) Σ_μ W_{i_c,μ} X_{μ,j_c}
    
    F = 1 for all edges and components (no random spreading).
    λ is the signal strength parameter.
    
    Onsager Correction (following paper equations):
        ω includes both W-side and X-side Onsager terms
        T Onsager uses g (not dg) with time-t (v - m²)
    """
    C = Y.shape[0]
    device = Y.device
    
    scale = lam / math.sqrt(M)        # λ/√M
    scale_sq = (lam ** 2) / M         # (λ/√M)² = λ²/M
    
    # ========================================================================
    # Step 1: Compute omega with Onsager correction
    # ========================================================================
    
    W_sel = m_W[i_idx.long(), :]      # (C, M)
    X_sel = m_X[:, j_idx.long()].T    # (C, M)
    vW_sel = v_W[i_idx.long(), :]     # (C, M)
    vX_sel = v_X[:, j_idx.long()].T   # (C, M)
    W_prev_sel = m_W_prev[i_idx.long(), :]    # (C, M)
    X_prev_sel = m_X_prev[:, j_idx.long()].T  # (C, M)
    
    # Main term: ω_main = (λ/√M) Σ_μ m_W × m_X
    # Note: F=1, so no F factor
    omega_main = scale * (W_sel * X_sel).sum(dim=1)  # (C,)
    
    # (v - m²) at time t - clamp to ensure non-negative (can be negative due to v clamping)
    var_term_X = torch.clamp(vX_sel - X_sel ** 2, min=0.0)  # (C, M)
    var_term_W = torch.clamp(vW_sel - W_sel ** 2, min=0.0)  # (C, M)
    
    # W-side Onsager: (λ²/M) Σ_μ (v_X - m_X²) m_W m_W^{t-1}
    onsager_W_side = scale_sq * (var_term_X * W_sel * W_prev_sel).sum(dim=1)  # (C,)
    
    # X-side Onsager: (λ²/M) Σ_μ (v_W - m_W²) m_X m_X^{t-1}
    onsager_X_side = scale_sq * (var_term_W * X_sel * X_prev_sel).sum(dim=1)  # (C,)
    
    # Combined omega with Onsager correction
    #omaga_main - g^{t-1} (onsager_W + onsager_X)
    omega = omega_main - g_prev * (onsager_W_side + onsager_X_side)  # (C,)
    
    # ========================================================================
    # Step 2: Compute V
    # ========================================================================
    
    # V = (λ²/M) Σ_μ (v_W × v_X - m_W² × m_X²)
    # Note: F=1, so no F² factor
    V = scale_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2)).sum(dim=1)
    V = torch.clamp(V, min=1e-10)  # Ensure positive
    
    # ========================================================================
    # Step 3: Output function
    # ========================================================================
    
    g, dg = g_out(omega, Y, V, noise_var)
    
    # Apply damping to g and clamp for stability (old-value biased)
    g = damping * g_prev + (1 - damping) * g
    g = torch.clamp(g, min=-100.0, max=100.0)  # Prevent extreme g values
    
    # ========================================================================
    # Step 4: Compute T Onsager correction coefficients
    # Uses node-dependent (v - m^2) together with g(t) g(t-1).
    # ========================================================================

    g_pair = g * g_prev

    # Onsager for W: (λ²/M) g(t) g(t-1) × (v_X^t - m_X^{t,2})
    # Note: m_W^{t-1} is multiplied later in T_W calculation
    onsager_W_contrib = scale_sq * g_pair.unsqueeze(1) * var_term_X  # (C, M)
    onsager_W = torch.zeros_like(m_W)
    onsager_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), onsager_W_contrib)
    
    # Onsager for X: (λ²/M) g(t) g(t-1) × (v_W^t - m_W^{t,2})
    # Note: m_X^{t-1} is multiplied later in T_X calculation
    onsager_X_contrib = scale_sq * g_pair.unsqueeze(1) * var_term_W  # (C, M)
    onsager_X = torch.zeros_like(m_X)
    onsager_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), onsager_X_contrib.T)
    
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
    Sigma_W_denom.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), dg_expanded)
    
    # Apply reciprocal: Σ = 1 / denominator
    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    
    # Sum contribution: Σ_c (λ/√M) g m_X
    # Note: F=1, so no F factor
    sum_W = torch.zeros_like(m_W)
    g_expanded = scale * g.unsqueeze(1) * X_sel  # (C, M)
    sum_W.scatter_add_(0, i_idx.long().unsqueeze(1).expand(-1, M), g_expanded)
    
    # T = m + Σ × (sum - Onsager × m_prev)
    # where Onsager = (λ²/M) g(t) g(t-1) (v_X - m_X²)
    T_W = m_W + Sigma_W * (sum_W - onsager_W * m_W_prev)
    
    # Apply f_input for W
    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=100.0)  # Stability clamp
    
    # Update X messages (symmetric to W)
    Sigma_X_denom = torch.zeros_like(m_X)
    
    # Note: F=1, so no F² factor
    dg_expanded_X = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2)
    Sigma_X_denom.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), dg_expanded_X.T)
    
    # Apply reciprocal for X
    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    
    # Sum contribution for X: Σ_c (λ/√M) g m_W
    # Note: F=1, so no F factor
    sum_X = torch.zeros_like(m_X)
    g_expanded_X = scale * g.unsqueeze(1) * W_sel
    sum_X.scatter_add_(1, j_idx.long().unsqueeze(0).expand(M, -1), g_expanded_X.T)
    
    # T = m + Σ × (sum - Onsager × m_prev)
    # where Onsager = (λ²/M) g(t) g(t-1) (v_W - m_W²)
    T_X = m_X + Sigma_X * (sum_X - onsager_X * m_X_prev)
    
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
        beta(step) = max(1 - step * beta_scale, beta_max)
    Starts at 1.0 (very gentle) and decreases, with beta_max as the
    lower bound. Ensures damping never falls below beta_max.
    Otherwise use the fixed base_damping.
    """
    if not use_step_damping:
        damping_t = base_damping
    else:
        damping_t = max(1.0 - step * beta_scale, beta_max)

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
            m_W_eval = normalize_to_unit_variance(m_W)
            m_X_eval = normalize_to_unit_variance(m_X)
            loss_tensor = compute_observed_loss(
                m_W_eval, m_X_eval, Y_noisy, i_idx, j_idx, scale
            )

            if return_history:
                qy_step = compute_qy(
                    m_W_eval,
                    m_X_eval,
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
