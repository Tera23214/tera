#!/usr/bin/env python
"""
Shared alternating edge-observed G-AMP logic used by Edge_Alternating variants.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.utils import (
    f_input,
    g_out,
    initialize_correlated_student,
    normalize_to_unit_variance,
)


def compute_y_cosine_similarity(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> float:
    """
    Compute cosine similarity between Y_teacher = W_teacher X_teacher and
    Y_student = W_student X_student without materializing the dense Y matrices.
    """
    teacher_w_t = W_teacher.T.contiguous()
    teacher_x_t = X_teacher.T.contiguous()
    teacher_w_gram = teacher_w_t @ W_teacher
    teacher_x_gram = X_teacher @ teacher_x_t
    teacher_norm_sq = torch.sum(teacher_w_gram * teacher_x_gram.T)

    cosine_similarity = compute_y_cosine_similarity_tensor(
        W_student=W_student,
        X_student=X_student,
        teacher_w_t=teacher_w_t,
        teacher_x_t=teacher_x_t,
        teacher_norm_sq=teacher_norm_sq,
    )

    return float(cosine_similarity.item())


def compute_y_cosine_similarity_tensor(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    teacher_w_t: torch.Tensor,
    teacher_x_t: torch.Tensor,
    teacher_norm_sq: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Y-space cosine similarity and keep the result on-device.
    """
    cross_w = teacher_w_t @ W_student
    cross_x_t = teacher_x_t.T @ X_student.T
    inner = torch.sum(cross_w * cross_x_t)

    student_w_gram = W_student.T @ W_student
    student_x_gram = X_student @ X_student.T
    student_norm_sq = torch.sum(student_w_gram * student_x_gram.T)
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))

    return inner / denom


def _compute_output_channel_terms(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    g_prev_edge: torch.Tensor,
    lam: float,
    noise_var: float,
    damping: float,
    M: int,
    F_edge: torch.Tensor | None = None,
) -> tuple[
    float,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute output-channel quantities for one alternating half-step.
    """
    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    i_long = i_idx.long()
    j_long = j_idx.long()

    W_sel = m_W[i_long, :]
    X_sel = m_X[:, j_long].T
    vW_sel = v_W[i_long, :]
    vX_sel = v_X[:, j_long].T
    W_prev_sel = m_W_prev[i_long, :]
    X_prev_sel = m_X_prev[:, j_long].T

    var_term_W = torch.clamp(vW_sel - W_sel ** 2, min=0.0)
    var_term_X = torch.clamp(vX_sel - X_sel ** 2, min=0.0)

    if F_edge is None:
        omega_main = scale * (W_sel * X_sel).sum(dim=1)
        onsager_W_side = scale_sq * (var_term_X * W_sel * W_prev_sel).sum(dim=1)
        onsager_X_side = scale_sq * (var_term_W * X_sel * X_prev_sel).sum(dim=1)
        V = scale_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2)).sum(dim=1)
        weighted_var_X = var_term_X
        weighted_var_W = var_term_W
    else:
        F_sq = F_edge ** 2
        omega_main = scale * (W_sel * F_edge * X_sel).sum(dim=1)
        onsager_W_side = scale_sq * (
            F_sq * var_term_X * W_sel * W_prev_sel
        ).sum(dim=1)
        onsager_X_side = scale_sq * (
            F_sq * var_term_W * X_sel * X_prev_sel
        ).sum(dim=1)
        V = scale_sq * (
            F_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2))
        ).sum(dim=1)
        weighted_var_X = F_sq * var_term_X
        weighted_var_W = F_sq * var_term_W

    omega = omega_main - g_prev_edge * (onsager_W_side + onsager_X_side)
    V = torch.clamp(V, min=1e-10)

    g_raw, dg = g_out(omega, Y_obs, V, noise_var)
    g_edge = damping * g_prev_edge + (1.0 - damping) * g_raw
    g_edge = torch.clamp(g_edge, min=-100.0, max=100.0)

    g_pair = g_edge * g_prev_edge
    onsager_W_contrib = scale_sq * g_pair.unsqueeze(1) * weighted_var_X
    onsager_X_contrib = scale_sq * g_pair.unsqueeze(1) * weighted_var_W

    onsager_W = torch.zeros_like(m_W)
    onsager_W.scatter_add_(0, i_long.unsqueeze(1).expand(-1, M), onsager_W_contrib)

    onsager_X = torch.zeros_like(m_X)
    onsager_X.scatter_add_(
        1,
        j_long.unsqueeze(0).expand(M, -1),
        onsager_X_contrib.T.contiguous(),
    )

    return scale, scale_sq, dg, g_edge, onsager_W, onsager_X


def alternating_half_step_W(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    g_prev_edge: torch.Tensor,
    lam: float,
    noise_var: float,
    damping: float,
    M: int,
    F_edge: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update W while freezing X using observed-edge messages.
    """
    scale, scale_sq, dg, g_edge, onsager_W, _ = _compute_output_channel_terms(
        m_W=m_W,
        v_W=v_W,
        m_X=m_X,
        v_X=v_X,
        m_W_prev=m_W_prev,
        m_X_prev=m_X_prev,
        Y_obs=Y_obs,
        i_idx=i_idx,
        j_idx=j_idx,
        g_prev_edge=g_prev_edge,
        lam=lam,
        noise_var=noise_var,
        damping=damping,
        M=M,
        F_edge=F_edge,
    )

    i_long = i_idx.long()
    j_long = j_idx.long()
    X_sel = m_X[:, j_long].T

    if F_edge is None:
        dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2)
        g_expanded = scale * g_edge.unsqueeze(1) * X_sel
    else:
        F_sq = F_edge ** 2
        dg_expanded = scale_sq * (-dg).unsqueeze(1) * F_sq * (X_sel ** 2)
        g_expanded = scale * g_edge.unsqueeze(1) * F_edge * X_sel

    Sigma_W_denom = torch.zeros_like(m_W)
    Sigma_W_denom.scatter_add_(0, i_long.unsqueeze(1).expand(-1, M), dg_expanded)
    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)

    sum_W = torch.zeros_like(m_W)
    sum_W.scatter_add_(0, i_long.unsqueeze(1).expand(-1, M), g_expanded)

    T_W = m_W + Sigma_W * (sum_W - onsager_W * m_W_prev)

    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=100.0)

    m_W_new = damping * m_W + (1.0 - damping) * m_W_new
    v_W_new = damping * v_W + (1.0 - damping) * v_W_new

    return m_W_new, v_W_new, g_edge


def alternating_half_step_X(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    g_prev_edge: torch.Tensor,
    lam: float,
    noise_var: float,
    damping: float,
    M: int,
    F_edge: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update X while freezing W using observed-edge messages.
    """
    scale, scale_sq, dg, g_edge, _, onsager_X = _compute_output_channel_terms(
        m_W=m_W,
        v_W=v_W,
        m_X=m_X,
        v_X=v_X,
        m_W_prev=m_W_prev,
        m_X_prev=m_X_prev,
        Y_obs=Y_obs,
        i_idx=i_idx,
        j_idx=j_idx,
        g_prev_edge=g_prev_edge,
        lam=lam,
        noise_var=noise_var,
        damping=damping,
        M=M,
        F_edge=F_edge,
    )

    i_long = i_idx.long()
    j_long = j_idx.long()
    W_sel = m_W[i_long, :]

    if F_edge is None:
        dg_expanded = scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2)
        g_expanded = scale * g_edge.unsqueeze(1) * W_sel
    else:
        F_sq = F_edge ** 2
        dg_expanded = scale_sq * (-dg).unsqueeze(1) * F_sq * (W_sel ** 2)
        g_expanded = scale * g_edge.unsqueeze(1) * F_edge * W_sel

    Sigma_X_denom = torch.zeros_like(m_X)
    Sigma_X_denom.scatter_add_(
        1,
        j_long.unsqueeze(0).expand(M, -1),
        dg_expanded.T.contiguous(),
    )
    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)

    sum_X = torch.zeros_like(m_X)
    sum_X.scatter_add_(
        1,
        j_long.unsqueeze(0).expand(M, -1),
        g_expanded.T.contiguous(),
    )

    T_X = m_X + Sigma_X * (sum_X - onsager_X * m_X_prev)

    m_X_new, v_X_new = f_input(Sigma_X, T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=100.0)

    m_X_new = damping * m_X + (1.0 - damping) * m_X_new
    v_X_new = damping * v_X + (1.0 - damping) * v_X_new

    return m_X_new, v_X_new, g_edge


def compute_observed_loss(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
    F_edge: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute MSE on observed entries.
    """
    i_long = i_idx.long()
    j_long = j_idx.long()
    W_sel = m_W[i_long, :]
    X_sel = m_X[:, j_long].T

    if F_edge is None:
        Y_pred = scale * (W_sel * X_sel).sum(dim=1)
    else:
        Y_pred = scale * (W_sel * F_edge * X_sel).sum(dim=1)

    return ((Y_obs - Y_pred) ** 2).mean()


def compute_observed_signal_cosine_tensor(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    Y_clean_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
    F_edge: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute cosine similarity in the observed signal space.
    """
    i_long = i_idx.long()
    j_long = j_idx.long()
    W_sel = W_student[i_long, :]
    X_sel = X_student[:, j_long].T

    if F_edge is None:
        Y_student = scale * (W_sel * X_sel).sum(dim=1)
    else:
        Y_student = scale * (W_sel * F_edge * X_sel).sum(dim=1)

    inner = torch.sum(Y_clean_obs * Y_student)
    teacher_norm_sq = torch.sum(Y_clean_obs ** 2)
    student_norm_sq = torch.sum(Y_student ** 2)
    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))

    return inner / denom


def compute_step_damping(
    step: int,
    base_damping: float,
    use_step_damping: bool,
    beta_scale: float,
    beta_max: float,
) -> float:
    """
    Compute damping factor for the current outer step.
    """
    if not use_step_damping:
        damping_t = base_damping
    else:
        damping_t = max(1.0 - step * beta_scale, beta_max)

    return float(max(0.0, min(1.0, damping_t)))


def prepare_global_shared_data(
    device: torch.device,
    seed: int = 1,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    noise_var: float = 1e-10,
    lam: float = 1.0,
) -> dict[str, torch.Tensor | float | int | str]:
    """
    Prepare teacher matrices and a full-grid noise field once for the
    whole simulation.
    """
    scale = lam / math.sqrt(M)

    torch.manual_seed(seed)
    W_teacher = torch.randn(N1, M, device=device, dtype=torch.float32)
    X_teacher = torch.randn(M, N2, device=device, dtype=torch.float32)
    teacher_w_t = W_teacher.T.contiguous()
    teacher_x_t = X_teacher.T.contiguous()
    teacher_w_gram = teacher_w_t @ W_teacher
    teacher_x_gram = X_teacher @ teacher_x_t
    teacher_norm_sq = torch.sum(teacher_w_gram * teacher_x_gram.T)

    torch.manual_seed(seed)
    noise_full = torch.randn((N1, N2), device=device, dtype=torch.float32)
    noise_full = noise_full * math.sqrt(noise_var)

    return {
        "seed": seed,
        "scale": scale,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "teacher_w_t": teacher_w_t,
        "teacher_x_t": teacher_x_t,
        "teacher_norm_sq": teacher_norm_sq,
        "noise_full": noise_full,
    }


def build_shared_alpha_data(
    alpha: float,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    E: int,
    global_data: dict[str, torch.Tensor | float | int | str],
    metadata: dict[str, torch.Tensor | float | int | str] | None = None,
) -> dict[str, torch.Tensor | float | int | str]:
    """
    Assemble the graph-dependent shared data dict from observed edges.
    """
    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    teacher_w_t = global_data["teacher_w_t"]
    teacher_x_t = global_data["teacher_x_t"]
    teacher_norm_sq = global_data["teacher_norm_sq"]
    noise_full = global_data["noise_full"]
    scale = float(global_data["scale"])

    shared_data: dict[str, torch.Tensor | float | int | str] = {
        "alpha": alpha,
        "E": E,
        "scale": scale,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "teacher_w_t": teacher_w_t,
        "teacher_x_t": teacher_x_t,
        "teacher_norm_sq": teacher_norm_sq,
        "Y_clean_obs": torch.empty(0, dtype=torch.float32, device=W_teacher.device),
        "Y_noisy_obs": torch.empty(0, dtype=torch.float32, device=W_teacher.device),
    }
    if metadata:
        shared_data.update(metadata)

    if E == 0:
        return shared_data

    i_long = i_idx.long()
    j_long = j_idx.long()
    W_sel = W_teacher[i_long, :]
    X_sel = X_teacher[:, j_long].T
    y_clean_obs = scale * (W_sel * X_sel).sum(dim=1)
    y_noisy_obs = y_clean_obs + noise_full[i_long, j_long]

    shared_data["Y_clean_obs"] = y_clean_obs
    shared_data["Y_noisy_obs"] = y_noisy_obs
    return shared_data


def _initialize_student_factors(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    seed: int,
    init_epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize student factors.

    When init_epsilon is provided:
        student = epsilon * teacher + sqrt(epsilon - epsilon^2) * N(0, 1).
    """
    torch.manual_seed(seed + 2000)

    if init_epsilon is None:
        m_W = torch.randn_like(W_teacher)
        m_X = torch.randn_like(X_teacher)
    else:
        m_W = initialize_correlated_student(W_teacher, init_epsilon)
        m_X = initialize_correlated_student(X_teacher, init_epsilon)

    v_W = torch.ones_like(m_W)
    v_X = torch.ones_like(m_X)

    return m_W, v_W, m_X, v_X


def train_single_replica_from_shared_data(
    device: torch.device,
    seed: int,
    max_steps: int = 500,
    damping: float = 0.5,
    use_step_damping: bool = False,
    damping_beta_scale: float = 1e-3,
    damping_beta_max: float = 0.5,
    noise_var: float = 1e-10,
    convergence_threshold: float = 1e-6,
    lam: float = 1.0,
    return_history: bool = False,
    loss_eval_interval: int = 50,
    early_stop: bool = True,
    init_epsilon: float | None = None,
    shared_data: dict[str, torch.Tensor | float | int | str] | None = None,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    """
    Train a single replica using alternating edge-observed G-AMP with prebuilt
    shared graph/observation data.
    """
    if shared_data is None:
        raise ValueError("shared_data must be provided.")

    E = int(shared_data["E"])
    if E == 0:
        return 0.0, 0.0, 0

    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    W_teacher = shared_data["W_teacher"]
    X_teacher = shared_data["X_teacher"]
    teacher_w_t = shared_data["teacher_w_t"]
    teacher_x_t = shared_data["teacher_x_t"]
    teacher_norm_sq = shared_data["teacher_norm_sq"]
    Y_noisy = shared_data["Y_noisy_obs"]
    Y_clean = shared_data["Y_clean_obs"]
    scale = float(shared_data["scale"])
    F_edge = shared_data.get("F_edge")
    _, M = W_teacher.shape

    m_W, v_W, m_X, v_X = _initialize_student_factors(
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        seed=seed,
        init_epsilon=init_epsilon,
    )
    g_prev_edge = torch.zeros(E, device=device)

    m_W_prev = m_W.clone()
    m_X_prev = m_X.clone()

    final_loss = 0.0
    steps_taken = max_steps
    prev_loss = float("inf")
    history = {"steps": [], "loss": [], "cosine_similarity": [], "damping": []}
    history_loss_tensors = []
    history_cosine_values = []

    for step in range(max_steps):
        damping_t = compute_step_damping(
            step=step,
            base_damping=damping,
            use_step_damping=use_step_damping,
            beta_scale=damping_beta_scale,
            beta_max=damping_beta_max,
        )

        m_W_before_W = m_W
        m_X_before_W = m_X
        m_W, v_W, g_prev_edge = alternating_half_step_W(
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            m_W_prev=m_W_prev,
            m_X_prev=m_X_prev,
            Y_obs=Y_noisy,
            i_idx=i_idx,
            j_idx=j_idx,
            g_prev_edge=g_prev_edge,
            lam=lam,
            noise_var=noise_var,
            damping=damping_t,
            M=M,
            F_edge=F_edge,
        )

        m_W_prev = m_W_before_W
        m_X_prev = m_X_before_W

        m_W_before_X = m_W
        m_X_before_X = m_X
        m_X, v_X, g_prev_edge = alternating_half_step_X(
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            m_W_prev=m_W_prev,
            m_X_prev=m_X_prev,
            Y_obs=Y_noisy,
            i_idx=i_idx,
            j_idx=j_idx,
            g_prev_edge=g_prev_edge,
            lam=lam,
            noise_var=noise_var,
            damping=damping_t,
            M=M,
            F_edge=F_edge,
        )

        m_W_prev = m_W_before_X
        m_X_prev = m_X_before_X

        if step % loss_eval_interval == 0 or step == max_steps - 1:
            m_W_eval = normalize_to_unit_variance(m_W)
            m_X_eval = normalize_to_unit_variance(m_X)
            loss_tensor = compute_observed_loss(
                m_W_eval,
                m_X_eval,
                Y_noisy,
                i_idx,
                j_idx,
                scale,
                F_edge=F_edge,
            )

            if return_history:
                if F_edge is None:
                    cosine_similarity_step = compute_y_cosine_similarity_tensor(
                        W_student=m_W_eval,
                        X_student=m_X_eval,
                        teacher_w_t=teacher_w_t,
                        teacher_x_t=teacher_x_t,
                        teacher_norm_sq=teacher_norm_sq,
                    )
                else:
                    cosine_similarity_step = compute_observed_signal_cosine_tensor(
                        W_student=m_W_eval,
                        X_student=m_X_eval,
                        Y_clean_obs=Y_clean,
                        i_idx=i_idx,
                        j_idx=j_idx,
                        scale=scale,
                        F_edge=F_edge,
                    )

                history["steps"].append(step + 1)
                history_loss_tensors.append(loss_tensor.detach())
                history_cosine_values.append(float(cosine_similarity_step.item()))
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

    m_W = normalize_to_unit_variance(m_W)
    m_X = normalize_to_unit_variance(m_X)

    if F_edge is None:
        cosine_similarity = compute_y_cosine_similarity_tensor(
            W_student=m_W,
            X_student=m_X,
            teacher_w_t=teacher_w_t,
            teacher_x_t=teacher_x_t,
            teacher_norm_sq=teacher_norm_sq,
        )
    else:
        cosine_similarity = compute_observed_signal_cosine_tensor(
            W_student=m_W,
            X_student=m_X,
            Y_clean_obs=Y_clean,
            i_idx=i_idx,
            j_idx=j_idx,
            scale=scale,
            F_edge=F_edge,
        )

    if return_history:
        if history_loss_tensors:
            history["loss"] = torch.stack(history_loss_tensors).cpu().tolist()
            history["cosine_similarity"] = history_cosine_values
            if not early_stop:
                final_loss = float(history["loss"][-1])
        return float(cosine_similarity.item()), final_loss, steps_taken, history

    return float(cosine_similarity.item()), final_loss, steps_taken


__all__ = [
    "alternating_half_step_W",
    "alternating_half_step_X",
    "build_shared_alpha_data",
    "compute_observed_loss",
    "compute_observed_signal_cosine_tensor",
    "compute_step_damping",
    "compute_y_cosine_similarity",
    "compute_y_cosine_similarity_tensor",
    "prepare_global_shared_data",
    "train_single_replica_from_shared_data",
]
