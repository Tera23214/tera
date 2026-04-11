#!/usr/bin/env python
"""
Dense-mask G-AMP core module with F=1, exact Onsager correction, and
alternating W -> X updates with cosine-similarity evaluation.

This variant keeps the dense-mask observation process from ``Dence_cosine``
but replaces the simultaneous update with an alternating scheme:

1. Update W using the current state (W(t), X(t))
2. Recompute the output-channel variables with (W(t+1), X(t))
3. Update X

The Onsager memory and ``g_prev_dense`` are advanced at every half-step.
One recorded step therefore means one full W -> X sweep.
"""

import math
import sys
from pathlib import Path

import torch

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.graph import BiregularGraph
from terao_gamp_gaussian.utils import f_input, normalize_to_unit_variance


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
    Y_full: torch.Tensor,
    mask: torch.Tensor,
    g_prev_dense: torch.Tensor,
    lam: float,
    noise_var: float,
    damping: float,
    M: int,
) -> tuple[
    float,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute output-channel quantities for one half-step.

    The current state is the input to the half-step and ``m_W_prev, m_X_prev``
    correspond to the immediately preceding half-step state.
    """
    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    var_term_W = torch.clamp(v_W - m_W ** 2, min=0.0)
    var_term_X = torch.clamp(v_X - m_X ** 2, min=0.0)

    omega_main = scale * (m_W @ m_X)
    onsager_W_side = scale_sq * ((m_W * m_W_prev) @ var_term_X)
    onsager_X_side = scale_sq * (var_term_W @ (m_X * m_X_prev))
    omega = omega_main - g_prev_dense * (onsager_W_side + onsager_X_side)

    V = scale_sq * (v_W @ v_X - (m_W ** 2) @ (m_X ** 2))
    V = torch.clamp(V, min=1e-10)

    denom = V + noise_var
    g_raw = mask * (Y_full - omega) / denom
    dg = -mask / denom

    g_dense = damping * g_prev_dense + (1.0 - damping) * g_raw
    g_dense = mask * torch.clamp(g_dense, min=-100.0, max=100.0)

    g_pair = mask * g_dense * g_prev_dense
    onsager_W = scale_sq * (g_pair @ var_term_X.T)
    onsager_X = scale_sq * (var_term_W.T @ g_pair)

    return scale, scale_sq, dg, g_dense, onsager_W, onsager_X


def alternating_half_step_W(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y_full: torch.Tensor,
    mask: torch.Tensor,
    g_prev_dense: torch.Tensor,
    lam: float,
    noise_var: float,
    damping: float,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update W while freezing X.
    """
    scale, scale_sq, dg, g_dense, onsager_W, _ = _compute_output_channel_terms(
        m_W=m_W,
        v_W=v_W,
        m_X=m_X,
        v_X=v_X,
        m_W_prev=m_W_prev,
        m_X_prev=m_X_prev,
        Y_full=Y_full,
        mask=mask,
        g_prev_dense=g_prev_dense,
        lam=lam,
        noise_var=noise_var,
        damping=damping,
        M=M,
    )

    Sigma_W_denom = scale_sq * ((-dg) @ (m_X ** 2).T)
    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    sum_W = scale * (g_dense @ m_X.T)
    T_W = m_W + Sigma_W * (sum_W - onsager_W * m_W_prev)

    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=100.0)

    m_W_new = damping * m_W + (1.0 - damping) * m_W_new
    v_W_new = damping * v_W + (1.0 - damping) * v_W_new

    return m_W_new, v_W_new, g_dense


def alternating_half_step_X(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y_full: torch.Tensor,
    mask: torch.Tensor,
    g_prev_dense: torch.Tensor,
    lam: float,
    noise_var: float,
    damping: float,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update X while freezing W.
    """
    scale, scale_sq, dg, g_dense, _, onsager_X = _compute_output_channel_terms(
        m_W=m_W,
        v_W=v_W,
        m_X=m_X,
        v_X=v_X,
        m_W_prev=m_W_prev,
        m_X_prev=m_X_prev,
        Y_full=Y_full,
        mask=mask,
        g_prev_dense=g_prev_dense,
        lam=lam,
        noise_var=noise_var,
        damping=damping,
        M=M,
    )

    Sigma_X_denom = scale_sq * ((m_W ** 2).T @ (-dg))
    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    sum_X = scale * (m_W.T @ g_dense)
    T_X = m_X + Sigma_X * (sum_X - onsager_X * m_X_prev)

    m_X_new, v_X_new = f_input(Sigma_X, T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=100.0)

    m_X_new = damping * m_X + (1.0 - damping) * m_X_new
    v_X_new = damping * v_X + (1.0 - damping) * v_X_new

    return m_X_new, v_X_new, g_dense


def compute_observed_loss(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_obs_full: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Compute MSE on observed entries using the dense observation mask.
    """
    num_observed = mask.sum().clamp_min(1.0)
    Y_pred = scale * (m_W @ m_X)

    return (mask * (Y_obs_full - Y_pred) ** 2).sum() / num_observed


def compute_step_damping(
    step: int,
    base_damping: float,
    use_step_damping: bool,
    beta_scale: float,
    beta_max: float,
) -> float:
    """
    Compute damping factor for the current outer step.

    One step means one full W -> X sweep.
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
) -> dict[str, torch.Tensor | float | int]:
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


def prepare_shared_alpha_data(
    alpha: float,
    device: torch.device,
    seed: int = 1,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    noise_var: float = 1e-10,
    lam: float = 1.0,
    global_data: dict[str, torch.Tensor | float | int] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """
    Prepare graph and observations once for a single alpha.
    """
    if global_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=seed,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
            lam=lam,
        )

    graph = BiregularGraph()
    mask, i_idx, j_idx, E, C1, C2, alpha2 = graph.generate_dense_mask(
        N1=N1,
        N2=N2,
        M=M,
        alpha1=alpha,
        device=device,
        seed=seed,
    )

    scale = float(global_data["scale"])
    y_clean_full = torch.zeros((N1, N2), dtype=torch.float32, device=device)
    y_noisy_full = torch.zeros((N1, N2), dtype=torch.float32, device=device)
    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    teacher_w_t = global_data["teacher_w_t"]
    teacher_x_t = global_data["teacher_x_t"]
    teacher_norm_sq = global_data["teacher_norm_sq"]
    noise_full = global_data["noise_full"]

    if E == 0:
        return {
            "alpha": alpha,
            "alpha2": alpha2,
            "E": E,
            "C1": C1,
            "C2": C2,
            "scale": scale,
            "mask": mask,
            "i_idx": i_idx,
            "j_idx": j_idx,
            "W_teacher": W_teacher,
            "X_teacher": X_teacher,
            "teacher_w_t": teacher_w_t,
            "teacher_x_t": teacher_x_t,
            "teacher_norm_sq": teacher_norm_sq,
            "Y_clean_full": y_clean_full,
            "Y_noisy_full": y_noisy_full,
        }

    y_clean_full = mask * (scale * (W_teacher @ X_teacher))
    y_noisy_full = mask * (scale * (W_teacher @ X_teacher) + noise_full)

    return {
        "alpha": alpha,
        "alpha2": alpha2,
        "E": E,
        "C1": C1,
        "C2": C2,
        "scale": scale,
        "mask": mask,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "teacher_w_t": teacher_w_t,
        "teacher_x_t": teacher_x_t,
        "teacher_norm_sq": teacher_norm_sq,
        "Y_clean_full": y_clean_full,
        "Y_noisy_full": y_noisy_full,
    }


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
    lam: float = 1.0,
    return_history: bool = False,
    loss_eval_interval: int = 50,
    early_stop: bool = True,
    shared_data: dict[str, torch.Tensor | float | int] | None = None,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    """
    Train a single replica using alternating dense-mask G-AMP.

    One recorded step means:
    1. W update with X frozen
    2. X update with the new W and recomputed output-channel quantities
    """
    if shared_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=1,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
            lam=lam,
        )
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=1,
            N1=N1,
            N2=N2,
            M=M,
            noise_var=noise_var,
            lam=lam,
            global_data=global_data,
        )

    E = int(shared_data["E"])
    if E == 0:
        return 0.0, 0.0, 0

    mask = shared_data["mask"]
    W_teacher = shared_data["W_teacher"]
    X_teacher = shared_data["X_teacher"]
    teacher_w_t = shared_data["teacher_w_t"]
    teacher_x_t = shared_data["teacher_x_t"]
    teacher_norm_sq = shared_data["teacher_norm_sq"]
    Y_noisy = shared_data["Y_noisy_full"]
    scale = float(shared_data["scale"])
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]

    torch.manual_seed(seed + 2000)
    m_W = torch.randn(N1, M, device=device)
    v_W = torch.ones(N1, M, device=device)
    m_X = torch.randn(M, N2, device=device)
    v_X = torch.ones(M, N2, device=device)
    g_prev_dense = torch.zeros((N1, N2), device=device)

    # The first half-step uses the current state itself as the previous state,
    # mirroring the initialization convention in the simultaneous-update code.
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

        # Half-step 1: update W using the current state and the previous
        # half-step memory.
        m_W_before_W = m_W
        m_X_before_W = m_X
        m_W, v_W, g_prev_dense = alternating_half_step_W(
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            m_W_prev=m_W_prev,
            m_X_prev=m_X_prev,
            Y_full=Y_noisy,
            mask=mask,
            g_prev_dense=g_prev_dense,
            lam=lam,
            noise_var=noise_var,
            damping=damping_t,
            M=M,
        )

        # Advance memory by one half-step.
        m_W_prev = m_W_before_W
        m_X_prev = m_X_before_W

        # Half-step 2: update X using the new W and the frozen X from the
        # start of the step. The output-channel quantities are recomputed.
        m_W_before_X = m_W
        m_X_before_X = m_X
        m_X, v_X, g_prev_dense = alternating_half_step_X(
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            m_W_prev=m_W_prev,
            m_X_prev=m_X_prev,
            Y_full=Y_noisy,
            mask=mask,
            g_prev_dense=g_prev_dense,
            lam=lam,
            noise_var=noise_var,
            damping=damping_t,
            M=M,
        )

        # Advance memory to the state right before the X update so the next
        # W half-step sees the immediately preceding half-step state.
        m_W_prev = m_W_before_X
        m_X_prev = m_X_before_X

        if step % loss_eval_interval == 0 or step == max_steps - 1:
            m_W_eval = normalize_to_unit_variance(m_W)
            m_X_eval = normalize_to_unit_variance(m_X)
            loss_tensor = compute_observed_loss(
                m_W_eval, m_X_eval, Y_noisy, mask, scale
            )

            if return_history:
                cosine_similarity_step = compute_y_cosine_similarity_tensor(
                    W_student=m_W_eval,
                    X_student=m_X_eval,
                    teacher_w_t=teacher_w_t,
                    teacher_x_t=teacher_x_t,
                    teacher_norm_sq=teacher_norm_sq,
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

    cosine_similarity = float(
        compute_y_cosine_similarity_tensor(
            W_student=m_W,
            X_student=m_X,
            teacher_w_t=teacher_w_t,
            teacher_x_t=teacher_x_t,
            teacher_norm_sq=teacher_norm_sq,
        ).item()
    )

    if return_history:
        if history_loss_tensors:
            history["loss"] = torch.stack(history_loss_tensors).cpu().tolist()
            history["cosine_similarity"] = history_cosine_values
            if not early_stop:
                final_loss = float(history["loss"][-1])
        return cosine_similarity, final_loss, steps_taken, history

    return cosine_similarity, final_loss, steps_taken
