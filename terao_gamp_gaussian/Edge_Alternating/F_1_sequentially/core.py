#!/usr/bin/env python
"""
Sequentially aggregated F=1 version of Edge_Alternating.

This variant keeps the memory-saving edge-chunk aggregation, but uses the
constant spreading coefficient F=1 on every observed edge and component.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Edge_Alternating.shared_core import (  # noqa: E402
    compute_step_damping,
    prepare_global_shared_data as _prepare_global_shared_data,
)
from terao_gamp_gaussian.graph import BiregularGraph  # noqa: E402
from terao_gamp_gaussian.utils import (  # noqa: E402
    f_input,
    g_out,
    initialize_correlated_student,
    normalize_to_unit_variance,
)

DEFAULT_EDGE_CHUNK_SIZE = 4096


def _resolve_edge_chunk_size(edge_chunk_size: int | None, E: int) -> int:
    if E <= 0:
        return 1
    if edge_chunk_size is None or edge_chunk_size <= 0:
        return min(E, DEFAULT_EDGE_CHUNK_SIZE)
    return min(E, int(edge_chunk_size))


def _edge_ranges(E: int, edge_chunk_size: int):
    for chunk_id, start in enumerate(range(0, E, edge_chunk_size)):
        yield chunk_id, start, min(start + edge_chunk_size, E)


def prepare_global_shared_data(
    device: torch.device,
    seed: int = 1,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    noise_var: float = 1e-10,
    lam: float = 1.0,
) -> dict[str, torch.Tensor | float | int | str]:
    return _prepare_global_shared_data(
        device=device,
        seed=seed,
        N1=N1,
        N2=N2,
        M=M,
        noise_var=noise_var,
        lam=lam,
    )


def prepare_shared_alpha_data(
    alpha: float,
    device: torch.device,
    seed: int = 1,
    N1: int = 1000,
    N2: int = 1000,
    M: int = 10,
    noise_var: float = 1e-10,
    lam: float = 1.0,
    edge_chunk_size: int | None = DEFAULT_EDGE_CHUNK_SIZE,
    global_data: dict[str, torch.Tensor | float | int | str] | None = None,
) -> dict[str, torch.Tensor | float | int | str]:
    """
    Prepare graph and observations for fixed F=1 without storing E x M
    intermediate tensors.
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
    i_idx, j_idx, E, C1, C2, alpha2 = graph.generate(
        N1=N1,
        N2=N2,
        M=M,
        alpha1=alpha,
        device=device,
        seed=seed,
    )

    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    teacher_w_t = global_data["teacher_w_t"]
    teacher_x_t = global_data["teacher_x_t"]
    teacher_norm_sq = global_data["teacher_norm_sq"]
    noise_full = global_data["noise_full"]
    scale = float(global_data["scale"])
    chunk_size = _resolve_edge_chunk_size(edge_chunk_size, E)

    y_clean_obs = torch.empty(E, dtype=torch.float32, device=device)
    y_noisy_obs = torch.empty(E, dtype=torch.float32, device=device)

    for _chunk_id, start, end in _edge_ranges(E, chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        W_sel = W_teacher[i_chunk, :]
        X_sel = X_teacher[:, j_chunk].T
        y_clean = scale * (W_sel * X_sel).sum(dim=1)
        y_clean_obs[start:end] = y_clean
        y_noisy_obs[start:end] = y_clean + noise_full[i_chunk, j_chunk]

    return {
        "alpha": alpha,
        "alpha2": alpha2,
        "C1": C1,
        "C2": C2,
        "E": E,
        "scale": scale,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "X_teacher": X_teacher,
        "teacher_w_t": teacher_w_t,
        "teacher_x_t": teacher_x_t,
        "teacher_norm_sq": teacher_norm_sq,
        "Y_clean_obs": y_clean_obs,
        "Y_noisy_obs": y_noisy_obs,
        "graph_model": "random_graph",
        "f_mode": "fixed",
        "f_distribution": "constant_one",
        "f_value": 1.0,
        "edge_chunk_size": chunk_size,
        "sequential_aggregation": True,
    }


def _initialize_student_factors(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    seed: int,
    init_epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + 2000)

    if init_epsilon is None:
        m_W = torch.randn_like(W_teacher)
        m_X = torch.randn_like(X_teacher)
    else:
        m_W = initialize_correlated_student(W_teacher, init_epsilon)
        m_X = initialize_correlated_student(X_teacher, init_epsilon)

    if init_epsilon is not None and math.isclose(
        float(init_epsilon), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        v_W = m_W ** 2
        v_X = m_X ** 2
    else:
        v_W = torch.ones_like(m_W)
        v_X = torch.ones_like(m_X)
    return m_W, v_W, m_X, v_X


def _compute_chunk_output_terms(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    m_W_prev: torch.Tensor,
    m_X_prev: torch.Tensor,
    Y_obs: torch.Tensor,
    i_chunk: torch.Tensor,
    j_chunk: torch.Tensor,
    g_prev_chunk: torch.Tensor,
    scale: float,
    scale_sq: float,
    noise_var: float,
    damping: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    W_sel = m_W[i_chunk, :]
    X_sel = m_X[:, j_chunk].T
    vW_sel = v_W[i_chunk, :]
    vX_sel = v_X[:, j_chunk].T
    W_prev_sel = m_W_prev[i_chunk, :]
    X_prev_sel = m_X_prev[:, j_chunk].T

    var_term_W = torch.clamp(vW_sel - W_sel ** 2, min=0.0)
    var_term_X = torch.clamp(vX_sel - X_sel ** 2, min=0.0)

    omega_main = scale * (W_sel * X_sel).sum(dim=1)
    onsager_W_side = scale_sq * (var_term_X * W_sel * W_prev_sel).sum(dim=1)
    onsager_X_side = scale_sq * (var_term_W * X_sel * X_prev_sel).sum(dim=1)
    V = scale_sq * (vW_sel * vX_sel - (W_sel ** 2) * (X_sel ** 2)).sum(dim=1)
    V = torch.clamp(V, min=1e-10)

    omega = omega_main - g_prev_chunk * (onsager_W_side + onsager_X_side)
    g_raw, dg = g_out(omega, Y_obs, V, noise_var)
    g_edge = damping * g_prev_chunk + (1.0 - damping) * g_raw
    g_edge = torch.clamp(g_edge, min=-100.0, max=100.0)

    return g_edge, dg, var_term_W, var_term_X, W_sel, X_sel, V, omega


def alternating_half_step_W_sequential(
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
    edge_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E = int(i_idx.numel())
    device = m_W.device
    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    Sigma_W_denom = torch.zeros_like(m_W)
    sum_W = torch.zeros_like(m_W)
    onsager_W = torch.zeros_like(m_W)
    g_next_edge = torch.empty_like(g_prev_edge)

    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        g_edge, dg, _, var_term_X, _, X_sel, _, _ = _compute_chunk_output_terms(
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            m_W_prev=m_W_prev,
            m_X_prev=m_X_prev,
            Y_obs=Y_obs[start:end],
            i_chunk=i_chunk,
            j_chunk=j_chunk,
            g_prev_chunk=g_prev_edge[start:end],
            scale=scale,
            scale_sq=scale_sq,
            noise_var=noise_var,
            damping=damping,
        )
        g_next_edge[start:end] = g_edge
        g_pair = g_edge * g_prev_edge[start:end]

        Sigma_W_denom.scatter_add_(
            0,
            i_chunk.unsqueeze(1).expand(-1, M),
            scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2),
        )
        sum_W.scatter_add_(
            0,
            i_chunk.unsqueeze(1).expand(-1, M),
            scale * g_edge.unsqueeze(1) * X_sel,
        )
        onsager_W.scatter_add_(
            0,
            i_chunk.unsqueeze(1).expand(-1, M),
            scale_sq * g_pair.unsqueeze(1) * var_term_X,
        )

    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    T_W = m_W + Sigma_W * (sum_W - onsager_W * m_W_prev)
    m_W_new, v_W_new = f_input(Sigma_W, T_W)
    v_W_new = torch.clamp(v_W_new, min=1e-8, max=100.0)
    m_W_new = damping * m_W + (1.0 - damping) * m_W_new
    v_W_new = damping * v_W + (1.0 - damping) * v_W_new
    return m_W_new, v_W_new, g_next_edge


def alternating_half_step_X_sequential(
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
    edge_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E = int(i_idx.numel())
    device = m_W.device
    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    Sigma_X_denom = torch.zeros_like(m_X)
    sum_X = torch.zeros_like(m_X)
    onsager_X = torch.zeros_like(m_X)
    g_next_edge = torch.empty_like(g_prev_edge)

    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        g_edge, dg, var_term_W, _, W_sel, _, _, _ = _compute_chunk_output_terms(
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            m_W_prev=m_W_prev,
            m_X_prev=m_X_prev,
            Y_obs=Y_obs[start:end],
            i_chunk=i_chunk,
            j_chunk=j_chunk,
            g_prev_chunk=g_prev_edge[start:end],
            scale=scale,
            scale_sq=scale_sq,
            noise_var=noise_var,
            damping=damping,
        )
        g_next_edge[start:end] = g_edge
        g_pair = g_edge * g_prev_edge[start:end]

        Sigma_X_denom.scatter_add_(
            1,
            j_chunk.unsqueeze(0).expand(M, -1),
            (scale_sq * (-dg).unsqueeze(1) * (W_sel ** 2)).T.contiguous(),
        )
        sum_X.scatter_add_(
            1,
            j_chunk.unsqueeze(0).expand(M, -1),
            (scale * g_edge.unsqueeze(1) * W_sel).T.contiguous(),
        )
        onsager_X.scatter_add_(
            1,
            j_chunk.unsqueeze(0).expand(M, -1),
            (scale_sq * g_pair.unsqueeze(1) * var_term_W).T.contiguous(),
        )

    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    T_X = m_X + Sigma_X * (sum_X - onsager_X * m_X_prev)
    m_X_new, v_X_new = f_input(Sigma_X, T_X)
    v_X_new = torch.clamp(v_X_new, min=1e-8, max=100.0)
    m_X_new = damping * m_X + (1.0 - damping) * m_X_new
    v_X_new = damping * v_X + (1.0 - damping) * v_X_new
    return m_X_new, v_X_new, g_next_edge


def compute_observed_loss(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
    edge_chunk_size: int,
) -> torch.Tensor:
    E = int(i_idx.numel())
    if E == 0:
        return torch.zeros((), dtype=m_W.dtype, device=m_W.device)

    _, M = m_W.shape
    loss_sum = torch.zeros((), dtype=m_W.dtype, device=m_W.device)
    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        W_sel = m_W[i_chunk, :]
        X_sel = m_X[:, j_chunk].T
        Y_pred = scale * (W_sel * X_sel).sum(dim=1)
        loss_sum = loss_sum + ((Y_obs[start:end] - Y_pred) ** 2).sum()

    return loss_sum / E


def compute_observed_signal_cosine(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_clean_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
    edge_chunk_size: int,
) -> torch.Tensor:
    E = int(i_idx.numel())
    if E == 0:
        return torch.zeros((), dtype=m_W.dtype, device=m_W.device)

    _, M = m_W.shape
    inner = torch.zeros((), dtype=m_W.dtype, device=m_W.device)
    teacher_norm_sq = torch.zeros_like(inner)
    student_norm_sq = torch.zeros_like(inner)

    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        W_sel = m_W[i_chunk, :]
        X_sel = m_X[:, j_chunk].T
        Y_student = scale * (W_sel * X_sel).sum(dim=1)
        Y_teacher = Y_clean_obs[start:end]
        inner = inner + torch.sum(Y_teacher * Y_student)
        teacher_norm_sq = teacher_norm_sq + torch.sum(Y_teacher ** 2)
        student_norm_sq = student_norm_sq + torch.sum(Y_student ** 2)

    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))
    return inner / denom


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
    if shared_data is None:
        raise ValueError("shared_data must be provided.")

    E = int(shared_data["E"])
    if E == 0:
        return 0.0, 0.0, 0

    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    W_teacher = shared_data["W_teacher"]
    X_teacher = shared_data["X_teacher"]
    Y_noisy = shared_data["Y_noisy_obs"]
    Y_clean = shared_data["Y_clean_obs"]
    scale = float(shared_data["scale"])
    edge_chunk_size = int(shared_data["edge_chunk_size"])
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

    if return_history:
        m_W_eval = normalize_to_unit_variance(m_W)
        m_X_eval = normalize_to_unit_variance(m_X)
        loss_tensor = compute_observed_loss(
            m_W=m_W_eval,
            m_X=m_X_eval,
            Y_obs=Y_noisy,
            i_idx=i_idx,
            j_idx=j_idx,
            scale=scale,
            edge_chunk_size=edge_chunk_size,
        )
        cosine_similarity_step = compute_observed_signal_cosine(
            m_W=m_W_eval,
            m_X=m_X_eval,
            Y_clean_obs=Y_clean,
            i_idx=i_idx,
            j_idx=j_idx,
            scale=scale,
            edge_chunk_size=edge_chunk_size,
        )
        history["steps"].append(0)
        history_loss_tensors.append(loss_tensor.detach())
        history_cosine_values.append(float(cosine_similarity_step.item()))
        history["damping"].append(0.0)

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
        m_W, v_W, g_prev_edge = alternating_half_step_W_sequential(
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
            edge_chunk_size=edge_chunk_size,
        )

        m_W_prev = m_W_before_W
        m_X_prev = m_X_before_W

        m_W_before_X = m_W
        m_X_before_X = m_X
        m_X, v_X, g_prev_edge = alternating_half_step_X_sequential(
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
            edge_chunk_size=edge_chunk_size,
        )

        m_W_prev = m_W_before_X
        m_X_prev = m_X_before_X

        if step % loss_eval_interval == 0 or step == max_steps - 1:
            m_W_eval = normalize_to_unit_variance(m_W)
            m_X_eval = normalize_to_unit_variance(m_X)
            loss_tensor = compute_observed_loss(
                m_W=m_W_eval,
                m_X=m_X_eval,
                Y_obs=Y_noisy,
                i_idx=i_idx,
                j_idx=j_idx,
                scale=scale,
                edge_chunk_size=edge_chunk_size,
            )

            if return_history:
                cosine_similarity_step = compute_observed_signal_cosine(
                    m_W=m_W_eval,
                    m_X=m_X_eval,
                    Y_clean_obs=Y_clean,
                    i_idx=i_idx,
                    j_idx=j_idx,
                    scale=scale,
                    edge_chunk_size=edge_chunk_size,
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
    cosine_similarity = compute_observed_signal_cosine(
        m_W=m_W,
        m_X=m_X,
        Y_clean_obs=Y_clean,
        i_idx=i_idx,
        j_idx=j_idx,
        scale=scale,
        edge_chunk_size=edge_chunk_size,
    )

    if return_history:
        if history_loss_tensors:
            history["loss"] = torch.stack(history_loss_tensors).cpu().tolist()
            history["cosine_similarity"] = history_cosine_values
            if not early_stop:
                final_loss = float(history["loss"][-1])
        return float(cosine_similarity.item()), final_loss, steps_taken, history

    return float(cosine_similarity.item()), final_loss, steps_taken


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
    init_epsilon: float | None = None,
    edge_chunk_size: int | None = DEFAULT_EDGE_CHUNK_SIZE,
    shared_data: dict[str, torch.Tensor | float | int | str] | None = None,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
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
            edge_chunk_size=edge_chunk_size,
            global_data=global_data,
        )

    return train_single_replica_from_shared_data(
        device=device,
        seed=seed,
        max_steps=max_steps,
        damping=damping,
        use_step_damping=use_step_damping,
        damping_beta_scale=damping_beta_scale,
        damping_beta_max=damping_beta_max,
        noise_var=noise_var,
        convergence_threshold=convergence_threshold,
        lam=lam,
        return_history=return_history,
        loss_eval_interval=loss_eval_interval,
        early_stop=early_stop,
        init_epsilon=init_epsilon,
        shared_data=shared_data,
    )
