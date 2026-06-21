#!/usr/bin/env python
"""
Sequentially aggregated random-F version of Edge_Alternating.

This variant does not materialize the full ``F_edge`` tensor.  It regenerates
Rademacher chunks of F deterministically and immediately accumulates their
contributions into node-wise sufficient statistics.
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


def _reset_f_stream(f_seed: int) -> None:
    torch.manual_seed(int(f_seed))


def _make_f_chunk(
    num_edges: int,
    M: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Draw the next Rademacher F chunk from the current torch RNG stream.
    """
    F_chunk = torch.empty((num_edges, M), device=device, dtype=dtype)
    F_chunk.bernoulli_(0.5).mul_(2.0).sub_(1.0)
    return F_chunk


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
    Prepare graph and observations without storing the full E x M random F.
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
    f_seed = int(seed) + 1000
    chunk_size = _resolve_edge_chunk_size(edge_chunk_size, E)

    y_clean_obs = torch.empty(E, dtype=torch.float32, device=device)
    y_noisy_obs = torch.empty(E, dtype=torch.float32, device=device)

    _reset_f_stream(f_seed)
    for _chunk_id, start, end in _edge_ranges(E, chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        F_chunk = _make_f_chunk(
            num_edges=end - start,
            M=M,
            device=device,
            dtype=torch.float32,
        )
        W_sel = W_teacher[i_chunk, :]
        X_sel = X_teacher[:, j_chunk].T
        y_clean = scale * (W_sel * F_chunk * X_sel).sum(dim=1)
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
        "f_mode": "random",
        "f_distribution": "rademacher_pm1",
        "f_seed": f_seed,
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
    F_chunk: torch.Tensor,
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

    omega_main = scale * (W_sel * F_chunk * X_sel).sum(dim=1)
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
    f_seed: int,
    edge_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    E = int(i_idx.numel())
    device = m_W.device
    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    Sigma_W_denom = torch.zeros_like(m_W)
    sum_W = torch.zeros_like(m_W)
    onsager_W = torch.zeros_like(m_W)
    g_next_edge = torch.empty_like(g_prev_edge)

    _reset_f_stream(f_seed)
    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        F_chunk = _make_f_chunk(end - start, M, device, m_W.dtype)
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
            F_chunk=F_chunk,
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
            scale * g_edge.unsqueeze(1) * F_chunk * X_sel,
        )
        onsager_W.scatter_add_(
            0,
            i_chunk.unsqueeze(1).expand(-1, M),
            scale_sq * g_pair.unsqueeze(1) * var_term_X,
        )

    Sigma_W = 1.0 / torch.clamp(Sigma_W_denom, min=1e-10)
    T_W = m_W + Sigma_W * (sum_W - onsager_W * m_W_prev)
    m_W_proposal, v_W_proposal = f_input(Sigma_W, T_W)
    convergence_sum = torch.sum(torch.abs(m_W_proposal - m_W))
    v_W_new = torch.clamp(v_W_proposal, min=1e-8, max=100.0)
    m_W_new = damping * m_W + (1.0 - damping) * m_W_proposal
    v_W_new = damping * v_W + (1.0 - damping) * v_W_new
    return m_W_new, v_W_new, g_next_edge, convergence_sum


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
    f_seed: int,
    edge_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    E = int(i_idx.numel())
    device = m_W.device
    scale = lam / math.sqrt(M)
    scale_sq = (lam ** 2) / M

    Sigma_X_denom = torch.zeros_like(m_X)
    sum_X = torch.zeros_like(m_X)
    onsager_X = torch.zeros_like(m_X)
    g_next_edge = torch.empty_like(g_prev_edge)

    _reset_f_stream(f_seed)
    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        F_chunk = _make_f_chunk(end - start, M, device, m_W.dtype)
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
            F_chunk=F_chunk,
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
            (scale * g_edge.unsqueeze(1) * F_chunk * W_sel).T.contiguous(),
        )
        onsager_X.scatter_add_(
            1,
            j_chunk.unsqueeze(0).expand(M, -1),
            (scale_sq * g_pair.unsqueeze(1) * var_term_W).T.contiguous(),
        )

    Sigma_X = 1.0 / torch.clamp(Sigma_X_denom, min=1e-10)
    T_X = m_X + Sigma_X * (sum_X - onsager_X * m_X_prev)
    m_X_proposal, v_X_proposal = f_input(Sigma_X, T_X)
    convergence_sum = torch.sum(torch.abs(m_X_proposal - m_X))
    v_X_new = torch.clamp(v_X_proposal, min=1e-8, max=100.0)
    m_X_new = damping * m_X + (1.0 - damping) * m_X_proposal
    v_X_new = damping * v_X + (1.0 - damping) * v_X_new
    return m_X_new, v_X_new, g_next_edge, convergence_sum


def compute_observed_loss(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
    f_seed: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    E = int(i_idx.numel())
    if E == 0:
        return torch.zeros((), dtype=m_W.dtype, device=m_W.device)

    _, M = m_W.shape
    loss_sum = torch.zeros((), dtype=m_W.dtype, device=m_W.device)
    _reset_f_stream(f_seed)
    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        F_chunk = _make_f_chunk(end - start, M, m_W.device, m_W.dtype)
        W_sel = m_W[i_chunk, :]
        X_sel = m_X[:, j_chunk].T
        Y_pred = scale * (W_sel * F_chunk * X_sel).sum(dim=1)
        loss_sum = loss_sum + ((Y_obs[start:end] - Y_pred) ** 2).sum()

    return loss_sum / E


def compute_observed_signal_cosine(
    m_W: torch.Tensor,
    m_X: torch.Tensor,
    Y_clean_obs: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    scale: float,
    f_seed: int,
    edge_chunk_size: int,
) -> torch.Tensor:
    E = int(i_idx.numel())
    if E == 0:
        return torch.zeros((), dtype=m_W.dtype, device=m_W.device)

    _, M = m_W.shape
    inner = torch.zeros((), dtype=m_W.dtype, device=m_W.device)
    teacher_norm_sq = torch.zeros_like(inner)
    student_norm_sq = torch.zeros_like(inner)

    _reset_f_stream(f_seed)
    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        F_chunk = _make_f_chunk(end - start, M, m_W.device, m_W.dtype)
        W_sel = m_W[i_chunk, :]
        X_sel = m_X[:, j_chunk].T
        Y_student = scale * (W_sel * F_chunk * X_sel).sum(dim=1)
        Y_teacher = Y_clean_obs[start:end]
        inner = inner + torch.sum(Y_teacher * Y_student)
        teacher_norm_sq = teacher_norm_sq + torch.sum(Y_teacher ** 2)
        student_norm_sq = student_norm_sq + torch.sum(Y_student ** 2)

    denom = torch.sqrt(torch.clamp(teacher_norm_sq * student_norm_sq, min=1e-30))
    return inner / denom


ORDER_PARAMETER_KEYS = [
    "m_overlap_Y",
    "m_overlap_W",
    "m_overlap_X",
    "Q_Y_teacher",
    "cosine_Y",
    "Q_W",
    "Q_X",
    "convergence",
]


def compute_dense_order_parameters(
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    convergence: torch.Tensor | float | None = None,
) -> dict[str, float]:
    """
    Compute CLAUDE.md order parameters on the dense N1 x N2 x M space.
    """
    N1, M = W_teacher.shape
    _, N2 = X_teacher.shape
    normalizer_y = float(N1 * N2 * M)

    w_overlap_by_mu = torch.sum(W_teacher * m_W, dim=0)
    x_overlap_by_mu = torch.sum(X_teacher * m_X, dim=1)
    m_overlap_Y = torch.sum(w_overlap_by_mu * x_overlap_by_mu) / normalizer_y

    teacher_w_sq_by_mu = torch.sum(W_teacher ** 2, dim=0)
    teacher_x_sq_by_mu = torch.sum(X_teacher ** 2, dim=1)
    Q_Y_teacher = torch.sum(teacher_w_sq_by_mu * teacher_x_sq_by_mu) / normalizer_y
    student_w_sq_by_mu = torch.sum(m_W ** 2, dim=0)
    student_x_sq_by_mu = torch.sum(m_X ** 2, dim=1)
    student_norm_Y = torch.sum(student_w_sq_by_mu * student_x_sq_by_mu) / normalizer_y
    cosine_Y = m_overlap_Y / torch.sqrt(
        torch.clamp(Q_Y_teacher * student_norm_Y, min=1e-30)
    )

    if convergence is None:
        convergence_value = 0.0
    elif isinstance(convergence, torch.Tensor):
        convergence_value = float(convergence.detach().item())
    else:
        convergence_value = float(convergence)

    return {
        "m_overlap_Y": float(m_overlap_Y.detach().item()),
        "m_overlap_W": float(torch.mean(W_teacher * m_W).detach().item()),
        "m_overlap_X": float(torch.mean(X_teacher * m_X).detach().item()),
        "Q_Y_teacher": float(Q_Y_teacher.detach().item()),
        "cosine_Y": float(cosine_Y.detach().item()),
        "Q_W": float(torch.mean(v_W).detach().item()),
        "Q_X": float(torch.mean(v_X).detach().item()),
        "convergence": convergence_value,
    }


def append_order_parameters(
    history: dict[str, list[float]],
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_X: torch.Tensor,
    v_X: torch.Tensor,
    convergence: torch.Tensor | float | None = None,
) -> dict[str, float]:
    order_parameters = compute_dense_order_parameters(
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        m_W=m_W,
        v_W=v_W,
        m_X=m_X,
        v_X=v_X,
        convergence=convergence,
    )
    for key in ORDER_PARAMETER_KEYS:
        history[key].append(order_parameters[key])
    return order_parameters


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
    eval_interval: int = 50,
    early_stop: bool = True,
    init_epsilon: float | None = None,
    track_loss: bool = True,
    loss_eval_interval: int | None = None,
    shared_data: dict[str, torch.Tensor | float | int | str] | None = None,
    return_final_state: bool = False,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    if shared_data is None:
        raise ValueError("shared_data must be provided.")
    if loss_eval_interval is not None:
        eval_interval = loss_eval_interval

    E = int(shared_data["E"])
    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    W_teacher = shared_data["W_teacher"]
    X_teacher = shared_data["X_teacher"]
    Y_noisy = shared_data["Y_noisy_obs"]
    scale = float(shared_data["scale"])
    f_seed = int(shared_data["f_seed"])
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

    final_loss = float("nan")
    steps_taken = max_steps
    prev_loss = float("inf")
    history = {
        "steps": [],
        "damping": [],
        **{key: [] for key in ORDER_PARAMETER_KEYS},
    }
    if track_loss:
        history["loss"] = []
    history_loss_tensors = []

    if E == 0:
        if return_history:
            history["steps"].append(0)
            history["damping"].append(0.0)
            append_order_parameters(
                history=history,
                W_teacher=W_teacher,
                X_teacher=X_teacher,
                m_W=m_W,
                v_W=v_W,
                m_X=m_X,
                v_X=v_X,
                convergence=float("nan"),
            )
        if return_final_state:
            final_state = {
                "m_W": m_W.detach().cpu(),
                "m_X": m_X.detach().cpu(),
            }
            if return_history:
                return 0.0, 0.0, 0, history, final_state
            return 0.0, 0.0, 0, final_state
        if return_history:
            return 0.0, 0.0, 0, history
        return 0.0, 0.0, 0

    if return_history:
        if track_loss:
            m_W_eval = normalize_to_unit_variance(m_W)
            m_X_eval = normalize_to_unit_variance(m_X)
            loss_tensor = compute_observed_loss(
                m_W=m_W_eval,
                m_X=m_X_eval,
                Y_obs=Y_noisy,
                i_idx=i_idx,
                j_idx=j_idx,
                scale=scale,
                f_seed=f_seed,
                edge_chunk_size=edge_chunk_size,
            )
            history_loss_tensors.append(loss_tensor.detach())
        history["steps"].append(0)
        history["damping"].append(0.0)
        append_order_parameters(
            history=history,
            W_teacher=W_teacher,
            X_teacher=X_teacher,
            m_W=m_W,
            v_W=v_W,
            m_X=m_X,
            v_X=v_X,
            convergence=float("nan"),
        )

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
        m_W, v_W, g_prev_edge, convergence_W_sum = alternating_half_step_W_sequential(
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
            f_seed=f_seed,
            edge_chunk_size=edge_chunk_size,
        )

        m_W_prev = m_W_before_W
        m_X_prev = m_X_before_W

        m_W_before_X = m_W
        m_X_before_X = m_X
        m_X, v_X, g_prev_edge, convergence_X_sum = alternating_half_step_X_sequential(
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
            f_seed=f_seed,
            edge_chunk_size=edge_chunk_size,
        )

        m_W_prev = m_W_before_X
        m_X_prev = m_X_before_X
        convergence = (convergence_W_sum + convergence_X_sum) / (
            (W_teacher.shape[0] + X_teacher.shape[1]) * M
        )

        if step % eval_interval == 0 or step == max_steps - 1:
            loss = None
            if track_loss and (early_stop or not return_history or step == max_steps - 1):
                m_W_eval = normalize_to_unit_variance(m_W)
                m_X_eval = normalize_to_unit_variance(m_X)
                loss_tensor = compute_observed_loss(
                    m_W=m_W_eval,
                    m_X=m_X_eval,
                    Y_obs=Y_noisy,
                    i_idx=i_idx,
                    j_idx=j_idx,
                    scale=scale,
                    f_seed=f_seed,
                    edge_chunk_size=edge_chunk_size,
                )
                loss = float(loss_tensor.item())
                final_loss = loss

            if return_history:
                if track_loss and loss is None:
                    m_W_eval = normalize_to_unit_variance(m_W)
                    m_X_eval = normalize_to_unit_variance(m_X)
                    loss_tensor = compute_observed_loss(
                        m_W=m_W_eval,
                        m_X=m_X_eval,
                        Y_obs=Y_noisy,
                        i_idx=i_idx,
                        j_idx=j_idx,
                        scale=scale,
                        f_seed=f_seed,
                        edge_chunk_size=edge_chunk_size,
                    )
                    loss = float(loss_tensor.item())
                    final_loss = loss
                if track_loss:
                    history_loss_tensors.append(loss_tensor.detach())
                history["steps"].append(step + 1)
                history["damping"].append(damping_t)
                append_order_parameters(
                    history=history,
                    W_teacher=W_teacher,
                    X_teacher=X_teacher,
                    m_W=m_W,
                    v_W=v_W,
                    m_X=m_X,
                    v_X=v_X,
                    convergence=convergence,
                )

            if early_stop and track_loss:
                if loss is None:
                    m_W_eval = normalize_to_unit_variance(m_W)
                    m_X_eval = normalize_to_unit_variance(m_X)
                    loss_tensor = compute_observed_loss(
                        m_W=m_W_eval,
                        m_X=m_X_eval,
                        Y_obs=Y_noisy,
                        i_idx=i_idx,
                        j_idx=j_idx,
                        scale=scale,
                        f_seed=f_seed,
                        edge_chunk_size=edge_chunk_size,
                    )
                    loss = float(loss_tensor.item())
                    final_loss = loss
                if abs(prev_loss - loss) < convergence_threshold:
                    steps_taken = step + 1
                    break
                prev_loss = loss

    final_order_parameters = compute_dense_order_parameters(
        W_teacher=W_teacher,
        X_teacher=X_teacher,
        m_W=m_W,
        v_W=v_W,
        m_X=m_X,
        v_X=v_X,
    )
    cosine_Y = final_order_parameters["cosine_Y"]

    if return_history:
        if track_loss and history_loss_tensors:
            history["loss"] = torch.stack(history_loss_tensors).cpu().tolist()
            if not early_stop:
                final_loss = float(history["loss"][-1])
        if return_final_state:
            final_state = {
                "m_W": m_W.detach().cpu(),
                "m_X": m_X.detach().cpu(),
            }
            return cosine_Y, final_loss, steps_taken, history, final_state
        return cosine_Y, final_loss, steps_taken, history

    if return_final_state:
        final_state = {
            "m_W": m_W.detach().cpu(),
            "m_X": m_X.detach().cpu(),
        }
        return cosine_Y, final_loss, steps_taken, final_state

    return cosine_Y, final_loss, steps_taken


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
    eval_interval: int = 50,
    early_stop: bool = True,
    init_epsilon: float | None = None,
    edge_chunk_size: int | None = DEFAULT_EDGE_CHUNK_SIZE,
    track_loss: bool = True,
    loss_eval_interval: int | None = None,
    shared_data: dict[str, torch.Tensor | float | int | str] | None = None,
    return_final_state: bool = False,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    if loss_eval_interval is not None:
        eval_interval = loss_eval_interval

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
        eval_interval=eval_interval,
        early_stop=early_stop,
        init_epsilon=init_epsilon,
        track_loss=track_loss,
        shared_data=shared_data,
        return_final_state=return_final_state,
    )
