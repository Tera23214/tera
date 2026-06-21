#!/usr/bin/env python
"""
Sequentially aggregated one-variable random-F AMP.

This variant is intended as a closer Python-side comparison target for the
p=2 one-variable C++ AMP implementation.  It uses a single student matrix W
and observations of the form

    y_ij = lambda / sqrt(M) * sum_mu F_ijmu W_i,mu W_j,mu + noise.

Self edges are removed, and each observed edge contributes simultaneously to
the updates of both endpoint rows.
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
)
from terao_gamp_gaussian.graph import BiregularGraph  # noqa: E402
from terao_gamp_gaussian.utils import (  # noqa: E402
    f_input,
    g_out,
    initialize_correlated_student,
)

DEFAULT_EDGE_CHUNK_SIZE = 4096
ORDER_PARAMETER_KEYS = ["m_overlap_W", "Q_W", "convergence"]


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
    F_chunk = torch.empty((num_edges, M), device=device, dtype=dtype)
    F_chunk.bernoulli_(0.5).mul_(2.0).sub_(1.0)
    return F_chunk


def prepare_global_shared_data(
    device: torch.device,
    seed: int = 1,
    N: int = 1000,
    M: int = 10,
    noise_var: float = 1e-10,
    lam: float = 1.0,
) -> dict[str, torch.Tensor | float | int | str]:
    scale = lam / math.sqrt(M)

    torch.manual_seed(seed)
    W_teacher = torch.randn(N, M, device=device, dtype=torch.float32)

    torch.manual_seed(seed)
    noise_full = torch.randn((N, N), device=device, dtype=torch.float32)
    noise_full = noise_full * math.sqrt(noise_var)

    return {
        "seed": seed,
        "scale": scale,
        "W_teacher": W_teacher,
        "noise_full": noise_full,
        "one_variable": True,
        "student_matrix": "W",
        "observation_model": "Y_ij = lambda/sqrt(M) * sum_mu F_ijmu W_i,mu W_j,mu + noise",
    }


def prepare_shared_alpha_data(
    alpha: float,
    device: torch.device,
    seed: int = 1,
    N: int = 1000,
    M: int = 10,
    noise_var: float = 1e-10,
    lam: float = 1.0,
    edge_chunk_size: int | None = DEFAULT_EDGE_CHUNK_SIZE,
    global_data: dict[str, torch.Tensor | float | int | str] | None = None,
) -> dict[str, torch.Tensor | float | int | str]:
    if global_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=seed,
            N=N,
            M=M,
            noise_var=noise_var,
            lam=lam,
        )

    graph = BiregularGraph()
    i_idx, j_idx, E_raw, C1, C2, alpha2 = graph.generate(
        N1=N,
        N2=N,
        M=M,
        alpha1=alpha,
        device=device,
        seed=seed,
    )
    nonself_mask = i_idx != j_idx
    removed_self_edges = int((~nonself_mask).sum().item())
    i_idx = i_idx[nonself_mask].contiguous()
    j_idx = j_idx[nonself_mask].contiguous()
    E = int(i_idx.numel())

    W_teacher = global_data["W_teacher"]
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
        W_i = W_teacher[i_chunk, :]
        W_j = W_teacher[j_chunk, :]
        y_clean = scale * (W_i * F_chunk * W_j).sum(dim=1)
        y_clean_obs[start:end] = y_clean
        y_noisy_obs[start:end] = y_clean + noise_full[i_chunk, j_chunk]

    return {
        "alpha": alpha,
        "alpha2": alpha2,
        "C1": C1,
        "C2": C2,
        "E": E,
        "E_raw": E_raw,
        "removed_self_edges": removed_self_edges,
        "scale": scale,
        "i_idx": i_idx,
        "j_idx": j_idx,
        "W_teacher": W_teacher,
        "Y_clean_obs": y_clean_obs,
        "Y_noisy_obs": y_noisy_obs,
        "graph_model": "random_graph_row_degree",
        "f_mode": "random",
        "f_distribution": "rademacher_pm1",
        "f_seed": f_seed,
        "edge_chunk_size": chunk_size,
        "self_edges": "excluded",
        "symmetric_endpoint_update": True,
    }


def _initialize_student(
    W_teacher: torch.Tensor,
    seed: int,
    init_epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed + 2000)
    if init_epsilon is None:
        m_W = torch.randn_like(W_teacher)
    else:
        m_W = initialize_correlated_student(W_teacher, init_epsilon)

    if init_epsilon is not None and math.isclose(
        float(init_epsilon), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        v_W = m_W ** 2
    else:
        v_W = torch.ones_like(m_W)
    return m_W, v_W


def _compute_edge_terms(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_W_prev: torch.Tensor,
    Y_obs: torch.Tensor,
    i_chunk: torch.Tensor,
    j_chunk: torch.Tensor,
    g_prev_chunk: torch.Tensor,
    F_chunk: torch.Tensor,
    scale: float,
    scale_sq: float,
    noise_var: float,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    W_i = m_W[i_chunk, :]
    W_j = m_W[j_chunk, :]
    v_i = v_W[i_chunk, :]
    v_j = v_W[j_chunk, :]
    W_i_prev = m_W_prev[i_chunk, :]
    W_j_prev = m_W_prev[j_chunk, :]

    var_i = torch.clamp(v_i - W_i ** 2, min=0.0)
    var_j = torch.clamp(v_j - W_j ** 2, min=0.0)

    omega_main = scale * (W_i * F_chunk * W_j).sum(dim=1)
    onsager_i_side = scale_sq * (var_j * W_i * W_i_prev).sum(dim=1)
    onsager_j_side = scale_sq * (var_i * W_j * W_j_prev).sum(dim=1)
    V = scale_sq * (v_i * v_j - (W_i ** 2) * (W_j ** 2)).sum(dim=1)
    V = torch.clamp(V, min=1e-10)

    omega = omega_main - g_prev_chunk * (onsager_i_side + onsager_j_side)
    g_raw, dg = g_out(omega, Y_obs, V, noise_var)
    g_edge = damping * g_prev_chunk + (1.0 - damping) * g_raw
    g_edge = torch.clamp(g_edge, min=-100.0, max=100.0)

    return g_edge, dg, var_i, var_j, W_i, W_j


def symmetric_w_step_sequential(
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    m_W_prev: torch.Tensor,
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

    Sigma_denom = torch.zeros_like(m_W)
    sum_W = torch.zeros_like(m_W)
    onsager_W = torch.zeros_like(m_W)
    g_next_edge = torch.empty_like(g_prev_edge)

    _reset_f_stream(f_seed)
    for _chunk_id, start, end in _edge_ranges(E, edge_chunk_size):
        i_chunk = i_idx[start:end].long()
        j_chunk = j_idx[start:end].long()
        F_chunk = _make_f_chunk(end - start, M, device, m_W.dtype)
        g_edge, dg, var_i, var_j, W_i, W_j = _compute_edge_terms(
            m_W=m_W,
            v_W=v_W,
            m_W_prev=m_W_prev,
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

        i_index = i_chunk.unsqueeze(1).expand(-1, M)
        j_index = j_chunk.unsqueeze(1).expand(-1, M)

        Sigma_denom.scatter_add_(
            0,
            i_index,
            scale_sq * (-dg).unsqueeze(1) * (W_j ** 2),
        )
        sum_W.scatter_add_(
            0,
            i_index,
            scale * g_edge.unsqueeze(1) * F_chunk * W_j,
        )
        onsager_W.scatter_add_(
            0,
            i_index,
            scale_sq * g_pair.unsqueeze(1) * var_j,
        )

        Sigma_denom.scatter_add_(
            0,
            j_index,
            scale_sq * (-dg).unsqueeze(1) * (W_i ** 2),
        )
        sum_W.scatter_add_(
            0,
            j_index,
            scale * g_edge.unsqueeze(1) * F_chunk * W_i,
        )
        onsager_W.scatter_add_(
            0,
            j_index,
            scale_sq * g_pair.unsqueeze(1) * var_i,
        )

    Sigma = 1.0 / torch.clamp(Sigma_denom, min=1e-10)
    T = m_W + Sigma * (sum_W - onsager_W * m_W_prev)
    m_proposal, v_proposal = f_input(Sigma, T)
    convergence_sum = torch.sum(torch.abs(m_proposal - m_W))
    v_new = torch.clamp(v_proposal, min=1e-8, max=100.0)
    m_new = damping * m_W + (1.0 - damping) * m_proposal
    v_new = damping * v_W + (1.0 - damping) * v_new
    return m_new, v_new, g_next_edge, convergence_sum


def compute_order_parameters(
    W_teacher: torch.Tensor,
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    convergence: torch.Tensor | float | None = None,
) -> dict[str, float]:
    if convergence is None:
        convergence_value = 0.0
    elif isinstance(convergence, torch.Tensor):
        convergence_value = float(convergence.detach().item())
    else:
        convergence_value = float(convergence)

    return {
        "m_overlap_W": float(torch.mean(W_teacher * m_W).detach().item()),
        "Q_W": float(torch.mean(v_W).detach().item()),
        "convergence": convergence_value,
    }


def append_order_parameters(
    history: dict[str, list[float]],
    W_teacher: torch.Tensor,
    m_W: torch.Tensor,
    v_W: torch.Tensor,
    convergence: torch.Tensor | float | None = None,
) -> None:
    order_parameters = compute_order_parameters(
        W_teacher=W_teacher,
        m_W=m_W,
        v_W=v_W,
        convergence=convergence,
    )
    for key in ORDER_PARAMETER_KEYS:
        history[key].append(order_parameters[key])


def train_single_replica(
    alpha: float,
    device: torch.device,
    seed: int = 1,
    N: int = 1000,
    M: int = 10,
    max_steps: int = 100,
    damping: float = 0.0,
    use_step_damping: bool = False,
    damping_beta_scale: float = 1e-2,
    damping_beta_max: float = 0.4,
    noise_var: float = 1e-10,
    lam: float = 1.0,
    convergence_threshold: float = 1e-6,
    return_history: bool = False,
    eval_interval: int = 1,
    early_stop: bool = False,
    init_epsilon: float | None = 0.01,
    edge_chunk_size: int | None = DEFAULT_EDGE_CHUNK_SIZE,
    shared_data: dict[str, torch.Tensor | float | int | str] | None = None,
    return_final_state: bool = False,
):
    del early_stop, convergence_threshold
    if shared_data is None:
        global_data = prepare_global_shared_data(
            device=device,
            seed=seed,
            N=N,
            M=M,
            noise_var=noise_var,
            lam=lam,
        )
        shared_data = prepare_shared_alpha_data(
            alpha=alpha,
            device=device,
            seed=seed,
            N=N,
            M=M,
            noise_var=noise_var,
            lam=lam,
            edge_chunk_size=edge_chunk_size,
            global_data=global_data,
        )

    W_teacher = shared_data["W_teacher"]
    Y_noisy = shared_data["Y_noisy_obs"]
    i_idx = shared_data["i_idx"]
    j_idx = shared_data["j_idx"]
    E = int(shared_data["E"])
    f_seed = int(shared_data["f_seed"])
    edge_chunk_size = int(shared_data["edge_chunk_size"])
    _, M = W_teacher.shape

    m_W, v_W = _initialize_student(
        W_teacher=W_teacher,
        seed=seed,
        init_epsilon=init_epsilon,
    )
    g_prev_edge = torch.zeros(E, device=device)
    m_W_prev = m_W.clone()

    steps_taken = max_steps
    history = {
        "steps": [],
        "damping": [],
        **{key: [] for key in ORDER_PARAMETER_KEYS},
    }

    if return_history:
        history["steps"].append(0)
        history["damping"].append(0.0)
        append_order_parameters(
            history=history,
            W_teacher=W_teacher,
            m_W=m_W,
            v_W=v_W,
            convergence=float("nan"),
        )

    if E == 0:
        final_state = {"m_W": m_W.detach().cpu()}
        if return_history and return_final_state:
            return 0.0, float("nan"), 0, history, final_state
        if return_history:
            return 0.0, float("nan"), 0, history
        if return_final_state:
            return 0.0, float("nan"), 0, final_state
        return 0.0, float("nan"), 0

    for step in range(max_steps):
        damping_t = compute_step_damping(
            step=step,
            base_damping=damping,
            use_step_damping=use_step_damping,
            beta_scale=damping_beta_scale,
            beta_max=damping_beta_max,
        )
        m_W_before = m_W
        m_W, v_W, g_prev_edge, convergence_sum = symmetric_w_step_sequential(
            m_W=m_W,
            v_W=v_W,
            m_W_prev=m_W_prev,
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
        m_W_prev = m_W_before
        convergence = convergence_sum / (W_teacher.shape[0] * M)

        if return_history and (step % eval_interval == 0 or step == max_steps - 1):
            history["steps"].append(step + 1)
            history["damping"].append(damping_t)
            append_order_parameters(
                history=history,
                W_teacher=W_teacher,
                m_W=m_W,
                v_W=v_W,
                convergence=convergence,
            )

    final_order_parameters = compute_order_parameters(
        W_teacher=W_teacher,
        m_W=m_W,
        v_W=v_W,
    )
    metric = final_order_parameters["m_overlap_W"]
    final_loss = float("nan")

    if return_final_state:
        final_state = {"m_W": m_W.detach().cpu()}
        if return_history:
            return metric, final_loss, steps_taken, history, final_state
        return metric, final_loss, steps_taken, final_state
    if return_history:
        return metric, final_loss, steps_taken, history
    return metric, final_loss, steps_taken

