#!/usr/bin/env python
"""
Random-F version of Edge_Alternating on the same random graph model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Edge_Alternating.shared_core import (
    alternating_half_step_W,
    alternating_half_step_X,
    build_shared_alpha_data,
    compute_y_cosine_similarity,
    prepare_global_shared_data as _prepare_global_shared_data,
    train_single_replica_from_shared_data,
)
from terao_gamp_gaussian.graph import BiregularGraph


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
    global_data: dict[str, torch.Tensor | float | int | str] | None = None,
) -> dict[str, torch.Tensor | float | int | str]:
    """
    Prepare graph, observed Rademacher spreading coefficients, and observations
    once for a single alpha.
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

    shared_data = build_shared_alpha_data(
        alpha=alpha,
        i_idx=i_idx,
        j_idx=j_idx,
        E=E,
        global_data=global_data,
        metadata={
            "alpha2": alpha2,
            "C1": C1,
            "C2": C2,
            "graph_model": "random_graph",
            "f_mode": "random",
            "f_distribution": "rademacher_pm1",
        },
    )

    if E == 0:
        shared_data["F_edge"] = torch.empty(
            (0, M),
            dtype=torch.float32,
            device=device,
        )
        return shared_data

    W_teacher = global_data["W_teacher"]
    X_teacher = global_data["X_teacher"]
    noise_full = global_data["noise_full"]
    scale = float(global_data["scale"])

    # This reseed is isolated to shared alpha data construction. Replica
    # initialization later uses its own explicit seed path.
    torch.manual_seed(seed + 1000)
    F_edge = torch.empty((E, M), device=device, dtype=torch.float32)
    F_edge.bernoulli_(0.5).mul_(2.0).sub_(1.0)

    i_long = i_idx.long()
    j_long = j_idx.long()
    W_sel = W_teacher[i_long, :]
    X_sel = X_teacher[:, j_long].T
    y_clean_obs = scale * (W_sel * F_edge * X_sel).sum(dim=1)
    y_noisy_obs = y_clean_obs + noise_full[i_long, j_long]

    shared_data["F_edge"] = F_edge
    shared_data["Y_clean_obs"] = y_clean_obs
    shared_data["Y_noisy_obs"] = y_noisy_obs
    return shared_data


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
    shared_data: dict[str, torch.Tensor | float | int | str] | None = None,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    """
    Train one replica with the random-F version.
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


__all__ = [
    "alternating_half_step_W",
    "alternating_half_step_X",
    "compute_y_cosine_similarity",
    "prepare_global_shared_data",
    "prepare_shared_alpha_data",
    "train_single_replica",
]
