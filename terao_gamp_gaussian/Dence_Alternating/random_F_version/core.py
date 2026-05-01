#!/usr/bin/env python
"""
Random-F version of Dence_Alternating on the same random graph model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from terao_gamp_gaussian.Dence_Alternating.shared_core import (
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
    Prepare teacher, noise, and F_{ijmu} ~ N(0, 1) for the random-F version.
    """
    return _prepare_global_shared_data(
        device=device,
        seed=seed,
        N1=N1,
        N2=N2,
        M=M,
        noise_var=noise_var,
        lam=lam,
        f_mode="random",
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
    Prepare graph and observations once for a single alpha using the existing
    random dense-mask graph generator and a fixed random F tensor.
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

    return build_shared_alpha_data(
        alpha=alpha,
        mask=mask,
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
        },
    )


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
