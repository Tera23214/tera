#!/usr/bin/env python
"""
Non-uniform N1 graph version of Dence_Alternating using graph_core.two_point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add parent directories to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from graph_core.two_point import generate_two_point_dense_mask
from terao_gamp_gaussian.Dence_Alternating.shared_core import (
    alternating_half_step_W,
    alternating_half_step_X,
    build_shared_alpha_data,
    compute_y_cosine_similarity,
    prepare_global_shared_data,
    train_single_replica_from_shared_data,
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
    p: float = 0.5,
    r: float = 0.5,
    global_data: dict[str, torch.Tensor | float | int] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """
    Prepare graph and observations once for a single alpha using the
    two-point non-uniform row-degree graph model.
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

    (
        mask,
        i_idx,
        j_idx,
        E,
        row_degrees,
        ca,
        cb,
        num_ca,
        num_cb,
        p_eff,
        alpha_eff,
    ) = generate_two_point_dense_mask(
        N1=N1,
        N2=N2,
        M=M,
        alpha=alpha,
        p=p,
        r=r,
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
            "p": p,
            "r": r,
            "p_eff": p_eff,
            "alpha_eff": alpha_eff,
            "ca": ca,
            "cb": cb,
            "num_ca": num_ca,
            "num_cb": num_cb,
            "row_degrees": row_degrees,
            "graph_model": "non_uniform_n1_graph",
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
    p: float = 0.5,
    r: float = 0.5,
    init_epsilon: float | None = None,
    shared_data: dict[str, torch.Tensor | float | int] | None = None,
) -> tuple[float, float, int] | tuple[float, float, int, dict[str, list[float]]]:
    """
    Train one replica with the non-uniform N1 graph version.
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
            p=p,
            r=r,
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
