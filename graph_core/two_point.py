"""
Two-point heterogeneous row-degree graph generators.

This module centralizes the logic for graphs where each row degree takes one
of two values:

    C_i = ca with probability p
    C_i = cb with probability 1 - p

with r = ca / cb and alpha controlling the mean row degree.
"""

from typing import Tuple

import numpy as np
import torch


def _make_rng(seed: int | None) -> np.random.Generator:
    """Create a local RNG so sampling does not mutate global state."""
    return np.random.default_rng(seed)


def _sample_edges_from_row_degrees(
    row_degrees: np.ndarray,
    N2: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Sample unique observed columns for each row given per-row degrees.
    """
    i_list: list[int] = []
    j_list: list[int] = []

    available = np.arange(N2)
    for i, degree in enumerate(row_degrees):
        if degree <= 0:
            continue
        selected = rng.choice(available, size=degree, replace=False)
        i_list.extend([i] * degree)
        j_list.extend(selected.tolist())

    i_idx = torch.tensor(i_list, dtype=torch.long, device=device)
    j_idx = torch.tensor(j_list, dtype=torch.long, device=device)

    return i_idx, j_idx, len(i_list)


def resolve_two_point_degrees(
    N1: int,
    N2: int,
    M: int,
    alpha: float,
    p: float,
    r: float,
) -> Tuple[int, int, int, int, float, float]:
    """
    Convert (alpha, p, r=ca/cb) into integer two-point row degrees.

    Returns:
        ca: Rounded degree level associated with mixture weight p
        cb: Rounded degree level associated with mixture weight 1 - p
        num_ca: Number of rows assigned degree ca
        num_cb: Number of rows assigned degree cb
        p_eff: Realized fraction num_ca / N1
        alpha_eff: Realized mean degree divided by M
    """
    if N1 <= 0:
        raise ValueError("N1 must be positive.")
    if N2 <= 0:
        raise ValueError("N2 must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must satisfy 0 <= p <= 1, got {p}.")
    if r <= 0.0:
        raise ValueError(f"r must be positive, got {r}.")
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")

    num_ca = int(round(p * N1))
    num_ca = max(0, min(num_ca, N1))
    num_cb = N1 - num_ca
    p_eff = num_ca / N1

    target_mean_degree = alpha * M
    target_total_degree_float = alpha * M * N1
    target_total_degree = int(round(target_total_degree_float))

    if not np.isclose(target_total_degree_float, target_total_degree, atol=1e-9):
        raise ValueError(
            "Exact mean degree is impossible because N1 * alpha * M is not an integer: "
            f"N1*alpha*M={target_total_degree_float}."
        )

    max_total_degree = N1 * N2
    if target_total_degree < 0 or target_total_degree > max_total_degree:
        raise ValueError(
            "Exact mean degree is impossible because the requested total number of "
            f"edges {target_total_degree} is outside [0, {max_total_degree}]."
        )

    if target_total_degree == 0:
        ca = 0
        cb = 0
    elif num_ca == 0:
        if target_total_degree % num_cb != 0:
            raise ValueError(
                "Exact mean degree is impossible with num_ca=0 because the total "
                f"edge count {target_total_degree} is not divisible by num_cb={num_cb}."
            )
        cb = target_total_degree // num_cb
        if not (0 <= cb <= N2):
            raise ValueError(
                "Exact mean degree is impossible because the resolved cb is outside "
                f"[0, N2]: cb={cb}, N2={N2}."
            )
        ca = int(round(r * cb))
        ca = max(0, min(ca, N2))
    elif num_cb == 0:
        if target_total_degree % num_ca != 0:
            raise ValueError(
                "Exact mean degree is impossible with num_cb=0 because the total "
                f"edge count {target_total_degree} is not divisible by num_ca={num_ca}."
            )
        ca = target_total_degree // num_ca
        if not (0 <= ca <= N2):
            raise ValueError(
                "Exact mean degree is impossible because the resolved ca is outside "
                f"[0, N2]: ca={ca}, N2={N2}."
            )
        cb = int(round(ca / r))
        cb = max(0, min(cb, N2))
    else:
        denom = p_eff * r + (1.0 - p_eff)
        cb_real = target_mean_degree / denom if denom != 0.0 else 0.0
        ca_real = r * cb_real

        best_pair: tuple[int, int] | None = None
        best_error: tuple[float, float, float] | None = None

        for ca_candidate in range(N2 + 1):
            remaining = target_total_degree - num_ca * ca_candidate
            if remaining < 0:
                break
            if remaining % num_cb != 0:
                continue

            cb_candidate = remaining // num_cb
            if not (0 <= cb_candidate <= N2):
                continue

            ratio_error = abs(ca_candidate - r * cb_candidate)
            distance_error = abs(ca_candidate - ca_real) + abs(cb_candidate - cb_real)
            imbalance_error = abs(ca_candidate - cb_candidate)
            score = (ratio_error, distance_error, imbalance_error)

            if best_error is None or score < best_error:
                best_pair = (ca_candidate, cb_candidate)
                best_error = score

        if best_pair is None:
            raise ValueError(
                "Exact mean degree is impossible with the current (alpha, p, r, N1, N2, M) "
                "because no integer pair (ca, cb) satisfies the total-edge constraint."
            )

        ca, cb = best_pair

    realized_mean_degree = p_eff * ca + (1.0 - p_eff) * cb
    alpha_eff = realized_mean_degree / M

    return ca, cb, num_ca, num_cb, p_eff, alpha_eff


def generate_two_point_row_degree_graph(
    N1: int,
    N2: int,
    M: int,
    alpha: float,
    p: float,
    r: float,
    device: torch.device,
    seed: int = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    """
    Generate a graph with a two-point mixture of row degrees.

    The realized mixture uses exactly round(p * N1) rows with degree ca.
    """
    ca, cb, num_ca, num_cb, p_eff, alpha_eff = resolve_two_point_degrees(
        N1=N1,
        N2=N2,
        M=M,
        alpha=alpha,
        p=p,
        r=r,
    )

    rng = _make_rng(seed)
    row_degrees_np = np.full(N1, cb, dtype=np.int32)
    if num_ca > 0:
        ca_rows = rng.choice(N1, size=num_ca, replace=False)
        row_degrees_np[ca_rows] = ca

    i_idx, j_idx, E = _sample_edges_from_row_degrees(
        row_degrees=row_degrees_np,
        N2=N2,
        device=device,
        rng=rng,
    )
    row_degrees = torch.tensor(row_degrees_np, dtype=torch.long, device=device)

    return i_idx, j_idx, E, row_degrees, ca, cb, num_ca, num_cb, p_eff, alpha_eff


def generate_two_point_dense_mask(
    N1: int,
    N2: int,
    M: int,
    alpha: float,
    p: float,
    r: float,
    device: torch.device,
    seed: int = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    """
    Generate a dense observation mask for the two-point row-degree model.
    """
    i_idx, j_idx, E, row_degrees, ca, cb, num_ca, num_cb, p_eff, alpha_eff = (
        generate_two_point_row_degree_graph(
            N1=N1,
            N2=N2,
            M=M,
            alpha=alpha,
            p=p,
            r=r,
            device=device,
            seed=seed,
        )
    )

    mask = torch.zeros((N1, N2), dtype=torch.float32, device=device)
    if E > 0:
        mask[i_idx.long(), j_idx.long()] = 1.0

    return (
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
    )
