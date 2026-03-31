#!/usr/bin/env python3
"""
Temporary demo for understanding:
1) `.sum(dim=1)` in Onsager X-side term
2) Sigma denominator accumulation with `scatter_add_`

This mirrors the logic in F_1_onsager/core.py on a tiny toy example.
"""

import math
import torch


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)

    # Tiny deterministic setup
    N1, N2, M = 3, 4, 2
    C = 5  # number of observed edges
    lam = 1.5
    scale_sq = (lam ** 2) / M

    # Messages (toy values)
    m_W = torch.tensor(
        [
            [0.20, -0.10],
            [0.05, 0.30],
            [-0.40, 0.10],
        ],
        dtype=torch.float64,
    )  # (N1, M)

    v_W = torch.tensor(
        [
            [0.90, 1.10],
            [1.20, 0.95],
            [0.85, 1.05],
        ],
        dtype=torch.float64,
    )  # (N1, M)

    m_X = torch.tensor(
        [
            [0.10, -0.20, 0.30, 0.15],
            [-0.40, 0.05, 0.20, -0.10],
        ],
        dtype=torch.float64,
    )  # (M, N2)

    m_X_prev = torch.tensor(
        [
            [0.08, -0.15, 0.35, 0.18],
            [-0.35, 0.02, 0.25, -0.12],
        ],
        dtype=torch.float64,
    )  # (M, N2)

    # Edge list: c-th edge is (i_idx[c], j_idx[c])
    i_idx = torch.tensor([0, 1, 0, 2, 1], dtype=torch.long)  # (C,)
    j_idx = torch.tensor([0, 2, 1, 3, 1], dtype=torch.long)  # (C,)

    # Same indexing pattern as core.py
    W_sel = m_W[i_idx.long(), :]  # (C, M)
    vW_sel = v_W[i_idx.long(), :]  # (C, M)
    X_sel = m_X[:, j_idx.long()].T  # (C, M)
    X_prev_sel = m_X_prev[:, j_idx.long()].T  # (C, M)

    var_term_W = torch.clamp(vW_sel - W_sel ** 2, min=0.0)  # (C, M)

    # ------------------------------------------------------------------
    # Part A: Onsager X-side
    # core.py line:
    # onsager_X_side = scale_sq * (var_term_W * X_sel * X_prev_sel).sum(dim=1)
    # ------------------------------------------------------------------
    onsager_x_vec = scale_sq * (var_term_W * X_sel * X_prev_sel).sum(dim=1)

    # Loop version of the same expression
    onsager_x_loop = torch.zeros(C, dtype=torch.float64)
    for c in range(C):
        total = 0.0
        for mu in range(M):
            total += (
                var_term_W[c, mu].item()
                * X_sel[c, mu].item()
                * X_prev_sel[c, mu].item()
            )
        onsager_x_loop[c] = scale_sq * total

    # ------------------------------------------------------------------
    # Part B: Sigma_W denominator
    # core.py uses scatter_add_ to accumulate edge contributions by row index i
    # ------------------------------------------------------------------
    dg = torch.tensor([-0.3, -1.2, -0.7, -0.5, -0.9], dtype=torch.float64)  # (C,)
    dg_expanded = scale_sq * (-dg).unsqueeze(1) * (X_sel ** 2)  # (C, M)

    sigma_w_denom_vec = torch.zeros_like(m_W)  # (N1, M)
    sigma_w_denom_vec.scatter_add_(
        0,
        i_idx.long().unsqueeze(1).expand(-1, M),
        dg_expanded,
    )

    # Loop version of scatter accumulation
    sigma_w_denom_loop = torch.zeros_like(m_W)
    for c in range(C):
        i = int(i_idx[c].item())
        for mu in range(M):
            sigma_w_denom_loop[i, mu] += dg_expanded[c, mu]

    sigma_w = 1.0 / torch.clamp(sigma_w_denom_vec, min=1e-10)

    # Checks
    assert torch.allclose(onsager_x_vec, onsager_x_loop, atol=1e-12)
    assert torch.allclose(sigma_w_denom_vec, sigma_w_denom_loop, atol=1e-12)

    # Output
    print("=== Tiny demo for `.sum()` vs Sigma accumulation ===")
    print(f"shape(var_term_W): {tuple(var_term_W.shape)}")
    print(f"shape(X_sel):      {tuple(X_sel.shape)}")
    print(f"shape(X_prev_sel): {tuple(X_prev_sel.shape)}")
    print()
    print("[A] Onsager X-side")
    print("onsager_x_vec  :", onsager_x_vec)
    print("onsager_x_loop :", onsager_x_loop)
    print("note: `.sum(dim=1)` reduces M dimension, output is shape (C,)")
    print()
    print("[B] Sigma_W denominator")
    print("sigma_w_denom_vec (scatter_add):")
    print(sigma_w_denom_vec)
    print("sigma_w_denom_loop (for-loop):")
    print(sigma_w_denom_loop)
    print("sigma_w = 1 / clamp(denom, min=1e-10):")
    print(sigma_w)
    print()
    print("takeaway:")
    print("- `.sum(dim=1)` is per-edge reduction over hidden index mu.")
    print("- Sigma denominator is NOT a plain `.sum()`.")
    print("- It is grouped accumulation by node index (i_idx or j_idx) via scatter_add_.")


if __name__ == "__main__":
    main()
