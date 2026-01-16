#!/usr/bin/env python
"""
Utility functions for G-AMP.

Provides helper functions for Q_Y calculation and normalization.
"""

import torch


def normalize_to_unit_variance(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor so that mean square equals 1.
    
    E[x^2] = 1  =>  x_new = x / sqrt(mean(x^2))
    """
    mean_sq = (tensor ** 2).mean()
    if mean_sq > 0:
        return tensor / torch.sqrt(mean_sq)
    return tensor


def compute_qy(
    W_student: torch.Tensor,
    X_student: torch.Tensor,
    W_teacher: torch.Tensor,
    X_teacher: torch.Tensor,
) -> float:
    """
    Compute Q_Y overlap using theoretical normalization.
    
    Q_Y = <Y_teacher, Y_student> / (N1 * N2 * M)
    
    When estimation is perfect (Y_student = Y_teacher), Q_Y ≈ 1.
    """
    N1, M = W_teacher.shape
    N2 = X_teacher.shape[1]
    
    # Full matrix products
    Y_teacher = W_teacher @ X_teacher  # (N1, N2)
    Y_student = W_student @ X_student  # (N1, N2)
    
    # Inner product with theoretical normalization
    inner_product = (Y_teacher * Y_student).sum()
    
    return (inner_product / (N1 * N2 * M)).item()


def compute_predictions(
    W: torch.Tensor,       # (N1, M)
    X: torch.Tensor,       # (M, N2)
    i_idx: torch.Tensor,   # (C,)
    j_idx: torch.Tensor,   # (C,)
    M: int,                # Rank for 1/√M scaling
) -> torch.Tensor:
    """
    Compute predictions Y_pred for observed entries.
    
    Y_pred[c] = (1/√M) * sum_mu W[i_c, mu] * X[mu, j_c]
    """
    import math
    W_sel = W[i_idx.long(), :]       # (C, M)
    X_sel = X[:, j_idx.long()].T     # (C, M)
    
    Y_pred = (W_sel * X_sel).sum(dim=1) / math.sqrt(M)  # (C,)
    return Y_pred
