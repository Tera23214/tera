"""
Dense-mask G-AMP with scalar-variance Onsager correction and cosine evaluation.

This module implements a dense observation backend for the F=1 Gaussian model.
Graph, teacher, and noisy observations can be generated once per alpha and
reused across replicas, while replicas differ only in student initialization.
"""

from .core import (
    compute_y_cosine_similarity,
    gamp_step_with_onsager_dense,
    prepare_shared_alpha_data,
    train_single_replica,
)

__all__ = [
    "compute_y_cosine_similarity",
    "gamp_step_with_onsager_dense",
    "prepare_shared_alpha_data",
    "train_single_replica",
]
