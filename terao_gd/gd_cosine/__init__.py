"""
Alternating gradient descent utilities with cosine-similarity evaluation.
"""

from .gd import (
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
    compute_y_cosine_similarity,
)

__all__ = [
    "prepare_global_shared_data",
    "prepare_shared_alpha_data",
    "train_single_replica",
    "compute_y_cosine_similarity",
]
