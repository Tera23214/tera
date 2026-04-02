"""
Alternating gradient descent utilities with cosine-similarity evaluation.
"""

from .gd import (
    train_single_replica,
    compute_y_cosine_similarity,
)

__all__ = [
    "train_single_replica",
    "compute_y_cosine_similarity",
]
