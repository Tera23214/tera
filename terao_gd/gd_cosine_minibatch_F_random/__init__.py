"""
Alternating mini-batch SGD utilities for the F-random observation model.
"""

from .gd import (
    compute_predictions,
    compute_y_cosine_similarity,
    prepare_shared_alpha_data,
    prepare_global_shared_data,
    train_single_replica,
)

__all__ = [
    "compute_predictions",
    "compute_y_cosine_similarity",
    "prepare_shared_alpha_data",
    "prepare_global_shared_data",
    "train_single_replica",
]
