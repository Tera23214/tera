"""
Sequentially aggregated F=1 Edge_Alternating G-AMP.
"""

from .core import (
    compute_observed_loss,
    compute_observed_signal_cosine,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)

__all__ = [
    "compute_observed_loss",
    "compute_observed_signal_cosine",
    "prepare_global_shared_data",
    "prepare_shared_alpha_data",
    "train_single_replica",
]
