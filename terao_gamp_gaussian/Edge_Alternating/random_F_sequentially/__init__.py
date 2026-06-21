"""
Sequentially aggregated random-F Edge_Alternating G-AMP.
"""

from .core import (
    ORDER_PARAMETER_KEYS,
    append_order_parameters,
    compute_dense_order_parameters,
    compute_observed_loss,
    compute_observed_signal_cosine,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)

__all__ = [
    "ORDER_PARAMETER_KEYS",
    "append_order_parameters",
    "compute_dense_order_parameters",
    "compute_observed_loss",
    "compute_observed_signal_cosine",
    "prepare_global_shared_data",
    "prepare_shared_alpha_data",
    "train_single_replica",
]
