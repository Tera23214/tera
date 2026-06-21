"""
Alternating gradient descent utilities with dense q_Y evaluation.
"""

from .gd import (
    ORDER_PARAMETER_KEYS,
    compute_order_parameters,
    initialize_student_factors,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
    compute_y_cosine_similarity,
)

__all__ = [
    "ORDER_PARAMETER_KEYS",
    "compute_order_parameters",
    "initialize_student_factors",
    "prepare_global_shared_data",
    "prepare_shared_alpha_data",
    "train_single_replica",
    "compute_y_cosine_similarity",
]
