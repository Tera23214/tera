"""
Dense-mask G-AMP with F=1, exact Onsager correction, and alternating W -> X
updates with cosine-similarity evaluation.
"""

from .core import (
    alternating_half_step_W,
    alternating_half_step_X,
    build_shared_alpha_data,
    compute_observed_loss,
    compute_step_damping,
    compute_y_cosine_similarity,
    compute_y_cosine_similarity_tensor,
    prepare_global_shared_data,
    train_single_replica_from_shared_data,
)
from terao_gamp_gaussian.utils import f_input, g_out

__all__ = [
    "alternating_half_step_W",
    "alternating_half_step_X",
    "build_shared_alpha_data",
    "compute_observed_loss",
    "compute_step_damping",
    "compute_y_cosine_similarity",
    "compute_y_cosine_similarity_tensor",
    "prepare_global_shared_data",
    "train_single_replica_from_shared_data",
    "f_input",
    "g_out",
]
