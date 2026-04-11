"""
Dense-mask G-AMP with F=1, exact Onsager correction, and alternating W -> X
updates with cosine-similarity evaluation.
"""

from .core import (
    alternating_half_step_W,
    alternating_half_step_X,
    compute_y_cosine_similarity,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    train_single_replica,
)
from terao_gamp_gaussian.utils import f_input, g_out

__all__ = [
    "alternating_half_step_W",
    "alternating_half_step_X",
    "compute_y_cosine_similarity",
    "prepare_global_shared_data",
    "prepare_shared_alpha_data",
    "train_single_replica",
    "f_input",
    "g_out",
]
