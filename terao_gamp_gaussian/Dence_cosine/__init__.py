"""
Dense-mask G-AMP with F=1, exact Onsager correction, and cosine-similarity
evaluation.

This module implements G-AMP with:
- F = 1 (constant, no spreading)
- Proper Onsager / memory term correction
- Dense observation mask generated from ``BiregularGraph``
"""

from .core import (
    train_single_replica,
    prepare_global_shared_data,
    prepare_shared_alpha_data,
    gamp_step_with_onsager,
    compute_y_cosine_similarity,
)
from terao_gamp_gaussian.utils import f_input, g_out

__all__ = [
    'train_single_replica',
    'prepare_global_shared_data',
    'prepare_shared_alpha_data',
    'gamp_step_with_onsager',
    'compute_y_cosine_similarity',
    'f_input',
    'g_out',
]
