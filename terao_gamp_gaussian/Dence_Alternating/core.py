"""
Shared alternating-GAMP core used by graph-specific Dence_Alternating variants.
"""

from terao_gamp_gaussian.Dence_Alternating.shared_core import (
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
]
