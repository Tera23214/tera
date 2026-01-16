"""
F_random_onsager_trial: G-AMP with True Biregular Graph and Onsager Correction.

This module tests Onsager correction with a strict biregular graph where
both row degrees AND column degrees are exactly controlled.
"""

from .core import train_single_replica
from .true_biregular import TrueBiregularGraph

__all__ = ['train_single_replica', 'TrueBiregularGraph']
