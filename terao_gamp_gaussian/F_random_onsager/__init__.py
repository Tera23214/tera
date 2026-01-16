"""
G-AMP with Random F and Onsager Correction.

This module implements G-AMP with:
- F ~ N(0,1) spreading per edge
- Proper Onsager/memory term correction
- BiregularGraph for Dense Limit
"""

from .core import train_single_replica, gamp_step_with_F_onsager, f_input, g_out

__all__ = [
    'train_single_replica',
    'gamp_step_with_F_onsager',
    'f_input',
    'g_out',
]
