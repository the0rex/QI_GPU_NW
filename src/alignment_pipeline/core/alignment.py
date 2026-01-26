"""
Core alignment functions from original main_pipeline.py
Author: Rowel Facunla
"""

import numpy as np
from ..algorithms.nw_affine import (
    nw_affine_chunk,
    nw_affine_chunk_with_reanchor,
    nw_overlap_realign,
    detect_gap_explosion
)

# Re-export functions
__all__ = [
    'nw_affine_chunk',
    'nw_affine_chunk_with_reanchor',
    'nw_overlap_realign',
    'detect_gap_explosion'
]