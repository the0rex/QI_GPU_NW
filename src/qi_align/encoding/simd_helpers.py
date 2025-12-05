# ================================================================
# simd_helpers.py
# SIMD-like vector operations on 4-bit encoded DNA
# ================================================================
import numpy as np

def simd_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized a == b.
    Both inputs must be 4-bit arrays (0..4).
    """
    if a.shape != b.shape:
        raise ValueError("Arrays must be same length for simd_equal")
    return (a == b)


def simd_not_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized a != b.
    """
    if a.shape != b.shape:
        raise ValueError("Arrays must be same length for simd_not_equal")
    return (a != b)


def simd_mask_count_same(mask: np.ndarray) -> int:
    """
    Count number of TRUE values in a boolean mask.
    Equivalent to population count.
    """
    return int(mask.sum())
