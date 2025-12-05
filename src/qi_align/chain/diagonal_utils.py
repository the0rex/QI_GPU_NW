# ================================================================
# diagonal_utils.py
# Utilities for seed diagonals and chain integrity
# ================================================================
def diagonal(x, y):
    """
    Compute alignment diagonal d = (y - x).
    Minimizers with similar diagonals belong to the same colinear region.
    """
    return y - x


def is_monotonic_chain(chain):
    """
    Check that a chain is strictly increasing in both coordinates.
    chain: list of (x, y)
    """
    for i in range(1, len(chain)):
        x1, y1 = chain[i-1]
        x2, y2 = chain[i]
        if not (x2 > x1 and y2 > y1):
            return False
    return True
