# ================================================================
# seed_grouping.py
# Group seeds into diagonal buckets to reduce DP complexity
# ================================================================
from collections import defaultdict
from .diagonal_utils import diagonal

def group_seeds_by_diagonal(seeds, diag_band=50_000):
    """
    Seeds: list of (pos_ref, pos_qry)
    Group seeds into buckets based on approx diagonal similarity.
    diag_band: max allowed deviation in diagonal group
    """
    buckets = defaultdict(list)

    for x, y in seeds:
        d = diagonal(x, y)
        key = d // diag_band
        buckets[key].append((x, y))

    # Sort each bucket by ref coordinate
    for key in buckets:
        buckets[key].sort()

    return buckets
