# ================================================================
# seed_filter.py
# Adaptive repeat filtering for minimizer/strobemer seeds
# ================================================================
from collections import Counter

def filter_repetitive_seeds(seeds, max_freq=50):
    """
    Remove seeds (hash,pos1,pos2) where a position is too frequent.
    This drops repeat-induced minimizers.
    """
    ref_ct = Counter([x for _, x, _ in seeds])
    qry_ct = Counter([y for _, _, y in seeds])

    out = []
    for h, x, y in seeds:
        if ref_ct[x] <= max_freq and qry_ct[y] <= max_freq:
            out.append((h, x, y))

    return out
