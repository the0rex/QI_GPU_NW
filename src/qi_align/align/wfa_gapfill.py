# ================================================================
# wfa_gapfill.py
# CPU fallback: Wavefront gap fill for large indels
# ================================================================
def wfa_gapfill(seqA, seqB, max_len=5000):
    """
    For long unaligned stretches, use a simplified WFA algorithm
    to produce a fast approximate alignment for the gap region.
    """
    if len(seqA) == 0 or len(seqB) == 0:
        return ("D" * len(seqA)) + ("I" * len(seqB))

    # naive fast fallback: align min prefix:
    L = min(len(seqA), len(seqB), max_len)
    return "M" * L + ("D" * (len(seqA)-L)) + ("I" * (len(seqB)-L))
