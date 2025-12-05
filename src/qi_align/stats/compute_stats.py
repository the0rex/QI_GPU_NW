# ================================================================
# compute_stats.py
# High-level alignment statistics from CIGAR + mismatch data
# ================================================================
from .cigar import count_ops

def compute_alignment_stats(
    cigar: str,
    mismatches: int = None,
    ref_length: int = None,
    qry_length: int = None
):
    """
    Compute alignment statistics.

    mismatches:
        If None → assume all M are matches.
        If provided → M operations are split into match + mismatch.

    ref_length, qry_length:
        Used for coverage or global reporting (optional).

    Returns:
        dict with fields:
            matches
            mismatches
            insertions
            deletions
            gaps
            aligned_length
            identity
            divergence
    """

    ops = count_ops(cigar)

    M = ops["M"]
    I = ops["I"]
    D = ops["D"]
    total = ops["total"]

    if mismatches is None:
        mismatches = 0
        matches = M
    else:
        matches = M - mismatches

    gaps = I + D
    aligned = total

    identity = matches / aligned if aligned > 0 else 0.0
    divergence = mismatches / aligned if aligned > 0 else 0.0

    out = {
        "matches": matches,
        "mismatches": mismatches,
        "insertions": I,
        "deletions": D,
        "gaps": gaps,
        "aligned_length": aligned,
        "identity": identity,
        "divergence": divergence,
    }

    if ref_length is not None:
        out["ref_coverage"] = aligned / ref_length

    if qry_length is not None:
        out["qry_coverage"] = aligned / qry_length

    return out
