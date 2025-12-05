# ================================================================
# stitcher.py
# Combine per-chunk CIGARs into a global CIGAR
# ================================================================
from .overlap_dp import local_overlap_align

def _trim_overlap(cigar, overlap):
    """
    Remove last overlap portion of the CIGAR from chunk A.
    Simplified: remove final ~overlap M/I/D operations.
    """
    trim_len = overlap
    trimmed = []

    count = 0
    for op in reversed(cigar):
        if count < trim_len:
            count += 1
            continue
        trimmed.append(op)

    return "".join(reversed(trimmed))


def stitch_cigars_global(chunk_cigars, overlap=3000):
    """
    chunk_cigars: list of CIGAR strings from chunk alignments.
    overlap: number of bases to re-align at boundaries.

    Returns single global CIGAR.
    """

    if not chunk_cigars:
        return ""

    if len(chunk_cigars) == 1:
        return chunk_cigars[0]

    global_cigar = chunk_cigars[0]

    for i in range(1, len(chunk_cigars)):
        prev = chunk_cigars[i-1]
        curr = chunk_cigars[i]

        # trim end of previous chunk
        prev_trimmed = _trim_overlap(prev, overlap)

        # trim beginning of current chunk
        curr_trimmed = curr[overlap:]

        # refine overlap boundary
        overlap_cigar = local_overlap_align(
            prev[-overlap:],   # last part of chunk i-1
            curr[:overlap]     # first part of chunk i
        )

        # assemble
        global_cigar = prev_trimmed + overlap_cigar + curr_trimmed

    return global_cigar
