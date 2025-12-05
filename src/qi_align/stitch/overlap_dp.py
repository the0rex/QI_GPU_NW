# ================================================================
# overlap_dp.py
# Local DP to refine overlapping chunk boundaries
# (size ~ 3000 bp → CPU is fast enough)
# ================================================================
import numpy as np

def local_overlap_align(seqA, seqB, match=2, mismatch=-2, gap=-3):
    """
    Local alignment (Smith-Waterman-style) for chunk boundary stitching.
    Returns a refined CIGAR string for the overlapping region.
    """
    n = len(seqA)
    m = len(seqB)

    H = np.zeros((n+1, m+1), dtype=np.int32)
    PTR = np.zeros((n+1, m+1), dtype=np.int8)

    # pointers:
    # 1 = diag, 2 = up, 3 = left, 0 = stop

    best = 0
    bi = bj = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            sc = match if seqA[i-1] == seqB[j-1] else mismatch
            diag = H[i-1, j-1] + sc
            up   = H[i-1, j] + gap
            left = H[i, j-1] + gap

            best_score = max(0, diag, up, left)
            H[i, j] = best_score

            if best_score == 0:
                PTR[i,j] = 0
            elif best_score == diag:
                PTR[i,j] = 1
            elif best_score == up:
                PTR[i,j] = 2
            else:
                PTR[i,j] = 3

            if best_score > best:
                best = best_score
                bi, bj = i, j

    # traceback
    cigar = []
    i, j = bi, bj

    while H[i,j] > 0:
        if PTR[i,j] == 1:
            cigar.append("M")
            i -= 1; j -= 1
        elif PTR[i,j] == 2:
            cigar.append("D")
            i -= 1
        elif PTR[i,j] == 3:
            cigar.append("I")
            j -= 1
        else:
            break

    cigar.reverse()
    return "".join(cigar)
