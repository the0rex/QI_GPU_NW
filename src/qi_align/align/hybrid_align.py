# ================================================================
# hybrid_align.py
# Hybrid Global Aligner (CPU fallback + deterministic traceback)
# Supports:
#   - Affine-gap dynamic programming (Gotoh)
#   - Deterministic traceback (diag > up > left)
#   - QI scoring (4-bit sequences)
# ================================================================

import numpy as np

# ---------------------------------------------------------------
# Helper: indexing for score matrix
# ---------------------------------------------------------------
def score_fn(score_matrix, a, b):
    return score_matrix[a][b]


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------
def hybrid_global_align(seqA, seqB, score_matrix, gopen, gext):
    """
    Performs full affine-gap global Needleman–Wunsch alignment.
    Used for:
        - Small sequences (<2kb)
        - Chunk-level alignment (CPU fallback or CPU mode)
    Deterministic traceback:
        priority DIAG > UP > LEFT
    """

    n = len(seqA)
    m = len(seqB)

    # M: match/mismatch matrix
    # I: gap in B (insertion in A)
    # D: gap in A (deletion in B)

    M = np.full((n + 1, m + 1), -10**12, dtype=np.int64)
    I = np.full((n + 1, m + 1), -10**12, dtype=np.int64)
    D = np.full((n + 1, m + 1), -10**12, dtype=np.int64)

    # -----------------------------------------------------------
    # Initialize DP
    # -----------------------------------------------------------
    M[0][0] = 0
    I[0][0] = D[0][0] = -10**12

    # first row (gaps in A)
    for j in range(1, m + 1):
        M[0][j] = -10**12
        I[0][j] = -10**12
        D[0][j] = gopen + (j - 1) * gext

    # first column (gaps in B)
    for i in range(1, n + 1):
        M[i][0] = -10**12
        D[i][0] = -10**12
        I[i][0] = gopen + (i - 1) * gext

    # -----------------------------------------------------------
    # Fill DP matrices
    # -----------------------------------------------------------
    for i in range(1, n + 1):
        ai = seqA[i - 1]
        for j in range(1, m + 1):
            bj = seqB[j - 1]

            # Match/Mismatch
            s = score_fn(score_matrix, ai, bj)
            M[i][j] = max(
                M[i - 1][j - 1] + s,
                I[i - 1][j - 1] + s,
                D[i - 1][j - 1] + s
            )

            # Insertion (gap in B)
            I[i][j] = max(
                M[i - 1][j] + gopen,
                I[i - 1][j] + gext
            )

            # Deletion (gap in A)
            D[i][j] = max(
                M[i][j - 1] + gopen,
                D[i][j - 1] + gext
            )

    # -----------------------------------------------------------
    # Choose best ending
    # -----------------------------------------------------------
    final_score = max(M[n][m], I[n][m], D[n][m])
    if final_score == M[n][m]:
        state = "M"
    elif final_score == I[n][m]:
        state = "I"
    else:
        state = "D"

    # -----------------------------------------------------------
    # Deterministic traceback
    # Priority: DIAG > UP > LEFT
    # -----------------------------------------------------------
    cigar = []
    i, j = n, m

    while i > 0 or j > 0:

        if state == "M":
            # DIAG
            if i > 0 and j > 0:
                ai = seqA[i - 1]
                bj = seqB[j - 1]
                s = score_fn(score_matrix, ai, bj)
                # Check diag
                if M[i][j] == M[i - 1][j - 1] + s:
                    cigar.append("M")
                    i -= 1
                    j -= 1
                    state = "M"
                    continue
                if M[i][j] == I[i - 1][j - 1] + s:
                    cigar.append("M")
                    i -= 1
                    j -= 1
                    state = "I"
                    continue
                if M[i][j] == D[i - 1][j - 1] + s:
                    cigar.append("M")
                    i -= 1
                    j -= 1
                    state = "D"
                    continue

        # INSERTION (gap in B)
        if state == "I":
            # Prefer staying in insertion if scores match
            if i > 0 and I[i][j] == I[i - 1][j] + gext:
                cigar.append("D")
                i -= 1
                state = "I"
                continue
            if i > 0 and I[i][j] == M[i - 1][j] + gopen:
                cigar.append("D")
                i -= 1
                state = "M"
                continue

        # DELETION (gap in A)
        if state == "D":
            if j > 0 and D[i][j] == D[i][j - 1] + gext:
                cigar.append("I")
                j -= 1
                state = "D"
                continue
            if j > 0 and D[i][j] == M[i][j - 1] + gopen:
                cigar.append("I")
                j -= 1
                state = "M"
                continue

        # Fallback: safety checks
        raise RuntimeError(f"Traceback failure at i={i}, j={j}, state={state}")

    # reverse CIGAR
    cigar.reverse()
    return "".join(cigar)
