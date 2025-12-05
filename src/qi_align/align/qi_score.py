# ================================================================
# qi_score.py
# Quantum-Inspired Scoring Matrix Construction
# ================================================================
import numpy as np

BASES = {"A":0,"C":1,"G":2,"T":3,"N":4}

def build_qi_matrix(match=2, mismatch=-2, ambiguous=-1, gamma=1.2):
    """
    Generate a quantum-inspired scoring matrix.
    Output is a 5x5 numpy int32 matrix for {A,C,G,T,N}.
    """
    M = np.zeros((5,5), dtype=np.int32)

    for i in range(5):
        for j in range(5):
            if i == j:
                M[i,j] = match
            else:
                M[i,j] = mismatch

    # Add QI biological adjustments:
    for b1 in range(5):
        for b2 in range(5):
            if b1 == 4 or b2 == 4:  # N
                M[b1,b2] = ambiguous

            # C↔T transitions (CpG decay)
            if {b1,b2} == {1,3}:  # C/T
                M[b1,b2] += int(gamma * 1.0)

            # G↔A transitions
            if {b1,b2} == {2,0}:  # G/A
                M[b1,b2] += int(gamma * 0.8)

            # GC vs AT pressure bias
            if {b1,b2} <= {1,2}:  # C/G
                M[b1,b2] += int(gamma * 1.5)
            elif {b1,b2} <= {0,3}:  # A/T
                M[b1,b2] += int(gamma * 0.5)

    return M
