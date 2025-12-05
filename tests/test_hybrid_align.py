import numpy as np
from qi_align.align.hybrid_align import hybrid_global_align
from qi_align.align.qi_score import build_qi_matrix

def test_hybrid_align_small():
    A = np.array([0,1,2], dtype=np.uint8)
    B = np.array([0,1,2], dtype=np.uint8)
    S = build_qi_matrix()
    cigar = hybrid_global_align(A, B, S)
    assert "M" in cigar
