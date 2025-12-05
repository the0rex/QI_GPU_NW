import numpy as np
from qi_align.align.nw_gpu import gpu_align_affine
from qi_align.align.qi_score import build_qi_matrix

def test_gpu_small_alignment():
    A = np.array([0,1,2], dtype=np.uint8)  # A C G
    B = np.array([0,1,2], dtype=np.uint8)
    S = build_qi_matrix()
    H = gpu_align_affine(A, B, S)
    assert H.shape == (4,4)
