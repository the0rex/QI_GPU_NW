# ================================================================
# hybrid_align.py
# Hybrid NW (GPU) + WFA gap-fill alignment
# ================================================================
from .nw_gpu import gpu_align_affine
from .wfa_gapfill import wfa_gapfill
from qi_align.encoding.restore import decode_4bit_to_ascii
import cupy as cp
from fnn.predictor import predict

def _traceback(H, seqA, seqB, S, gopen, gext):
    """
    Traceback from bottom-right to top-left using H matrix.
    WARNING: H is on GPU. Transfer slices as needed.
    """
    Hcpu = H.get()
    i = Hcpu.shape[0]-1
    j = Hcpu.shape[1]-1
    cigar = []

    while i>0 or j>0:
        if i>0 and j>0:
            ai = seqA[i-1]
            bj = seqB[j-1]
            sc = S[ai, bj]

            if Hcpu[i,j] == Hcpu[i-1,j-1] + sc:
                cigar.append("M")
                i-=1; j-=1
                continue

        if i>0:
            # vertical gap
            if Hcpu[i,j] == Hcpu[i-1,j] + gext or Hcpu[i,j] == Hcpu[i-1,j] + gopen:
                cigar.append("D")
                i-=1
                continue

        if j>0:
            # horizontal gap
            if Hcpu[i,j] == Hcpu[i,j-1] + gext or Hcpu[i,j] == Hcpu[i,j-1] + gopen:
                cigar.append("I")
                j-=1
                continue

        raise RuntimeError("Traceback ambiguity at i=%d j=%d" % (i,j))

    return "".join(reversed(cigar))


def hybrid_global_align(seqA, seqB, score_matrix, gopen=-5, gext=-1):
    """
    Perform full GPU NW alignment on one chunk.
    """
    
    pred = predict(seqA[:500])  # look at beginning of chunk
    d_match, d_mismatch, d_gamma = pred["score_adj"]
    
    score_matrix = build_qi_matrix(
        match + d_match,
        mismatch + d_mismatch,
        ambiguous,
        gamma + d_gamma
    )
    # Convert score matrix to GPU
    S = cp.asarray(score_matrix, dtype=cp.int32)

    # Run GPU NW
    H = gpu_align_affine(seqA, seqB, score_matrix, gopen, gext)

    # Traceback (CPU)
    cigar = _traceback(H, seqA, seqB, score_matrix, gopen, gext)

    # Post-process large gaps with WFA
    # (Simplified; in full pipeline stitcher handles gap filling)
    return cigar
