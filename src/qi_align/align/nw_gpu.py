# ================================================================
# nw_gpu.py
# GPU Needleman-Wunsch with affine gaps using CuPy + CUDA kernels
# ================================================================
import cupy as cp
from pathlib import Path

# Load CUDA kernels
CUDA_KERNEL_PATH = Path(__file__).with_name("cuda_kernels.cu")
CUDA_SRC = CUDA_KERNEL_PATH.read_text()

module = cp.RawModule(code=CUDA_SRC)
kernel_nw_affine = module.get_function("nw_affine_kernel")


def gpu_align_affine(seqA, seqB, score_matrix, g_open=-5, g_ext=-1, block=128):
    """
    Run NW alignment on GPU with affine gaps.
    seqA, seqB: 4-bit encoded uint8 arrays (numpy)
    score_matrix: 5x5 int32 matrix (numpy)

    Returns: DP matrix H (cupy array)
    """
    a = cp.asarray(seqA, dtype=cp.uint8)
    b = cp.asarray(seqB, dtype=cp.uint8)
    S = cp.asarray(score_matrix, dtype=cp.int32)

    n = a.size
    m = b.size

    # DP matrices
    H = cp.zeros((n+1, m+1), dtype=cp.int32)
    E = cp.zeros((n+1, m+1), dtype=cp.int32)
    F = cp.zeros((n+1, m+1), dtype=cp.int32)

    # Launch configuration
    threads = block
    blocks = ((n + threads - 1) // threads,)

    # Launch kernel
    kernel_nw_affine(
        (blocks[0],), (threads,),
        (a, b, S, H, E, F,
         n, m, g_open, g_ext)
    )

    return H
