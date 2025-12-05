# GPU Dynamic Programming (DP) in QI-ALIGN

We use:
- Anti-diagonal wavefront
- Affine gap costs (H, E, F matrices)
- CuPy + Raw CUDA kernels
- Warp-shuffle propagation
- 4-bit encoded bases

Advantages:
- Reduces global memory traffic
- Supports tiles up to 50kb × 50kb
- Each chunk fits GPU memory easily

Overall speed:
- 50–100× faster than CPU NW
- ~20× faster than C++ SIMD NW (single-threaded)