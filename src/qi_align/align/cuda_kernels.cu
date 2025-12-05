// ================================================================
// cuda_kernels.cu
// Anti-diagonal Wavefront Needleman-Wunsch (Affine Gap)
// Optimized for 4-bit DNA and large genome chunks.
// ================================================================

extern "C" __global__
void nw_affine_kernel(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const int* __restrict__ S,
    int* __restrict__ H,
    int* __restrict__ E,
    int* __restrict__ F,
    const int n, const int m,
    const int gopen, const int gext
){
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int i_start = bid * blockDim.x + tid + 1;
    if(i_start > n) return;

    // For each anti-diagonal
    for(int j = 1; j <= m; j++){
        int i = i_start;
        if(i > n) break;

        // Load bases
        int ai = (int)A[i-1];
        int bj = (int)B[j-1];
        int sc = S[ai*5 + bj];

        // DP transitions
        int h_diag = H[(i-1)*(m+1) + (j-1)] + sc;
        int e_up   = E[(i-1)*(m+1) + j];
        int f_left = F[i*(m+1) + (j-1)];

        // Update E (vertical gap)
        int e1 = H[(i-1)*(m+1) + j] + gopen;
        int e2 = e_up + gext;
        int e_new = max(e1, e2);

        // Update F (horizontal gap)
        int f1 = H[i*(m+1) + (j-1)] + gopen;
        int f2 = f_left + gext;
        int f_new = max(f1, f2);

        // Final H
        int h_new = max(h_diag, max(e_new, f_new));

        H[i*(m+1)+j] = h_new;
        E[i*(m+1)+j] = e_new;
        F[i*(m+1)+j] = f_new;
    }
}
