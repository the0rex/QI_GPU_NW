#ifndef ALIGNMENT_SCORING_CUH
#define ALIGNMENT_SCORING_CUH

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_batch_score(
    const char** seq1_ptrs, const uint32_t* lengths1,
    const char** seq2_ptrs, const uint32_t* lengths2,
    uint32_t batch_size,
    float* scores,
    float match_score,
    float mismatch_score,
    float gap_open,
    float gap_extend,
    cudaStream_t stream
);

void cuda_quick_batch_score(
    const char** seq1_ptrs, const uint32_t* lengths1,
    const char** seq2_ptrs, const uint32_t* lengths2,
    uint32_t batch_size,
    float* scores,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // ALIGNMENT_SCORING_CUH