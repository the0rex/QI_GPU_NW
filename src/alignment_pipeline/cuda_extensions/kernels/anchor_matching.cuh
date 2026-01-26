#ifndef ANCHOR_MATCHING_CUH
#define ANCHOR_MATCHING_CUH

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_anchor_matching(
    const uint64_t* seq1_hashes, const uint32_t* seq1_positions, uint32_t seq1_len,
    const uint64_t* seq2_hashes, const uint32_t* seq2_positions, uint32_t seq2_len,
    uint32_t* anchor_qpos, uint32_t* anchor_tpos, uint64_t* anchor_hashes,
    uint32_t* num_anchors, uint32_t max_occ, cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // ANCHOR_MATCHING_CUH