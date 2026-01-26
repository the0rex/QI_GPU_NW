#ifndef BEAM_PRUNING_CUH
#define BEAM_PRUNING_CUH

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_beam_prune(
    const float* log_scores,
    const float* energies,
    const int* prev_i,
    const int* prev_j,
    const char* moves,
    const int* gap_lengths,
    uint32_t beam_size,
    uint32_t beam_width,
    float* pruned_log_scores,
    float* pruned_energies,
    int* pruned_prev_i,
    int* pruned_prev_j,
    char* pruned_moves,
    int* pruned_gap_lengths,
    uint32_t* pruned_count,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream
);

size_t cuda_beam_prune_temp_storage(
    uint32_t beam_size,
    uint32_t beam_width
);

#ifdef __cplusplus
}
#endif

#endif // BEAM_PRUNING_CUH