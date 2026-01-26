#ifndef SEQUENCE_ENCODING_CUH
#define SEQUENCE_ENCODING_CUH

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_one_hot_encode(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    float* encoded,
    cudaStream_t stream
);

void cuda_packed_4bit_encode(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    uint8_t* encoded,
    cudaStream_t stream
);

void cuda_sequence_similarity(
    const char* seq1,
    const char* seq2,
    uint32_t len,
    float* similarity,
    cudaStream_t stream
);

void cuda_batch_gc_content(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    float* gc_contents,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // SEQUENCE_ENCODING_CUH