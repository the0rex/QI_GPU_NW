#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

/**
 * CUDA kernels for DNA sequence encoding and processing
 * Converts ASCII sequences to encoded formats for efficient processing
 */

// One-hot encoding: A=1000, C=0100, G=0010, T=0001
__global__ void one_hot_encode_kernel(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    float* encoded,
    uint32_t encoded_stride
) {
    uint32_t batch_idx = blockIdx.y;
    uint32_t pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || pos_idx >= seq_len) return;
    
    uint32_t seq_offset = batch_idx * seq_len;
    uint32_t encode_offset = batch_idx * encoded_stride + pos_idx * 4;
    
    char base = sequences[seq_offset + pos_idx];
    
    switch (base) {
        case 'A': case 'a':
            encoded[encode_offset + 0] = 1.0;
            encoded[encode_offset + 1] = 0.0;
            encoded[encode_offset + 2] = 0.0;
            encoded[encode_offset + 3] = 0.0;
            break;
        case 'C': case 'c':
            encoded[encode_offset + 0] = 0.0;
            encoded[encode_offset + 1] = 1.0;
            encoded[encode_offset + 2] = 0.0;
            encoded[encode_offset + 3] = 0.0;
            break;
        case 'G': case 'g':
            encoded[encode_offset + 0] = 0.0;
            encoded[encode_offset + 1] = 0.0;
            encoded[encode_offset + 2] = 1.0;
            encoded[encode_offset + 3] = 0.0;
            break;
        case 'T': case 't': case 'U': case 'u':
            encoded[encode_offset + 0] = 0.0;
            encoded[encode_offset + 1] = 0.0;
            encoded[encode_offset + 2] = 0.0;
            encoded[encode_offset + 3] = 1.0;
            break;
        default: // N or other
            encoded[encode_offset + 0] = 0.25;
            encoded[encode_offset + 1] = 0.25;
            encoded[encode_offset + 2] = 0.25;
            encoded[encode_offset + 3] = 0.25;
            break;
    }
}

// 4-bit encoding (packed)
__global__ void packed_4bit_encode_kernel(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    uint8_t* encoded,
    uint32_t encoded_stride
) {
    uint32_t batch_idx = blockIdx.y;
    uint32_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    uint32_t seq_offset = batch_idx * seq_len;
    uint32_t encode_offset = batch_idx * encoded_stride;
    
    // Each byte stores two bases
    uint32_t base_pos = byte_idx * 2;
    
    if (base_pos >= seq_len) return;
    
    char base1 = (base_pos < seq_len) ? sequences[seq_offset + base_pos] : 'N';
    char base2 = (base_pos + 1 < seq_len) ? sequences[seq_offset + base_pos + 1] : 'N';
    
    uint8_t code1 = 0, code2 = 0;
    
    // Encode first base
    switch (base1) {
        case 'A': case 'a': code1 = 0x1; break;
        case 'C': case 'c': code1 = 0x2; break;
        case 'G': case 'g': code1 = 0x4; break;
        case 'T': case 't': case 'U': case 'u': code1 = 0x8; break;
        default: code1 = 0xF; break; // N
    }
    
    // Encode second base
    switch (base2) {
        case 'A': case 'a': code2 = 0x1; break;
        case 'C': case 'c': code2 = 0x2; break;
        case 'G': case 'g': code2 = 0x4; break;
        case 'T': case 't': case 'U': case 'u': code2 = 0x8; break;
        default: code2 = 0xF; break; // N
    }
    
    encoded[encode_offset + byte_idx] = (code1 << 4) | code2;
}

// Kernel for sequence similarity calculation
__global__ void sequence_similarity_kernel(
    const char* seq1,
    const char* seq2,
    uint32_t len,
    float* similarity
) {
    __shared__ uint32_t shared_matches[256];
    __shared__ uint32_t shared_total[256];
    
    uint32_t tid = threadIdx.x;
    shared_matches[tid] = 0;
    shared_total[tid] = 0;
    
    // Each thread processes multiple positions
    for (uint32_t i = tid; i < len; i += blockDim.x) {
        if (seq1[i] == seq2[i] && seq1[i] != 'N' && seq2[i] != 'N') {
            shared_matches[tid]++;
        }
        shared_total[tid]++;
    }
    
    __syncthreads();
    
    // Reduce within block
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_matches[tid] += shared_matches[tid + stride];
            shared_total[tid] += shared_total[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        if (shared_total[0] > 0) {
            *similarity = (float)shared_matches[0] / shared_total[0];
        } else {
            *similarity = 0.0;
        }
    }
}

// Kernel for GC content calculation
__global__ void gc_content_kernel(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    float* gc_contents
) {
    uint32_t batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    extern __shared__ uint32_t shared_counts[];
    uint32_t* gc_count = &shared_counts[0];
    uint32_t* total_count = &shared_counts[1];
    
    if (threadIdx.x == 0) {
        *gc_count = 0;
        *total_count = 0;
    }
    __syncthreads();
    
    uint32_t seq_offset = batch_idx * seq_len;
    
    // Each thread processes part of the sequence
    for (uint32_t i = threadIdx.x; i < seq_len; i += blockDim.x) {
        char base = sequences[seq_offset + i];
        if (base == 'G' || base == 'C' || base == 'g' || base == 'c') {
            atomicAdd(gc_count, 1);
        }
        if (base != 'N' && base != 'n') {
            atomicAdd(total_count, 1);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        if (*total_count > 0) {
            gc_contents[batch_idx] = (float)(*gc_count) / (*total_count);
        } else {
            gc_contents[batch_idx] = 0.0;
        }
    }
}

// Wrapper functions
extern "C" void cuda_one_hot_encode(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    float* encoded,
    cudaStream_t stream
) {
    dim3 block_size(256);
    dim3 grid_size((seq_len + block_size.x - 1) / block_size.x, batch_size);
    
    uint32_t encoded_stride = seq_len * 4;
    
    one_hot_encode_kernel<<<grid_size, block_size, 0, stream>>>(
        sequences, seq_len, batch_size, encoded, encoded_stride
    );
}

extern "C" void cuda_packed_4bit_encode(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    uint8_t* encoded,
    cudaStream_t stream
) {
    uint32_t bytes_needed = (seq_len + 1) / 2;
    dim3 block_size(256);
    dim3 grid_size((bytes_needed + block_size.x - 1) / block_size.x, batch_size);
    
    uint32_t encoded_stride = bytes_needed;
    
    packed_4bit_encode_kernel<<<grid_size, block_size, 0, stream>>>(
        sequences, seq_len, batch_size, encoded, encoded_stride
    );
}

extern "C" void cuda_sequence_similarity(
    const char* seq1,
    const char* seq2,
    uint32_t len,
    float* similarity,
    cudaStream_t stream
) {
    uint32_t block_size = 256;
    uint32_t grid_size = 1;
    
    sequence_similarity_kernel<<<grid_size, block_size, 0, stream>>>(
        seq1, seq2, len, similarity
    );
}

extern "C" void cuda_batch_gc_content(
    const char* sequences,
    uint32_t seq_len,
    uint32_t batch_size,
    float* gc_contents,
    cudaStream_t stream
) {
    uint32_t block_size = 256;
    uint32_t grid_size = batch_size;
    uint32_t shared_mem = 2 * sizeof(uint32_t); // For gc_count and total_count
    
    gc_content_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        sequences, seq_len, batch_size, gc_contents
    );
}