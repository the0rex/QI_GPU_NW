#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

/**
 * CUDA kernels for batch alignment scoring
 * Computes quick scores for multiple sequence pairs in parallel
 */

// Score matrix for DNA bases
__constant__ char cuda_score_matrix[16] = {
    /* A C G T */
    /* A */ 5, -4, -4, -4,
    /* C */ -4, 5, -4, -4,
    /* G */ -4, -4, 5, -4,
    /* T */ -4, -4, -4, 5
};

// Encode base to index
__device__ inline uint8_t base_to_idx(char base) {
    switch (base) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': case 'U': case 'u': return 3;
        default: return 0; // N or other
    }
}

// Kernel for scoring a single sequence pair
__global__ void score_pair_kernel(
    const char* seq1,
    const char* seq2,
    uint32_t len1,
    uint32_t len2,
    float* score,
    float gap_open,
    float gap_extend
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t max_len = max(len1, len2);
    
    if (idx >= max_len) return;
    
    char base1 = (idx < len1) ? seq1[idx] : '-';
    char base2 = (idx < len2) ? seq2[idx] : '-';
    
    float local_score = 0;
    
    if (base1 == '-' && base2 == '-') {
        // Both gaps - shouldn't happen in proper alignment
        local_score = 0;
    } else if (base1 == '-' || base2 == '-') {
        // Gap penalty
        local_score = (idx == 0) ? gap_open : gap_extend;
    } else {
        // Match/mismatch
        uint8_t idx1 = base_to_idx(base1);
        uint8_t idx2 = base_to_idx(base2);
        
        if (idx1 == idx2) {
            local_score = 5.0; // Match score
        } else {
            local_score = -4.0; // Mismatch score
        }
    }
    
    atomicAdd(score, local_score);
}

// Kernel for batch scoring multiple sequence pairs
__global__ void batch_score_kernel(
    const char** batch_seq1,
    const char** batch_seq2,
    const uint32_t* lengths1,
    const uint32_t* lengths2,
    uint32_t batch_size,
    float* batch_scores,
    float gap_open,
    float gap_extend
) {
    // Each block handles one sequence pair
    uint32_t pair_idx = blockIdx.x;
    
    if (pair_idx >= batch_size) return;
    
    extern __shared__ float shared_scores[];
    
    uint32_t tid = threadIdx.x;
    const char* seq1 = batch_seq1[pair_idx];
    const char* seq2 = batch_seq2[pair_idx];
    uint32_t len1 = lengths1[pair_idx];
    uint32_t len2 = lengths2[pair_idx];
    uint32_t max_len = max(len1, len2);
    
    // Initialize shared memory
    shared_scores[tid] = 0.0;
    __syncthreads();
    
    // Each thread processes multiple positions
    for (uint32_t pos = tid; pos < max_len; pos += blockDim.x) {
        char base1 = (pos < len1) ? seq1[pos] : '-';
        char base2 = (pos < len2) ? seq2[pos] : '-';
        
        float local_score = 0;
        
        if (base1 == '-' && base2 == '-') {
            local_score = 0;
        } else if (base1 == '-' || base2 == '-') {
            // Check if this continues a gap
            bool continue_gap = false;
            if (pos > 0) {
                char prev1 = (pos-1 < len1) ? seq1[pos-1] : '-';
                char prev2 = (pos-1 < len2) ? seq2[pos-1] : '-';
                if ((base1 == '-' && prev1 == '-') || 
                    (base2 == '-' && prev2 == '-')) {
                    continue_gap = true;
                }
            }
            local_score = continue_gap ? gap_extend : gap_open;
        } else {
            // Match/mismatch
            uint8_t idx1 = base_to_idx(base1);
            uint8_t idx2 = base_to_idx(base2);
            
            if (idx1 == idx2) {
                local_score = 5.0; // Match
            } else {
                local_score = -4.0; // Mismatch
            }
        }
        
        atomicAdd(&shared_scores[tid], local_score);
    }
    
    __syncthreads();
    
    // Reduce within block
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_scores[tid] += shared_scores[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final score
    if (tid == 0) {
        batch_scores[pair_idx] = shared_scores[0];
    }
}

// Kernel for quick match counting (faster but less accurate)
__global__ void quick_match_count_kernel(
    const char** batch_seq1,
    const char** batch_seq2,
    const uint32_t* lengths1,
    const uint32_t* lengths2,
    uint32_t batch_size,
    float* batch_scores
) {
    uint32_t pair_idx = blockIdx.x;
    
    if (pair_idx >= batch_size) return;
    
    const char* seq1 = batch_seq1[pair_idx];
    const char* seq2 = batch_seq2[pair_idx];
    uint32_t len = min(lengths1[pair_idx], lengths2[pair_idx]);
    
    uint32_t matches = 0;
    uint32_t comparisons = 0;
    
    // Each thread in block processes part of the sequence
    for (uint32_t i = threadIdx.x; i < len; i += blockDim.x) {
        if (seq1[i] == seq2[i] && seq1[i] != 'N' && seq2[i] != 'N') {
            atomicAdd(&matches, 1);
        }
        atomicAdd(&comparisons, 1);
    }
    
    __syncthreads();
    
    // Reduce matches in shared memory
    extern __shared__ uint32_t shared_matches[];
    shared_matches[threadIdx.x] = matches;
    __syncthreads();
    
    // Tree reduction
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_matches[threadIdx.x] += shared_matches[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float match_ratio = (comparisons > 0) ? 
                           (float)shared_matches[0] / comparisons : 0.0;
        batch_scores[pair_idx] = match_ratio * 100.0; // Percentage
    }
}

// Wrapper function for batch scoring
extern "C" void cuda_batch_score(
    const char** seq1_ptrs, const uint32_t* lengths1,
    const char** seq2_ptrs, const uint32_t* lengths2,
    uint32_t batch_size,
    float* scores,
    float match_score,
    float mismatch_score,
    float gap_open,
    float gap_extend,
    cudaStream_t stream
) {
    // Copy data to device
    thrust::device_vector<const char*> d_seq1_ptrs(seq1_ptrs, seq1_ptrs + batch_size);
    thrust::device_vector<const char*> d_seq2_ptrs(seq2_ptrs, seq2_ptrs + batch_size);
    thrust::device_vector<uint32_t> d_lengths1(lengths1, lengths1 + batch_size);
    thrust::device_vector<uint32_t> d_lengths2(lengths2, lengths2 + batch_size);
    thrust::device_vector<float> d_scores(batch_size, 0.0);
    
    // Launch kernel
    uint32_t block_size = 256;
    uint32_t grid_size = batch_size;
    uint32_t shared_mem = block_size * sizeof(float);
    
    batch_score_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        thrust::raw_pointer_cast(d_seq1_ptrs.data()),
        thrust::raw_pointer_cast(d_seq2_ptrs.data()),
        thrust::raw_pointer_cast(d_lengths1.data()),
        thrust::raw_pointer_cast(d_lengths2.data()),
        batch_size,
        thrust::raw_pointer_cast(d_scores.data()),
        gap_open,
        gap_extend
    );
    
    // Copy results back
    thrust::copy(d_scores.begin(), d_scores.end(), scores);
}

// Quick scoring wrapper (faster)
extern "C" void cuda_quick_batch_score(
    const char** seq1_ptrs, const uint32_t* lengths1,
    const char** seq2_ptrs, const uint32_t* lengths2,
    uint32_t batch_size,
    float* scores,
    cudaStream_t stream
) {
    thrust::device_vector<const char*> d_seq1_ptrs(seq1_ptrs, seq1_ptrs + batch_size);
    thrust::device_vector<const char*> d_seq2_ptrs(seq2_ptrs, seq2_ptrs + batch_size);
    thrust::device_vector<uint32_t> d_lengths1(lengths1, lengths1 + batch_size);
    thrust::device_vector<uint32_t> d_lengths2(lengths2, lengths2 + batch_size);
    thrust::device_vector<float> d_scores(batch_size, 0.0);
    
    uint32_t block_size = 128;
    uint32_t grid_size = batch_size;
    uint32_t shared_mem = block_size * sizeof(uint32_t);
    
    quick_match_count_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        thrust::raw_pointer_cast(d_seq1_ptrs.data()),
        thrust::raw_pointer_cast(d_seq2_ptrs.data()),
        thrust::raw_pointer_cast(d_lengths1.data()),
        thrust::raw_pointer_cast(d_lengths2.data()),
        batch_size,
        thrust::raw_pointer_cast(d_scores.data())
    );
    
    thrust::copy(d_scores.begin(), d_scores.end(), scores);
}