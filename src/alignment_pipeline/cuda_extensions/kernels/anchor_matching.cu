#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <iostream>

/**
 * CUDA kernel for batch anchor matching
 * Matches strobe hashes between two sequences on GPU
 */

// Kernel for building hash index
__global__ void build_hash_index_kernel(
    const uint64_t* hashes,
    const uint32_t* positions,
    uint32_t num_items,
    uint64_t* unique_hashes,
    uint32_t* hash_counts,
    uint32_t* hash_offsets,
    uint32_t* position_lists,
    uint32_t max_occ
) {
    extern __shared__ uint32_t shared_mem[];
    uint32_t* shared_counts = shared_mem;
    uint32_t* shared_offsets = &shared_mem[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t idx = bid * blockDim.x + tid;
    
    // Initialize shared memory
    if (tid < 256) {
        shared_counts[tid] = 0;
        shared_offsets[tid] = 0;
    }
    __syncthreads();
    
    if (idx < num_items) {
        uint64_t hash = hashes[idx];
        uint32_t hash_bucket = hash % 256;
        
        // Atomic increment for hash bucket count
        atomicAdd(&shared_counts[hash_bucket], 1);
    }
    __syncthreads();
    
    // Write back to global memory
    if (tid < 256) {
        atomicAdd(&hash_counts[hash_bucket], shared_counts[hash_bucket]);
    }
}

// Kernel for finding hash matches
__global__ void find_hash_matches_kernel(
    const uint64_t* seq1_hashes,
    const uint32_t* seq1_positions,
    uint32_t seq1_len,
    const uint64_t* seq2_hashes_sorted,
    const uint32_t* seq2_positions_sorted,
    uint32_t seq2_len,
    uint32_t* match_counts,
    uint32_t* match_indices
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= seq1_len) return;
    
    uint64_t target_hash = seq1_hashes[idx];
    
    // Binary search in sorted seq2 hashes
    uint32_t* begin = (uint32_t*)seq2_hashes_sorted;
    uint32_t* end = begin + seq2_len;
    
    // Use thrust-like binary search (simplified)
    uint32_t left = 0;
    uint32_t right = seq2_len;
    uint32_t count = 0;
    
    // Find first occurrence
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (seq2_hashes_sorted[mid] < target_hash) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    uint32_t first = left;
    
    // Find last occurrence
    right = seq2_len;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (seq2_hashes_sorted[mid] <= target_hash) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    uint32_t last = left;
    count = last - first;
    
    match_counts[idx] = count;
    
    // Store match indices
    for (uint32_t i = 0; i < count && i < 10; i++) { // Limit to 10 matches per hash
        match_indices[idx * 10 + i] = first + i;
    }
}

// Kernel for generating anchors from matches
__global__ void generate_anchors_kernel(
    const uint64_t* seq1_hashes,
    const uint32_t* seq1_positions,
    const uint32_t* match_counts,
    const uint32_t* match_indices,
    const uint64_t* seq2_hashes,
    const uint32_t* seq2_positions,
    uint32_t seq1_len,
    uint32_t* anchor_qpos,
    uint32_t* anchor_tpos,
    uint64_t* anchor_hashes,
    uint32_t* anchor_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= seq1_len) return;
    
    uint32_t matches = match_counts[idx];
    if (matches == 0) return;
    
    uint32_t base_offset = atomicAdd(anchor_count, matches);
    
    for (uint32_t i = 0; i < matches; i++) {
        uint32_t match_idx = match_indices[idx * 10 + i];
        
        anchor_qpos[base_offset + i] = seq1_positions[idx];
        anchor_tpos[base_offset + i] = seq2_positions[match_idx];
        anchor_hashes[base_offset + i] = seq1_hashes[idx];
    }
}

// Wrapper function for anchor matching
extern "C" void cuda_anchor_matching(
    const uint64_t* seq1_hashes, const uint32_t* seq1_positions, uint32_t seq1_len,
    const uint64_t* seq2_hashes, const uint32_t* seq2_positions, uint32_t seq2_len,
    uint32_t* anchor_qpos, uint32_t* anchor_tpos, uint64_t* anchor_hashes,
    uint32_t* num_anchors, uint32_t max_occ, cudaStream_t stream
) {
    // Device vectors
    thrust::device_vector<uint64_t> d_seq1_hashes(seq1_hashes, seq1_hashes + seq1_len);
    thrust::device_vector<uint32_t> d_seq1_positions(seq1_positions, seq1_positions + seq1_len);
    thrust::device_vector<uint64_t> d_seq2_hashes(seq2_hashes, seq2_hashes + seq2_len);
    thrust::device_vector<uint32_t> d_seq2_positions(seq2_positions, seq2_positions + seq2_len);
    
    // Sort seq2 hashes for fast lookup
    thrust::sort_by_key(d_seq2_hashes.begin(), d_seq2_hashes.end(), d_seq2_positions.begin());
    
    // Temporary arrays for matches
    thrust::device_vector<uint32_t> d_match_counts(seq1_len, 0);
    thrust::device_vector<uint32_t> d_match_indices(seq1_len * 10, 0);
    thrust::device_vector<uint32_t> d_anchor_count(1, 0);
    
    // Launch kernels
    uint32_t block_size = 256;
    uint32_t grid_size = (seq1_len + block_size - 1) / block_size;
    
    // Find matches
    find_hash_matches_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_seq1_hashes.data()),
        thrust::raw_pointer_cast(d_seq1_positions.data()),
        seq1_len,
        thrust::raw_pointer_cast(d_seq2_hashes.data()),
        thrust::raw_pointer_cast(d_seq2_positions.data()),
        seq2_len,
        thrust::raw_pointer_cast(d_match_counts.data()),
        thrust::raw_pointer_cast(d_match_indices.data())
    );
    
    // Generate anchors
    uint32_t max_possible_anchors = seq1_len * 10; // Upper bound
    thrust::device_vector<uint32_t> d_anchor_qpos(max_possible_anchors);
    thrust::device_vector<uint32_t> d_anchor_tpos(max_possible_anchors);
    thrust::device_vector<uint64_t> d_anchor_hashes(max_possible_anchors);
    
    generate_anchors_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(d_seq1_hashes.data()),
        thrust::raw_pointer_cast(d_seq1_positions.data()),
        thrust::raw_pointer_cast(d_match_counts.data()),
        thrust::raw_pointer_cast(d_match_indices.data()),
        thrust::raw_pointer_cast(d_seq2_hashes.data()),
        thrust::raw_pointer_cast(d_seq2_positions.data()),
        seq1_len,
        thrust::raw_pointer_cast(d_anchor_qpos.data()),
        thrust::raw_pointer_cast(d_anchor_tpos.data()),
        thrust::raw_pointer_cast(d_anchor_hashes.data()),
        thrust::raw_pointer_cast(d_anchor_count.data())
    );
    
    // Copy results back
    uint32_t count;
    cudaMemcpyAsync(&count, thrust::raw_pointer_cast(d_anchor_count.data()), 
                   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (count > 0) {
        cudaMemcpyAsync(anchor_qpos, thrust::raw_pointer_cast(d_anchor_qpos.data()),
                       count * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(anchor_tpos, thrust::raw_pointer_cast(d_anchor_tpos.data()),
                       count * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(anchor_hashes, thrust::raw_pointer_cast(d_anchor_hashes.data()),
                       count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    }
    
    *num_anchors = count;
}