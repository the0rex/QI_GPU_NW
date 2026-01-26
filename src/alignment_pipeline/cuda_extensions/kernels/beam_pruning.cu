#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <cub/cub.cuh>

/**
 * CUDA kernel for beam pruning
 * Implements top-k selection for beam search on GPU
 */

// Kernel for parallel reduction to find top-k
template<typename T>
__global__ void top_k_kernel(
    const T* scores,
    uint32_t* indices,
    uint32_t n,
    uint32_t k,
    T* top_scores,
    uint32_t* top_indices
) {
    extern __shared__ uint8_t shared_mem[];
    
    // Each thread handles multiple elements
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    uint32_t grid_size = gridDim.x;
    
    // Local top-k for this thread
    T local_top_k[32];
    uint32_t local_indices[32];
    
    for (uint32_t i = 0; i < 32; i++) {
        local_top_k[i] = -INFINITY;
        local_indices[i] = 0;
    }
    
    // Process elements
    for (uint32_t i = bid * block_size + tid; i < n; i += grid_size * block_size) {
        T score = scores[i];
        uint32_t idx = indices ? indices[i] : i;
        
        // Insert into local top-k (bubble down)
        if (score > local_top_k[31]) {
            local_top_k[31] = score;
            local_indices[31] = idx;
            
            for (int j = 31; j > 0; j--) {
                if (local_top_k[j] > local_top_k[j-1]) {
                    T temp_score = local_top_k[j];
                    uint32_t temp_idx = local_indices[j];
                    local_top_k[j] = local_top_k[j-1];
                    local_indices[j] = local_indices[j-1];
                    local_top_k[j-1] = temp_score;
                    local_indices[j-1] = temp_idx;
                } else {
                    break;
                }
            }
        }
    }
    
    // Merge local top-k results in shared memory
    // ... (implementation of parallel merge)
}

// Specialized kernel for beam pruning with log scores
__global__ void beam_prune_kernel(
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
    uint32_t* pruned_count
) {
    __shared__ float shared_scores[1024];
    __shared__ uint32_t shared_indices[1024];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    
    // Load scores into shared memory
    if (tid < beam_size) {
        shared_scores[tid] = log_scores[tid];
        shared_indices[tid] = tid;
    }
    __syncthreads();
    
    // Bitonic sort in shared memory (simplified top-k)
    for (uint32_t k = 2; k <= 1024; k *= 2) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = tid; i < 1024; i += blockDim.x) {
                uint32_t ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        // Ascending
                        if (shared_scores[i] < shared_scores[ixj]) {
                            float temp_score = shared_scores[i];
                            uint32_t temp_idx = shared_indices[i];
                            shared_scores[i] = shared_scores[ixj];
                            shared_indices[i] = shared_indices[ixj];
                            shared_scores[ixj] = temp_score;
                            shared_indices[ixj] = temp_idx;
                        }
                    } else {
                        // Descending
                        if (shared_scores[i] > shared_scores[ixj]) {
                            float temp_score = shared_scores[i];
                            uint32_t temp_idx = shared_indices[i];
                            shared_scores[i] = shared_scores[ixj];
                            shared_indices[i] = shared_indices[ixj];
                            shared_scores[ixj] = temp_score;
                            shared_indices[ixj] = temp_idx;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Write top-k results
    if (tid < beam_width && tid < beam_size) {
        uint32_t original_idx = shared_indices[tid];
        
        pruned_log_scores[tid] = log_scores[original_idx];
        pruned_energies[tid] = energies[original_idx];
        pruned_prev_i[tid] = prev_i[original_idx];
        pruned_prev_j[tid] = prev_j[original_idx];
        pruned_moves[tid] = moves[original_idx];
        pruned_gap_lengths[tid] = gap_lengths[original_idx];
    }
    
    if (tid == 0) {
        *pruned_count = min(beam_width, beam_size);
    }
}

// Wrapper function using CUB library for efficient top-k
extern "C" void cuda_beam_prune(
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
) {
    // Create device vectors
    thrust::device_vector<float> d_log_scores(log_scores, log_scores + beam_size);
    thrust::device_vector<uint32_t> d_indices(beam_size);
    thrust::sequence(d_indices.begin(), d_indices.end());
    
    // Sort scores and indices together
    thrust::sort_by_key(d_log_scores.begin(), d_log_scores.end(), 
                       d_indices.begin(), thrust::greater<float>());
    
    // Copy top-k results
    uint32_t k = min(beam_width, beam_size);
    
    // Get top-k indices
    thrust::host_vector<uint32_t> h_top_indices(k);
    thrust::copy(d_indices.begin(), d_indices.begin() + k, h_top_indices.begin());
    
    // Copy corresponding data for each top index
    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = h_top_indices[i];
        
        cudaMemcpyAsync(&pruned_log_scores[i], &log_scores[idx], 
                       sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&pruned_energies[i], &energies[idx], 
                       sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&pruned_prev_i[i], &prev_i[idx], 
                       sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&pruned_prev_j[i], &prev_j[idx], 
                       sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&pruned_moves[i], &moves[idx], 
                       sizeof(char), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(&pruned_gap_lengths[i], &gap_lengths[idx], 
                       sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }
    
    *pruned_count = k;
}

// Fast top-k using CUB (more efficient)
extern "C" size_t cuda_beam_prune_temp_storage(
    uint32_t beam_size,
    uint32_t beam_width
) {
    // Determine temporary storage requirement for CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    thrust::device_vector<float> d_keys_in(beam_size);
    thrust::device_vector<float> d_keys_out(beam_width);
    thrust::device_vector<uint32_t> d_values_in(beam_size);
    thrust::device_vector<uint32_t> d_values_out(beam_width);
    
    // Use CUB DeviceSegmentedRadixSort
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes,
        d_keys_in.data().get(), d_keys_out.data().get(),
        d_values_in.data().get(), d_values_out.data().get(),
        beam_size, 1, 0, sizeof(float) * 8,
        (cudaStream_t)0
    );
    
    return temp_storage_bytes;
}