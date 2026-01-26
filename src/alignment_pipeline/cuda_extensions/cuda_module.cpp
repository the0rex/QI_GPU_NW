#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "kernels/anchor_matching.cuh"
#include "kernels/beam_pruning.cuh"
#include "kernels/alignment_scoring.cuh"
#include "kernels/sequence_encoding.cuh"

namespace py = pybind11;

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + \
                cudaGetErrorString(err)); \
        } \
    } while(0)

// Anchor matching wrapper
py::tuple cuda_anchor_matching_wrapper(
    py::array_t<uint64_t> seq1_hashes,
    py::array_t<uint32_t> seq1_positions,
    py::array_t<uint64_t> seq2_hashes,
    py::array_t<uint32_t> seq2_positions,
    uint32_t max_occ
) {
    // Get array info
    auto seq1_hashes_buf = seq1_hashes.request();
    auto seq1_positions_buf = seq1_positions.request();
    auto seq2_hashes_buf = seq2_hashes.request();
    auto seq2_positions_buf = seq2_positions.request();
    
    uint32_t seq1_len = seq1_hashes_buf.shape[0];
    uint32_t seq2_len = seq2_hashes_buf.shape[0];
    
    // Create output arrays
    uint32_t max_anchors = seq1_len * 10; // Upper bound
    auto anchor_qpos = py::array_t<uint32_t>(max_anchors);
    auto anchor_tpos = py::array_t<uint32_t>(max_anchors);
    auto anchor_hashes = py::array_t<uint64_t>(max_anchors);
    uint32_t num_anchors = 0;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Call CUDA function
    cuda_anchor_matching(
        static_cast<uint64_t*>(seq1_hashes_buf.ptr),
        static_cast<uint32_t*>(seq1_positions_buf.ptr),
        seq1_len,
        static_cast<uint64_t*>(seq2_hashes_buf.ptr),
        static_cast<uint32_t*>(seq2_positions_buf.ptr),
        seq2_len,
        static_cast<uint32_t*>(anchor_qpos.request().ptr),
        static_cast<uint32_t*>(anchor_tpos.request().ptr),
        static_cast<uint64_t*>(anchor_hashes.request().ptr),
        &num_anchors,
        max_occ,
        stream
    );
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    // Resize output arrays to actual number of anchors
    auto anchor_qpos_slice = anchor_qpos[py::slice(0, num_anchors, 1)];
    auto anchor_tpos_slice = anchor_tpos[py::slice(0, num_anchors, 1)];
    auto anchor_hashes_slice = anchor_hashes[py::slice(0, num_anchors, 1)];
    
    return py::make_tuple(
        anchor_qpos_slice,
        anchor_tpos_slice,
        anchor_hashes_slice,
        num_anchors
    );
}

// Beam pruning wrapper
py::tuple cuda_beam_prune_wrapper(
    py::array_t<float> log_scores,
    py::array_t<float> energies,
    py::array_t<int32_t> prev_i,
    py::array_t<int32_t> prev_j,
    py::array_t<int8_t> moves,
    py::array_t<int32_t> gap_lengths,
    uint32_t beam_width
) {
    auto log_scores_buf = log_scores.request();
    auto energies_buf = energies.request();
    auto prev_i_buf = prev_i.request();
    auto prev_j_buf = prev_j.request();
    auto moves_buf = moves.request();
    auto gap_lengths_buf = gap_lengths.request();
    
    uint32_t beam_size = log_scores_buf.shape[0];
    
    // Create output arrays
    auto pruned_log_scores = py::array_t<float>(beam_width);
    auto pruned_energies = py::array_t<float>(beam_width);
    auto pruned_prev_i = py::array_t<int32_t>(beam_width);
    auto pruned_prev_j = py::array_t<int32_t>(beam_width);
    auto pruned_moves = py::array_t<int8_t>(beam_width);
    auto pruned_gap_lengths = py::array_t<int32_t>(beam_width);
    uint32_t pruned_count = 0;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Get temporary storage size
    size_t temp_storage_bytes = cuda_beam_prune_temp_storage(beam_size, beam_width);
    
    // Allocate temporary storage
    void* d_temp_storage;
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Call CUDA function
    cuda_beam_prune(
        static_cast<float*>(log_scores_buf.ptr),
        static_cast<float*>(energies_buf.ptr),
        static_cast<int32_t*>(prev_i_buf.ptr),
        static_cast<int32_t*>(prev_j_buf.ptr),
        static_cast<int8_t*>(moves_buf.ptr),
        static_cast<int32_t*>(gap_lengths_buf.ptr),
        beam_size,
        beam_width,
        static_cast<float*>(pruned_log_scores.request().ptr),
        static_cast<float*>(pruned_energies.request().ptr),
        static_cast<int32_t*>(pruned_prev_i.request().ptr),
        static_cast<int32_t*>(pruned_prev_j.request().ptr),
        static_cast<int8_t*>(pruned_moves.request().ptr),
        static_cast<int32_t*>(pruned_gap_lengths.request().ptr),
        &pruned_count,
        d_temp_storage,
        temp_storage_bytes,
        stream
    );
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    // Resize output arrays
    auto pruned_log_scores_slice = pruned_log_scores[py::slice(0, pruned_count, 1)];
    auto pruned_energies_slice = pruned_energies[py::slice(0, pruned_count, 1)];
    auto pruned_prev_i_slice = pruned_prev_i[py::slice(0, pruned_count, 1)];
    auto pruned_prev_j_slice = pruned_prev_j[py::slice(0, pruned_count, 1)];
    auto pruned_moves_slice = pruned_moves[py::slice(0, pruned_count, 1)];
    auto pruned_gap_lengths_slice = pruned_gap_lengths[py::slice(0, pruned_count, 1)];
    
    return py::make_tuple(
        pruned_log_scores_slice,
        pruned_energies_slice,
        pruned_prev_i_slice,
        pruned_prev_j_slice,
        pruned_moves_slice,
        pruned_gap_lengths_slice,
        pruned_count
    );
}

// Batch scoring wrapper
py::array_t<float> cuda_batch_score_wrapper(
    py::list seq1_list,
    py::list seq2_list,
    float match_score,
    float mismatch_score,
    float gap_open,
    float gap_extend
) {
    uint32_t batch_size = seq1_list.size();
    
    // Convert Python strings to C strings
    std::vector<const char*> seq1_ptrs(batch_size);
    std::vector<const char*> seq2_ptrs(batch_size);
    std::vector<uint32_t> lengths1(batch_size);
    std::vector<uint32_t> lengths2(batch_size);
    std::vector<std::string> seq1_strings(batch_size);
    std::vector<std::string> seq2_strings(batch_size);
    
    for (uint32_t i = 0; i < batch_size; i++) {
        seq1_strings[i] = seq1_list[i].cast<std::string>();
        seq2_strings[i] = seq2_list[i].cast<std::string>();
        
        seq1_ptrs[i] = seq1_strings[i].c_str();
        seq2_ptrs[i] = seq2_strings[i].c_str();
        
        lengths1[i] = seq1_strings[i].length();
        lengths2[i] = seq2_strings[i].length();
    }
    
    // Create output array
    auto scores = py::array_t<float>(batch_size);
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // Call CUDA function
    cuda_batch_score(
        seq1_ptrs.data(), lengths1.data(),
        seq2_ptrs.data(), lengths2.data(),
        batch_size,
        static_cast<float*>(scores.request().ptr),
        match_score, mismatch_score,
        gap_open, gap_extend,
        stream
    );
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    return scores;
}

// Quick batch scoring wrapper
py::array_t<float> cuda_quick_batch_score_wrapper(
    py::list seq1_list,
    py::list seq2_list
) {
    uint32_t batch_size = seq1_list.size();
    
    std::vector<const char*> seq1_ptrs(batch_size);
    std::vector<const char*> seq2_ptrs(batch_size);
    std::vector<uint32_t> lengths1(batch_size);
    std::vector<uint32_t> lengths2(batch_size);
    std::vector<std::string> seq1_strings(batch_size);
    std::vector<std::string> seq2_strings(batch_size);
    
    for (uint32_t i = 0; i < batch_size; i++) {
        seq1_strings[i] = seq1_list[i].cast<std::string>();
        seq2_strings[i] = seq2_list[i].cast<std::string>();
        
        seq1_ptrs[i] = seq1_strings[i].c_str();
        seq2_ptrs[i] = seq2_strings[i].c_str();
        
        lengths1[i] = seq1_strings[i].length();
        lengths2[i] = seq2_strings[i].length();
    }
    
    auto scores = py::array_t<float>(batch_size);
    
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    cuda_quick_batch_score(
        seq1_ptrs.data(), lengths1.data(),
        seq2_ptrs.data(), lengths2.data(),
        batch_size,
        static_cast<float*>(scores.request().ptr),
        stream
    );
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    return scores;
}

// Sequence encoding wrapper
py::array_t<float> cuda_one_hot_encode_wrapper(
    py::array_t<char> sequences,
    uint32_t seq_len
) {
    auto sequences_buf = sequences.request();
    uint32_t batch_size = sequences_buf.shape[0] / seq_len;
    
    // Create output array
    uint32_t encoded_size = batch_size * seq_len * 4;
    auto encoded = py::array_t<float>(encoded_size);
    
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    cuda_one_hot_encode(
        static_cast<char*>(sequences_buf.ptr),
        seq_len,
        batch_size,
        static_cast<float*>(encoded.request().ptr),
        stream
    );
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    // Reshape to (batch_size, seq_len, 4)
    encoded.resize({batch_size, seq_len, 4});
    return encoded;
}

// Module definition
PYBIND11_MODULE(_cuda_kernels, m) {
    m.doc() = "CUDA kernels for alignment pipeline acceleration";
    
    m.def("anchor_matching", &cuda_anchor_matching_wrapper,
          "GPU-accelerated anchor matching");
    
    m.def("beam_prune", &cuda_beam_prune_wrapper,
          "GPU-accelerated beam pruning");
    
    m.def("batch_score", &cuda_batch_score_wrapper,
          "Batch alignment scoring on GPU");
    
    m.def("quick_batch_score", &cuda_quick_batch_score_wrapper,
          "Quick batch scoring on GPU");
    
    m.def("one_hot_encode", &cuda_one_hot_encode_wrapper,
          "One-hot encode DNA sequences on GPU");
    
    m.def("check_cuda", []() -> bool {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }, "Check if CUDA is available");
    
    m.def("get_cuda_info", []() -> py::dict {
        py::dict info;
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        info["device_count"] = device_count;
        
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            
            info["device_name"] = std::string(prop.name);
            info["compute_capability"] = std::to_string(prop.major) + "." + 
                                         std::to_string(prop.minor);
            info["total_memory_mb"] = prop.totalGlobalMem / (1024 * 1024);
            info["multi_processor_count"] = prop.multiProcessorCount;
        }
        
        return info;
    }, "Get CUDA device information");
}