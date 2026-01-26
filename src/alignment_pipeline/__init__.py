"""
Core modules for the alignment pipeline with optional GPU acceleration.
"""

import os
import sys
import types

# Version info - keep at top
__version__ = "2.0.0"
__author__ = "Rowel Facunla"
__description__ = "High-performance sequence alignment pipeline with optional GPU acceleration"

# Empty placeholders for modules - will be loaded on demand
compress_4bit_sequence = None
decompress_slice = None
decompress_base = None
load_two_fasta_sequences = None
IUPAC_4BIT = None
REV_IUPAC = None
nw_affine_chunk = None
nw_affine_chunk_with_reanchor = None
nw_overlap_realign = None
detect_gap_explosion = None
process_and_save_chunks_parallel = None
reload_and_merge_chunks = None
save_chunk_result = None
load_chunk_result = None
strobes_from_4bit_buffer = None
Strobemer = None
iupac_is_match = None
iupac_partial_overlap = None
compute_alignment_stats = None
build_cigar = None
realign_overlap_and_stitch = None
write_cigar_gz = None
write_paf = None
write_maf = None
write_summary = None
IUPAC_MAP = None
base_set = None

# CUDA extensions (dummy placeholders)
CUDAStrobeProcessor = None
CUDABeamPruner = None
CUDABatchAligner = None
TorchWindowPredictor = None
batch_generate_anchors_gpu = None
compute_batch_scores_gpu = None
predict_window_gpu = None
load_torch_predictor = None
CUDA_AVAILABLE = False
PYCUDA_AVAILABLE = False
USE_CUDA = False
CUDA_EXTENSIONS_AVAILABLE = False
CUDA_SUPPORT_AVAILABLE = False
ALIGNMENT_AVAILABLE = False
SEEDING_AVAILABLE = False

__all__ = [
    # Compression functions
    'compress_4bit_sequence',
    'decompress_slice',
    'decompress_base',
    'load_two_fasta_sequences',
    'IUPAC_4BIT',
    'REV_IUPAC',
    
    # Alignment functions
    'nw_affine_chunk',
    'nw_affine_chunk_with_reanchor',
    'nw_overlap_realign',
    'detect_gap_explosion',
    
    # Chunking functions
    'process_and_save_chunks_parallel',
    'reload_and_merge_chunks',
    'save_chunk_result',
    'load_chunk_result',
    
    # Seeding functions
    'strobes_from_4bit_buffer',
    'Strobemer',
    
    # Utility functions
    'iupac_is_match',
    'iupac_partial_overlap',
    'compute_alignment_stats',
    'build_cigar',
    'realign_overlap_and_stitch',
    'write_cigar_gz',
    'write_paf',
    'write_maf',
    'write_summary',
    'IUPAC_MAP',
    'base_set',
    
    # CUDA functions (new)
    'CUDAStrobeProcessor',
    'CUDABeamPruner',
    'CUDABatchAligner',
    'TorchWindowPredictor',
    'batch_generate_anchors_gpu',
    'compute_batch_scores_gpu',
    'predict_window_gpu',
    'load_torch_predictor',
    
    # Availability flags
    'ALIGNMENT_AVAILABLE',
    'SEEDING_AVAILABLE',
    'CUDA_SUPPORT_AVAILABLE',
    'CUDA_AVAILABLE',
    'PYCUDA_AVAILABLE',
    'USE_CUDA',
    'CUDA_EXTENSIONS_AVAILABLE',
    
    # Version info
    '__version__',
    '__author__',
    '__description__'
]

def lazy_import():
    """Lazy import of all modules to avoid startup overhead."""
    global compress_4bit_sequence, decompress_slice, decompress_base, load_two_fasta_sequences
    global IUPAC_4BIT, REV_IUPAC, nw_affine_chunk, nw_affine_chunk_with_reanchor
    global nw_overlap_realign, detect_gap_explosion, process_and_save_chunks_parallel
    global reload_and_merge_chunks, save_chunk_result, load_chunk_result, strobes_from_4bit_buffer
    global Strobemer, iupac_is_match, iupac_partial_overlap, compute_alignment_stats, build_cigar
    global realign_overlap_and_stitch, write_cigar_gz, write_paf, write_maf, write_summary, IUPAC_MAP, base_set
    global CUDAStrobeProcessor, CUDABeamPruner, CUDABatchAligner, TorchWindowPredictor
    global batch_generate_anchors_gpu, compute_batch_scores_gpu, predict_window_gpu, load_torch_predictor
    global CUDA_AVAILABLE, PYCUDA_AVAILABLE, USE_CUDA, CUDA_EXTENSIONS_AVAILABLE, CUDA_SUPPORT_AVAILABLE
    global ALIGNMENT_AVAILABLE, SEEDING_AVAILABLE
    
    # Set CUDA environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Mock pycuda to prevent errors
    sys.modules['pycuda'] = types.ModuleType('pycuda')
    sys.modules['pycuda'].autoinit = lambda: None
    sys.modules['pycuda'].driver = types.ModuleType('driver')
    sys.modules['pycuda'].driver.init = lambda: True
    
    try:
        # Import compression functions
        from .core.compression import (
            compress_4bit_sequence as _compress_4bit_sequence,
            decompress_slice as _decompress_slice,
            decompress_base as _decompress_base,
            load_two_fasta_sequences as _load_two_fasta_sequences,
            IUPAC_4BIT as _IUPAC_4BIT,
            REV_IUPAC as _REV_IUPAC
        )
        
        compress_4bit_sequence = _compress_4bit_sequence
        decompress_slice = _decompress_slice
        decompress_base = _decompress_base
        load_two_fasta_sequences = _load_two_fasta_sequences
        IUPAC_4BIT = _IUPAC_4BIT
        REV_IUPAC = _REV_IUPAC
        
    except ImportError as e:
        print(f"[Alignment Pipeline] Could not import compression functions: {e}")
    
    try:
        # Import alignment functions
        from .algorithms.nw_affine import (
            nw_affine_chunk as _nw_affine_chunk,
            nw_affine_chunk_with_reanchor as _nw_affine_chunk_with_reanchor,
            nw_overlap_realign as _nw_overlap_realign,
            detect_gap_explosion as _detect_gap_explosion
        )
        
        nw_affine_chunk = _nw_affine_chunk
        nw_affine_chunk_with_reanchor = _nw_affine_chunk_with_reanchor
        nw_overlap_realign = _nw_overlap_realign
        detect_gap_explosion = _detect_gap_explosion
        ALIGNMENT_AVAILABLE = True
        
    except ImportError as e:
        ALIGNMENT_AVAILABLE = False
        print(f"[Alignment Pipeline] Could not import alignment functions: {e}")
    
    try:
        # Import chunking functions
        from .core.chunking import (
            process_and_save_chunks_parallel as _process_and_save_chunks_parallel,
            reload_and_merge_chunks as _reload_and_merge_chunks,
            save_chunk_result as _save_chunk_result,
            load_chunk_result as _load_chunk_result
        )
        
        process_and_save_chunks_parallel = _process_and_save_chunks_parallel
        reload_and_merge_chunks = _reload_and_merge_chunks
        save_chunk_result = _save_chunk_result
        load_chunk_result = _load_chunk_result
        
    except ImportError as e:
        print(f"[Alignment Pipeline] Could not import chunking functions: {e}")
    
    try:
        # Import seeding functions
        from .core.seeding import strobes_from_4bit_buffer as _strobes_from_4bit_buffer, Strobemer as _Strobemer
        strobes_from_4bit_buffer = _strobes_from_4bit_buffer
        Strobemer = _Strobemer
        SEEDING_AVAILABLE = True
        
    except ImportError as e:
        SEEDING_AVAILABLE = False
        print(f"[Alignment Pipeline] Could not import seeding functions: {e}")
    
    try:
        # Import utility functions
        from .core.utilities import (
            iupac_is_match as _iupac_is_match,
            iupac_partial_overlap as _iupac_partial_overlap,
            compute_alignment_stats as _compute_alignment_stats,
            build_cigar as _build_cigar,
            realign_overlap_and_stitch as _realign_overlap_and_stitch,
            write_cigar_gz as _write_cigar_gz,
            write_paf as _write_paf,
            write_maf as _write_maf,
            write_summary as _write_summary,
            IUPAC_MAP as _IUPAC_MAP,
            base_set as _base_set
        )
        
        iupac_is_match = _iupac_is_match
        iupac_partial_overlap = _iupac_partial_overlap
        compute_alignment_stats = _compute_alignment_stats
        build_cigar = _build_cigar
        realign_overlap_and_stitch = _realign_overlap_and_stitch
        write_cigar_gz = _write_cigar_gz
        write_paf = _write_paf
        write_maf = _write_maf
        write_summary = _write_summary
        IUPAC_MAP = _IUPAC_MAP
        base_set = _base_set
        
    except ImportError as e:
        print(f"[Alignment Pipeline] Could not import utility functions: {e}")
    
    try:
        # Import CUDA extensions
        import torch
        CUDA_AVAILABLE = torch.cuda.is_available()
        PYCUDA_AVAILABLE = False  # Never use pycuda
        USE_CUDA = CUDA_AVAILABLE
    except:
        CUDA_AVAILABLE = False
        PYCUDA_AVAILABLE = False
        USE_CUDA = False
    
    try:
        from .cuda_extensions import (
            CUDAStrobeProcessor as _CUDAStrobeProcessor,
            CUDABeamPruner as _CUDABeamPruner,
            CUDABatchAligner as _CUDABatchAligner,
            TorchWindowPredictor as _TorchWindowPredictor,
            batch_generate_anchors_gpu as _batch_generate_anchors_gpu,
            compute_batch_scores_gpu as _compute_batch_scores_gpu,
            predict_window_gpu as _predict_window_gpu,
            load_torch_predictor as _load_torch_predictor,
            CUDA_EXTENSIONS_AVAILABLE as _CUDA_EXTENSIONS_AVAILABLE
        )
        
        CUDAStrobeProcessor = _CUDAStrobeProcessor
        CUDABeamPruner = _CUDABeamPruner
        CUDABatchAligner = _CUDABatchAligner
        TorchWindowPredictor = _TorchWindowPredictor
        batch_generate_anchors_gpu = _batch_generate_anchors_gpu
        compute_batch_scores_gpu = _compute_batch_scores_gpu
        predict_window_gpu = _predict_window_gpu
        load_torch_predictor = _load_torch_predictor
        CUDA_EXTENSIONS_AVAILABLE = _CUDA_EXTENSIONS_AVAILABLE
        CUDA_SUPPORT_AVAILABLE = True
        
        # Print GPU status
        if USE_CUDA:
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"[Alignment Pipeline] GPU acceleration enabled on: {torch.cuda.get_device_name(0)}")
                else:
                    print("[Alignment Pipeline] CUDA available but device not detected")
            except:
                print("[Alignment Pipeline] GPU acceleration enabled")
        else:
            print("[Alignment Pipeline] Running in CPU mode")
            
    except ImportError as e:
        CUDA_SUPPORT_AVAILABLE = False
        CUDA_EXTENSIONS_AVAILABLE = False
        print(f"[Alignment Pipeline] CUDA extensions import error: {e}")
        print("[Alignment Pipeline] Running in CPU mode")

# Export lazy import function
__all__.append('lazy_import')