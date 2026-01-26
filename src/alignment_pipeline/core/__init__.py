"""
Core modules for the alignment pipeline.
"""

# Import compression functions
from .compression import (
    compress_4bit_sequence,
    decompress_slice,
    decompress_base,
    load_two_fasta_sequences,
    IUPAC_4BIT,
    REV_IUPAC
)

# Import utility functions
from .utilities import (
    iupac_is_match,
    iupac_partial_overlap,
    compute_alignment_stats,
    build_cigar,
    realign_overlap_and_stitch,
    write_cigar_gz,
    write_paf,
    write_maf,
    write_summary,
    IUPAC_MAP,
    base_set
)

# Import seeding functions - using deferred import to avoid circular dependencies
try:
    from .seeding import strobes_from_4bit_buffer, Strobemer
    SEEDING_AVAILABLE = True
except ImportError:
    SEEDING_AVAILABLE = False
    strobes_from_4bit_buffer = None
    Strobemer = None

# Alignment functions - import lazily to avoid circular dependencies
# We'll define them as None first and import when needed
nw_affine_chunk = None
nw_affine_chunk_with_reanchor = None
nw_overlap_realign = None
detect_gap_explosion = None
ALIGNMENT_AVAILABLE = False

# Chunking functions - import lazily to avoid circular dependencies
process_and_save_chunks_parallel = None
reload_and_merge_chunks = None
save_chunk_result = None
load_chunk_result = None

def lazy_import_alignment():
    """Lazy import alignment functions to avoid circular imports."""
    global nw_affine_chunk, nw_affine_chunk_with_reanchor, nw_overlap_realign, detect_gap_explosion
    global ALIGNMENT_AVAILABLE
    
    try:
        from ..algorithms.nw_affine import (
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

def lazy_import_chunking():
    """Lazy import chunking functions to avoid circular imports."""
    global process_and_save_chunks_parallel, reload_and_merge_chunks
    global save_chunk_result, load_chunk_result
    
    try:
        from .chunking import (
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

__all__ = [
    # Compression functions
    'compress_4bit_sequence',
    'decompress_slice',
    'decompress_base',
    'load_two_fasta_sequences',
    'IUPAC_4BIT',
    'REV_IUPAC',
    
    # Alignment functions (lazy-loaded)
    'nw_affine_chunk',
    'nw_affine_chunk_with_reanchor',
    'nw_overlap_realign',
    'detect_gap_explosion',
    
    # Chunking functions (lazy-loaded)
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
    
    # Lazy import functions
    'lazy_import_alignment',
    'lazy_import_chunking',
    
    # Availability flags
    'ALIGNMENT_AVAILABLE',
    'SEEDING_AVAILABLE',
]