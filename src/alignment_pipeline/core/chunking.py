"""
Chunk processing and management functions with GPU acceleration.
FIXED VERSION - Handles length mismatch and type errors.
Author: Rowel Facunla
"""

import os
import numpy as np
import glob
import json
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings
import math
from dataclasses import dataclass

# Try to import CUDA extensions
try:
    from ..cuda_extensions import (
        CUDAStrobeProcessor,
        CUDABeamPruner,
        batch_generate_anchors_gpu,
        CUDA_AVAILABLE
    )
    CUDA_EXTENSIONS_AVAILABLE = True
except ImportError:
    CUDA_EXTENSIONS_AVAILABLE = False
    CUDA_AVAILABLE = False
    warnings.warn("CUDA extensions not available. Using CPU only.")

# Import config loader
try:
    from ..config.config_loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn("Config loader not available. Using default parameters.")

# Import other required modules
try:
    from ..algorithms.nw_affine import nw_affine_chunk, nw_affine_chunk_with_reanchor
    from ..algorithms.anchor_chaining import Chunk
    from ..core.utilities import iupac_is_match
    from ..core.compression import decompress_slice
    from ..algorithms.seed_matcher import build_strobe_index, generate_true_anchors
    from ..core.seeding import strobes_from_4bit_buffer
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    warnings.warn(f"Failed to import required modules: {e}")

# =====================================================================
# FIXED: SIMPLE BUT ROBUST CHUNKING APPROACH
# =====================================================================

def save_chunk_result(
    outdir: str,
    chunk_id: int,
    s1_start: int,
    s1_end: int,
    s2_start: int,
    s2_end: int,
    score: float,
    a1: str,
    comp: str,
    a2: str
) -> str:
    """
    Save chunk alignment result to disk.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"chunk_{chunk_id:06d}.npz")
    
    np.savez_compressed(
        path,
        meta=json.dumps({
            "chunk_id": int(chunk_id),
            "s1_start": int(s1_start),
            "s1_end": int(s1_end),
            "s2_start": int(s2_start),
            "s2_end": int(s2_end),
            "score": float(score)
        }),
        a1=np.array([a1], dtype=object),
        comp=np.array([comp], dtype=object),
        a2=np.array([a2], dtype=object)
    )
    return path


def load_chunk_result(path: str) -> Tuple[Dict, str, str, str]:
    """
    Load chunk alignment result from disk.
    """
    data = np.load(path, allow_pickle=True)
    
    if 'meta' in data:
        meta = json.loads(str(data['meta'].tolist()))
    else:
        meta = {
            'chunk_id': int(data.get('chunk_id', 0)),
            's1_start': int(data.get('s1_start', 0)),
            's1_end': int(data.get('s1_end', 0)),
            's2_start': int(data.get('s2_start', 0)),
            's2_end': int(data.get('s2_end', 0)),
            'score': float(data.get('score', 0))
        }
    
    # Extract strings
    if 'a1' in data:
        a1 = str(data['a1'].tolist()[0]) if data['a1'].ndim == 1 else str(data['a1'].tolist())
    else:
        a1 = ""
    
    if 'comp' in data:
        comp = str(data['comp'].tolist()[0]) if data['comp'].ndim == 1 else str(data['comp'].tolist())
    else:
        comp = ""
    
    if 'a2' in data:
        a2 = str(data['a2'].tolist()[0]) if data['a2'].ndim == 1 else str(data['a2'].tolist())
    else:
        a2 = ""
    
    return meta, a1, comp, a2


def calculate_alignment_score(a1: str, a2: str, gap_open: float, gap_extend: float) -> float:
    """
    Calculate alignment score from aligned sequences.
    """
    score = 0.0
    in_gap = False
    gap_type = None
    
    for x, y in zip(a1, a2):
        if x != '-' and y != '-':
            if iupac_is_match(x, y):
                score += 1.0
            else:
                score += gap_open
            in_gap = False
        elif x == '-' and y != '-':
            if not in_gap or gap_type != 'I':
                score += gap_open
                in_gap = True
                gap_type = 'I'
            else:
                score += gap_extend
        elif x != '-' and y == '-':
            if not in_gap or gap_type != 'D':
                score += gap_open
                in_gap = True
                gap_type = 'D'
            else:
                score += gap_extend
    
    return score


def simple_nw_fallback(seq1: str, seq2: str, match: float = 1.0, 
                      mismatch: float = -1.0, gap: float = -1.0) -> Tuple[str, str, str]:
    """
    Simple Needleman-Wunsch fallback when the main algorithm fails.
    """
    m, n = len(seq1), len(seq2)
    
    # Initialize DP matrix
    dp = np.zeros((m + 1, n + 1), dtype=float)
    
    # Fill first row and column
    for i in range(m + 1):
        dp[i][0] = i * gap
    for j in range(n + 1):
        dp[0][j] = j * gap
    
    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = dp[i-1][j-1] + (match if iupac_is_match(seq1[i-1], seq2[j-1]) else mismatch)
            delete = dp[i-1][j] + gap
            insert = dp[i][j-1] + gap
            dp[i][j] = max(match_score, delete, insert)
    
    # Traceback
    i, j = m, n
    a1_rev, a2_rev = [], []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if iupac_is_match(seq1[i-1], seq2[j-1]) else mismatch):
            a1_rev.append(seq1[i-1])
            a2_rev.append(seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap:
            a1_rev.append(seq1[i-1])
            a2_rev.append('-')
            i -= 1
        else:
            a1_rev.append('-')
            a2_rev.append(seq2[j-1])
            j -= 1
    
    a1 = ''.join(reversed(a1_rev))
    a2 = ''.join(reversed(a2_rev))
    comp = ''.join('|' if iupac_is_match(x, y) else ' ' for x, y in zip(a1, a2))
    
    return a1, comp, a2


def process_single_chunk_safe(args):
    """
    Safe version of chunk processing with robust error handling.
    """
    try:
        chunk_idx, chunk, seq1_tuple, seq2_tuple, nw_params_tuple, config_params = args
        buf1, L1 = seq1_tuple
        buf2, L2 = seq2_tuple
        
        q0, q1 = chunk.q_start, chunk.q_end
        t0, t1 = chunk.t_start, chunk.t_end
        
        # Extract sequences
        s1 = decompress_slice(buf1, L1, q0, q1)
        s2 = decompress_slice(buf2, L2, t0, t1)
        
        gap_open, gap_extend, penalty_matrix, beam_width = nw_params_tuple
        
        # Get NW affine parameters
        tau = config_params.get('tau', 3.0)
        energy_match = config_params.get('energy_match', -5.0)
        energy_mismatch = config_params.get('energy_mismatch', 40.0)
        energy_gap_open = config_params.get('energy_gap_open', 30.0)
        energy_gap_extend = config_params.get('energy_gap_extend', 0.5)
        carry_gap_penalty = config_params.get('carry_gap_penalty', True)
        
        # Determine window size - FIXED: use dynamic sizing
        min_len = min(len(s1), len(s2))
        ws = max(10, min_len // 5)  # More conservative window
        ws = min(ws, 100)  # Cap at 100
        
        # Try alignment with safety checks
        try:
            # First, validate sequences are not empty
            if not s1 or not s2:
                raise ValueError("Empty sequence in chunk")
            
            # Try the main alignment function
            score, a1, comp, a2, gap_state = nw_affine_chunk(
                s1, s2,
                gap_open, gap_extend,
                penalty_matrix,
                ws,
                beam_width,
                tau=tau,
                energy_match=energy_match,
                energy_mismatch=energy_mismatch,
                energy_gap_open=energy_gap_open,
                energy_gap_extend=energy_gap_extend,
                carry_gap_penalty=carry_gap_penalty,
                prev_gap_state=None
            )
            
            # CRITICAL: Validate alignment output
            a1_no_gaps = a1.replace('-', '')
            a2_no_gaps = a2.replace('-', '')
            
            if len(a1_no_gaps) != len(s1):
                print(f"WARNING: Chunk {chunk.cid}: seq1 length mismatch: {len(a1_no_gaps)} vs {len(s1)}")
                # Try fallback
                a1, comp, a2 = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
                score = calculate_alignment_score(a1, a2, gap_open, gap_extend)
            
            if len(a2_no_gaps) != len(s2):
                print(f"WARNING: Chunk {chunk.cid}: seq2 length mismatch: {len(a2_no_gaps)} vs {len(s2)}")
                # Try fallback
                a1, comp, a2 = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
                score = calculate_alignment_score(a1, a2, gap_open, gap_extend)
            
        except Exception as e:
            print(f"WARNING: nw_affine_chunk failed for chunk {chunk.cid}: {e}")
            # Use fallback
            a1, comp, a2 = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
            score = calculate_alignment_score(a1, a2, gap_open, gap_extend)
        
        return (chunk_idx, chunk.cid, q0, q1, t0, t1, score, a1, comp, a2)
        
    except Exception as e:
        import traceback
        print(f"ERROR in chunk processing: {e}")
        traceback.print_exc()
        return (chunk_idx if 'chunk_idx' in locals() else 0,
                chunk.cid if 'chunk' in locals() else 0,
                0, 0, 0, 0,
                0.0, "", "", "")


# =====================================================================
# SIMPLE BUT EFFECTIVE CHUNKING
# =====================================================================

def create_simple_chunks(L1: int, L2: int, chunk_size: int = 5000, 
                        overlap: int = 500, verbose: bool = False) -> List[Chunk]:
    """
    Create simple diagonal chunks. This works better than complex chunking
    for sequences with long deletions.
    """
    from ..algorithms.anchor_chaining import Chunk
    
    chunks = []
    cid = 0
    
    # Use a simple diagonal approach
    max_len = max(L1, L2)
    step = chunk_size - overlap
    
    for i in range(0, max_len, step):
        q_start = i
        q_end = min(i + chunk_size, L1)
        t_start = i
        t_end = min(i + chunk_size, L2)
        
        # Only create chunk if both sequences have content in this region
        if q_end > q_start and t_end > t_start:
            chunks.append(Chunk(
                cid=cid,
                q_start=q_start,
                q_end=q_end,
                t_start=t_start,
                t_end=t_end
            ))
            cid += 1
    
    if verbose:
        print(f"    - Created {len(chunks)} simple diagonal chunks")
    
    return chunks


# =====================================================================
# FIXED MAIN FUNCTION
# =====================================================================

def process_and_save_chunks_parallel(
    seq1_tuple: Tuple[bytes, int],
    seq2_tuple: Tuple[bytes, int],
    outdir: str,
    gap_open: Optional[float] = None,
    gap_extend: Optional[float] = None,
    penalty_matrix: Optional[Dict] = None,
    predict_window_fn=None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    beam_width: Optional[int] = None,
    verbose: Optional[bool] = None,
    carry_gap_state: Optional[bool] = None,
    num_workers: Optional[Union[int, str]] = None,
    use_gpu: Optional[bool] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    SIMPLIFIED AND FIXED version of chunk processing.
    Uses simple diagonal chunking with robust error handling.
    """
    if not IMPORT_SUCCESS:
        raise ImportError("Required modules not imported successfully")
    
    import traceback
    
    # Load configuration
    if CONFIG_AVAILABLE:
        if config_path:
            from ..config.config_loader import reload_config
            reload_config(config_path)
        
        config = get_config()
        alignment_params = config.get_alignment_params()
        chunking_params = config.get_chunking_params()
        nw_affine_params = config.get_nw_affine_params()
        performance_params = config.get_performance_params()
        gpu_params = config.get_gpu_params()
        debug_params = config.config.get('debug', {})
    else:
        # Use defaults
        alignment_params = {}
        chunking_params = {}
        nw_affine_params = {}
        performance_params = {}
        gpu_params = {}
        debug_params = {}
        warnings.warn("Config loader not available. Using default parameters.")
    
    # Set parameters from config if not provided
    if gap_open is None:
        gap_open = alignment_params.get('gap_open', -30)
    
    if gap_extend is None:
        gap_extend = alignment_params.get('gap_extend', -0.5)
    
    if beam_width is None:
        beam_width = alignment_params.get('beam_width', 30)
    
    if chunk_size is None:
        chunk_size = chunking_params.get('default_chunk_size', 5000)
    
    if overlap is None:
        overlap = chunking_params.get('overlap', 500)
    
    if verbose is None:
        verbose = debug_params.get('verbose', False)
    
    if carry_gap_state is None:
        carry_gap_state = False  # Disable for now - causes issues
    
    # Get NW affine parameters
    tau = nw_affine_params.get('tau', 3.0)
    energy_match = nw_affine_params.get('energy_match', -5.0)
    energy_mismatch = nw_affine_params.get('energy_mismatch', 40.0)
    energy_gap_open = nw_affine_params.get('energy_gap_open', 30.0)
    energy_gap_extend = nw_affine_params.get('energy_gap_extend', 0.5)
    carry_gap_penalty = nw_affine_params.get('carry_gap_penalty', True)
    
    # Create simple penalty matrix if not provided
    if penalty_matrix is None:
        match_score = alignment_params.get('match_score', 5)
        mismatch_score = alignment_params.get('mismatch_score', -40)
        penalty_matrix = {
            'match': match_score,
            'mismatch': mismatch_score
        }
    
    # Prepare NW affine parameters dictionary
    nw_affine_params_dict = {
        'tau': tau,
        'energy_match': energy_match,
        'energy_mismatch': energy_mismatch,
        'energy_gap_open': energy_gap_open,
        'energy_gap_extend': energy_gap_extend,
        'carry_gap_penalty': carry_gap_penalty
    }
    
    # Clean up old files
    if os.path.exists(outdir):
        old_files = glob.glob(os.path.join(outdir, "chunk_*.npz"))
        if old_files and verbose:
            print(f"[*] Cleaning up {len(old_files)} old chunk files")
        for f in old_files:
            try:
                os.remove(f)
            except:
                pass
    else:
        os.makedirs(outdir, exist_ok=True)
    
    buf1, L1 = seq1_tuple  # L1 is already an integer
    buf2, L2 = seq2_tuple  # L2 is already an integer
    
    # Ensure we have bytes
    if isinstance(buf1, bytearray):
        buf1_bytes = bytes(buf1)
    else:
        buf1_bytes = buf1
    
    if isinstance(buf2, bytearray):
        buf2_bytes = bytes(buf2)
    else:
        buf2_bytes = buf2
    
    if verbose:
        print(f"[*] Processing sequences: S1={L1}bp, S2={L2}bp")
        print(f"[*] Using chunk_size={chunk_size}, overlap={overlap}")
    
    # =================================================================
    # CREATE SIMPLE CHUNKS (NO COMPLEX ANCHORING)
    # =================================================================
    
    # For sequences with long deletions, simple diagonal chunking works best
    chunks = create_simple_chunks(L1, L2, chunk_size, overlap, verbose)
    
    # PRINT TOTAL NUMBER OF CHUNKS
    total_chunks = len(chunks)
    print(f"[*] Total chunks to process: {total_chunks}")
    
    if verbose:
        if chunks:
            print(f"    - First chunk: Q[{chunks[0].q_start}:{chunks[0].q_end}], "
                  f"T[{chunks[0].t_start}:{chunks[0].t_end}]")
            print(f"    - Last chunk: Q[{chunks[-1].q_start}:{chunks[-1].q_end}], "
                  f"T[{chunks[-1].t_start}:{chunks[-1].t_end}]")
    
    # =================================================================
    # PROCESS CHUNKS SEQUENTIALLY (MORE RELIABLE FOR LONG DELETIONS)
    # =================================================================
    
    saved = []
    
    # Prepare NW parameters tuple
    nw_params_tuple = (gap_open, gap_extend, penalty_matrix, beam_width)
    
    if verbose:
        print(f"[*] Processing chunks sequentially for better continuity...")
    
    prev_gap_state = None
    
    for chunk_idx, chunk in enumerate(chunks):
        q0, q1 = chunk.q_start, chunk.q_end
        t0, t1 = chunk.t_start, chunk.t_end
        
        # PRINT CURRENT CHUNK BEING PROCESSED
        print(f"[*] Processing chunk {chunk_idx + 1} of {total_chunks} (ID: {chunk.cid})")
        
        if verbose:
            print(f"  Q[{q0}:{q1}] T[{t0}:{t1}]")
        
        # Extract sequences
        s1 = decompress_slice(buf1_bytes, L1, q0, q1)
        s2 = decompress_slice(buf2_bytes, L2, t0, t1)
        
        if not s1 or not s2:
            if verbose:
                print(f"  - Empty sequence, skipping")
            continue
        
        # Determine window size
        min_len = min(len(s1), len(s2))
        ws = max(10, min_len // 5)
        ws = min(ws, 100)  # Cap at 100
        
        try:
            # Try to align with gap state carry if enabled
            if carry_gap_state and chunk_idx > 0 and prev_gap_state is not None:
                # Prepare extended parameters for nw_affine_chunk_with_reanchor
                nw_args = (
                    gap_open, gap_extend, penalty_matrix, ws, beam_width,
                    tau, energy_match, energy_mismatch, 
                    energy_gap_open, energy_gap_extend, carry_gap_penalty
                )
                
                # Need strobes for reanchor function - create empty lists if not available
                strobes1, strobes2 = [], []
                
                result = nw_affine_chunk_with_reanchor(
                    s1, s2,
                    strobes1, strobes2,
                    q0, t0,
                    *nw_args,
                    depth=0,
                    max_depth=2,
                    prev_gap_state=prev_gap_state,
                    config=nw_affine_params_dict
                )
            else:
                # Use simple nw_affine_chunk
                result = nw_affine_chunk(
                    s1, s2,
                    gap_open, gap_extend,
                    penalty_matrix,
                    ws,
                    beam_width,
                    tau=tau,
                    energy_match=energy_match,
                    energy_mismatch=energy_mismatch,
                    energy_gap_open=energy_gap_open,
                    energy_gap_extend=energy_gap_extend,
                    carry_gap_penalty=carry_gap_penalty,
                    prev_gap_state=None
                )
            
            # Handle result
            if len(result) == 5:
                score, a1_aligned, comp, a2_aligned, gap_state = result
            elif len(result) >= 3:
                score, a1_aligned, comp = result[:3]
                a2_aligned = ""
                gap_state = (False, False, 0)
            else:
                raise ValueError(f"Unexpected result format: {result}")
            
            # CRITICAL: Validate alignment
            a1_no_gaps = a1_aligned.replace('-', '')
            a2_no_gaps = a2_aligned.replace('-', '')
            
            if len(a1_no_gaps) != len(s1):
                if verbose:
                    print(f"  WARNING: seq1 length mismatch: {len(a1_no_gaps)} vs {len(s1)}")
                # Use fallback
                a1_aligned, comp, a2_aligned = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
                score = calculate_alignment_score(a1_aligned, a2_aligned, gap_open, gap_extend)
                gap_state = (False, False, 0)
            
            if len(a2_no_gaps) != len(s2):
                if verbose:
                    print(f"  WARNING: seq2 length mismatch: {len(a2_no_gaps)} vs {len(s2)}")
                # Use fallback
                a1_aligned, comp, a2_aligned = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
                score = calculate_alignment_score(a1_aligned, a2_aligned, gap_open, gap_extend)
                gap_state = (False, False, 0)
            
            prev_gap_state = gap_state
            
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
                traceback.print_exc()
            
            # Use fallback
            a1_aligned, comp, a2_aligned = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
            score = calculate_alignment_score(a1_aligned, a2_aligned, gap_open, gap_extend)
            prev_gap_state = None
        
        # FIXED: Save with correct parameters - seq1_tuple[1] is already L1 (an integer)
        path = save_chunk_result(outdir, chunk.cid, q0, q1, t0, t1, score, a1_aligned, comp, a2_aligned)
        saved.append(path)
        
        if verbose:
            matches = comp.count('|') if comp else 0
            gaps = (a1_aligned.count('-') + a2_aligned.count('-')) if a1_aligned and a2_aligned else 0
            print(f"  → Saved: {len(a1_aligned)}bp, {matches} matches, {gaps} gaps, score: {score:.1f}")
    
    # =================================================================
    # FINAL STATISTICS
    # =================================================================
    
    if verbose:
        print(f"\n[*] Processing complete. Saved {len(saved)} chunk files to {outdir}")
        if saved:
            total_aligned = 0
            total_score = 0
            total_gaps = 0
            total_matches = 0
            
            for path in saved:
                try:
                    meta, a1, comp, a2 = load_chunk_result(path)
                    total_aligned += len(a1) if a1 else 0
                    total_score += meta.get('score', 0)
                    total_gaps += (a1.count('-') + a2.count('-')) if a1 and a2 else 0
                    total_matches += comp.count('|') if comp else 0
                except:
                    continue
            
            if total_aligned > 0:
                print(f"    - Total aligned length: {total_aligned}bp")
                print(f"    - Total matches: {total_matches}")
                print(f"    - Total gaps: {total_gaps}")
                print(f"    - Total score: {total_score:.1f}")
                print(f"    - Match rate: {total_matches/total_aligned*100:.1f}%")
    
    return saved


def reload_and_merge_chunks(
    outdir: str,
    overlap_raw: int,
    cleanup: bool = True,
    verbose: bool = True
) -> Tuple[float, str, str, str]:
    """
    Reload and merge chunk results.
    """
    from ..core.utilities import realign_overlap_and_stitch
    
    files = sorted(glob.glob(os.path.join(outdir, "chunk_*.npz")))
    if not files:
        raise FileNotFoundError("No chunk files found in " + outdir)
    
    stitched_a1 = stitched_a2 = stitched_comp = ""
    total_score = 0.0
    
    for idx, path in enumerate(files):
        meta, a1, comp, a2 = load_chunk_result(path)
        total_score += float(meta.get("score", 0.0))
        
        if idx == 0:
            stitched_a1, stitched_a2, stitched_comp = a1, a2, comp
        else:
            stitched_a1, stitched_a2, stitched_comp = realign_overlap_and_stitch(
                stitched_a1, stitched_a2, a1, a2, overlap_raw
            )
        
        if verbose:
            print(f"Merged {os.path.basename(path)}")
    
    if cleanup:
        for p in files:
            try:
                os.remove(p)
            except Exception:
                pass
    
    return total_score, stitched_a1, stitched_comp, stitched_a2


# =====================================================================
# ALTERNATIVE: SINGLE CHUNK MODE FOR PROBLEMATIC SEQUENCES
# =====================================================================

def process_as_single_chunk(
    seq1_tuple: Tuple[bytes, int],
    seq2_tuple: Tuple[bytes, int],
    outdir: str,
    gap_open: float = -30,
    gap_extend: float = -0.5,
    penalty_matrix: Optional[Dict] = None,
    beam_width: int = 30,
    verbose: bool = False,
    config_path: Optional[str] = None
) -> List[str]:
    """
    Process sequences as a single chunk (no chunking).
    Use this when the chunked approach fails.
    """
    # Load configuration
    if CONFIG_AVAILABLE:
        if config_path:
            from ..config.config_loader import reload_config
            reload_config(config_path)
        
        config = get_config()
        alignment_params = config.get_alignment_params()
        nw_affine_params = config.get_nw_affine_params()
        debug_params = config.config.get('debug', {})
    else:
        alignment_params = {}
        nw_affine_params = {}
        debug_params = {}
    
    # Get parameters
    if gap_open is None:
        gap_open = alignment_params.get('gap_open', -30)
    if gap_extend is None:
        gap_extend = alignment_params.get('gap_extend', -0.5)
    if beam_width is None:
        beam_width = alignment_params.get('beam_width', 30)
    
    # Get NW affine parameters
    tau = nw_affine_params.get('tau', 3.0)
    energy_match = nw_affine_params.get('energy_match', -5.0)
    energy_mismatch = nw_affine_params.get('energy_mismatch', 40.0)
    energy_gap_open = nw_affine_params.get('energy_gap_open', 30.0)
    energy_gap_extend = nw_affine_params.get('energy_gap_extend', 0.5)
    carry_gap_penalty = nw_affine_params.get('carry_gap_penalty', True)
    
    # Prepare penalty matrix
    if penalty_matrix is None:
        match_score = alignment_params.get('match_score', 5)
        mismatch_score = alignment_params.get('mismatch_score', -40)
        penalty_matrix = {
            'match': match_score,
            'mismatch': mismatch_score
        }
    
    buf1, L1 = seq1_tuple
    buf2, L2 = seq2_tuple
    
    # Ensure bytes
    if isinstance(buf1, bytearray):
        buf1_bytes = bytes(buf1)
    else:
        buf1_bytes = buf1
    
    if isinstance(buf2, bytearray):
        buf2_bytes = bytes(buf2)
    else:
        buf2_bytes = buf2
    
    if verbose:
        print(f"[*] Processing as single chunk: {L1}bp vs {L2}bp")
    
    # Extract full sequences
    s1 = decompress_slice(buf1_bytes, L1, 0, L1)
    s2 = decompress_slice(buf2_bytes, L2, 0, L2)
    
    # Determine window size
    ws = max(10, min(L1, L2) // 10)
    ws = max(5, ws)
    
    try:
        # Try the main algorithm
        score, a1, comp, a2, _ = nw_affine_chunk(
            s1, s2,
            gap_open, gap_extend,
            penalty_matrix,
            ws,
            beam_width,
            tau=tau,
            energy_match=energy_match,
            energy_mismatch=energy_mismatch,
            energy_gap_open=energy_gap_open,
            energy_gap_extend=energy_gap_extend,
            carry_gap_penalty=carry_gap_penalty,
            prev_gap_state=None
        )
        
        # Validate
        a1_no_gaps = a1.replace('-', '')
        a2_no_gaps = a2.replace('-', '')
        
        if len(a1_no_gaps) != len(s1) or len(a2_no_gaps) != len(s2):
            if verbose:
                print(f"WARNING: Length mismatch, using fallback")
            a1, comp, a2 = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
            score = calculate_alignment_score(a1, a2, gap_open, gap_extend)
    
    except Exception as e:
        if verbose:
            print(f"ERROR in single chunk alignment: {e}")
        # Use fallback
        a1, comp, a2 = simple_nw_fallback(s1, s2, 1.0, gap_open, gap_extend)
        score = calculate_alignment_score(a1, a2, gap_open, gap_extend)
    
    # Save result
    path = save_chunk_result(outdir, 0, 0, L1, 0, L2, score, a1, comp, a2)
    
    if verbose:
        matches = comp.count('|')
        gaps = a1.count('-') + a2.count('-')
        print(f"[*] Alignment complete:")
        print(f"    - Length: {len(a1)}bp")
        print(f"    - Matches: {matches}")
        print(f"    - Gaps: {gaps}")
        print(f"    - Score: {score:.1f}")
        if len(a1) > 0:
            print(f"    - Match rate: {matches/len(a1)*100:.1f}%")
    
    return [path]


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    """Test the chunking module."""
    print("Chunking module loaded successfully")
    print(f"CUDA available: {CUDA_EXTENSIONS_AVAILABLE and CUDA_AVAILABLE}")
    print(f"Config available: {CONFIG_AVAILABLE}")
    print(f"All imports successful: {IMPORT_SUCCESS}")


if __name__ == "__main__":
    main()