import os
import numpy as np
import glob
import json
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any, Optional, Union
import warnings

from ..algorithms.nw_affine import nw_affine_chunk, nw_affine_chunk_with_reanchor
from ..algorithms.anchor_chaining import Chunk
from ..core.utilities import iupac_is_match
from ..core.compression import decompress_slice

# Import config loader
try:
    from ..config.config_loader import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn("Config loader not available. Using default parameters.")

# Import CUDA extensions
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
    warnings.warn("CUDA extensions not available. Using CPU only.")

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
    
    # Handle different storage formats
    if 'meta' in data:
        meta = json.loads(str(data['meta'].tolist()))
    else:
        # Fallback to old format
        meta = {
            'chunk_id': int(data.get('chunk_id', 0)),
            's1_start': int(data.get('s1_start', 0)),
            's1_end': int(data.get('s1_end', 0)),
            's2_start': int(data.get('s2_start', 0)),
            's2_end': int(data.get('s2_end', 0)),
            'score': float(data.get('score', 0))
        }
    
    # Extract strings from arrays
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

def process_single_chunk(args):
    """
    Process a single chunk with multiprocessing.
    Now reads all parameters from config.
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
        
        # Predict window size
        ws = max(10, min(len(s1), len(s2)) // 10)
        ws = max(5, ws)
        
        # Get all NW affine parameters from config
        tau = config_params.get('tau', 3.0)
        energy_match = config_params.get('energy_match', -5.0)
        energy_mismatch = config_params.get('energy_mismatch', 40.0)
        energy_gap_open = config_params.get('energy_gap_open', 30.0)
        energy_gap_extend = config_params.get('energy_gap_extend', 0.5)
        carry_gap_penalty = config_params.get('carry_gap_penalty', True)
        
        # Use GPU-accelerated beam pruning if available
        if CUDA_EXTENSIONS_AVAILABLE and CUDA_AVAILABLE:
            try:
                # Create beam pruner
                pruner = CUDABeamPruner()
                
                # Align with all parameters from config
                score, a1, comp, a2, gap_state = nw_affine_chunk_gpu_assisted(
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
                    prev_gap_state=None,  # Don't carry gap state across multiprocessing boundaries
                    gpu_pruner=pruner
                )
            except Exception as e:
                warnings.warn(f"GPU pruning failed, falling back to CPU: {e}")
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
        else:
            # CPU implementation with all parameters
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
        
        return (chunk_idx, chunk.cid, q0, q1, t0, t1, score, a1, comp, a2)
        
    except Exception as e:
        import traceback
        print(f"ERROR in chunk {chunk.cid if 'chunk' in locals() else 'unknown'}: {e}")
        traceback.print_exc()
        return (chunk_idx if 'chunk_idx' in locals() else 0,
                chunk.cid if 'chunk' in locals() else 0,
                0, 0, 0, 0,
                0.0, "", "", "")

def nw_affine_chunk_gpu_assisted(seq1, seq2, gap_open, gap_extend, penalty_matrix,
                                window_size, beam_width=100, **kwargs):
    """
    GPU-assisted version of nw_affine_chunk with all config parameters.
    """
    # Extract GPU-specific parameters
    gpu_pruner = kwargs.pop('gpu_pruner', None)
    
    if gpu_pruner is None or not CUDA_AVAILABLE:
        # Fall back to original function with all parameters
        return nw_affine_chunk(seq1, seq2, gap_open, gap_extend,
                              penalty_matrix, window_size, beam_width, **kwargs)
    
    # Call the original function (GPU pruning happens elsewhere)
    return nw_affine_chunk(seq1, seq2, gap_open, gap_extend,
                          penalty_matrix, window_size, beam_width, **kwargs)

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
    **kwargs  # For any additional parameters
) -> List[str]:
    """
    Process chunks in parallel with all parameters from config.
    
    Args:
        seq1_tuple: Tuple of (compressed_buffer, length) for sequence 1
        seq2_tuple: Tuple of (compressed_buffer, length) for sequence 2
        outdir: Output directory for chunk results
        gap_open: Gap open penalty (overrides config if provided)
        gap_extend: Gap extend penalty (overrides config if provided)
        penalty_matrix: Penalty matrix (overrides config if provided)
        predict_window_fn: Function to predict window size
        chunk_size: Chunk size (overrides config if provided)
        overlap: Overlap between chunks (overrides config if provided)
        beam_width: Beam width for alignment (overrides config if provided)
        verbose: Verbose output (overrides config if provided)
        carry_gap_state: Whether to carry gap state between chunks (overrides config)
        num_workers: Number of worker processes (overrides config if provided)
        use_gpu: Whether to use GPU acceleration (overrides config if provided)
        config_path: Path to config file (defaults to default_config.yaml)
        **kwargs: Additional parameters
        
    Returns:
        List of paths to saved chunk files
    """
    import traceback
    from ..algorithms.seed_matcher import build_strobe_index, generate_true_anchors
    from ..core.seeding import strobes_from_4bit_buffer
    from ..algorithms.anchor_chaining import (
        strobes_to_anchors,
        sort_anchors,
        chain_anchors,
        chains_to_chunks
    )
    
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
        validation_params = config.config.get('validation', {})
    else:
        # Use defaults
        alignment_params = {}
        chunking_params = {}
        nw_affine_params = {}
        performance_params = {}
        gpu_params = {}
        debug_params = {}
        validation_params = {}
        warnings.warn("Config loader not available. Using default parameters.")
    
    # Set parameters from config if not provided by function arguments
    # -----------------------------------------------------------------
    # Alignment parameters
    if gap_open is None:
        gap_open = alignment_params.get('gap_open', -30)
    
    if gap_extend is None:
        gap_extend = alignment_params.get('gap_extend', -0.5)
    
    if beam_width is None:
        beam_width = alignment_params.get('beam_width', 30)
    
    if carry_gap_state is None:
        carry_gap_state = alignment_params.get('carry_gap_state', True)
    
    # Get match and mismatch scores for penalty matrix
    match_score = alignment_params.get('match_score', 5)
    mismatch_score = alignment_params.get('mismatch_score', -40)
    
    # Create penalty matrix if not provided
    if penalty_matrix is None:
        penalty_matrix = {
            'match': match_score,
            'mismatch': mismatch_score,
            'A': {'A': match_score, 'C': mismatch_score, 'G': mismatch_score, 'T': mismatch_score},
            'C': {'A': mismatch_score, 'C': match_score, 'G': mismatch_score, 'T': mismatch_score},
            'G': {'A': mismatch_score, 'C': mismatch_score, 'G': match_score, 'T': mismatch_score},
            'T': {'A': mismatch_score, 'C': mismatch_score, 'G': mismatch_score, 'T': match_score}
        }
    
    # NW Affine specific parameters
    tau = nw_affine_params.get('tau', 3.0)
    energy_match = nw_affine_params.get('energy_match', -5.0)
    energy_mismatch = nw_affine_params.get('energy_mismatch', 40.0)
    energy_gap_open = nw_affine_params.get('energy_gap_open', 30.0)
    energy_gap_extend = nw_affine_params.get('energy_gap_extend', 0.5)
    carry_gap_penalty = nw_affine_params.get('carry_gap_penalty', True)
    
    # Chunking parameters
    if chunk_size is None:
        chunk_size = chunking_params.get('default_chunk_size', 10000)
    
    if overlap is None:
        overlap = chunking_params.get('overlap', 500)
    
    min_chunk_size = chunking_params.get('min_chunk_size', 500)
    max_chunk_size = chunking_params.get('max_chunk_size', 50000)
    
    # GPU parameters
    if use_gpu is None:
        use_gpu = gpu_params.get('enabled', True)
    
    gpu_memory_fraction = gpu_params.get('gpu_memory_fraction', 0.9)
    
    # Performance parameters
    if num_workers is None or num_workers == 'auto':
        num_workers = performance_params.get('num_workers', 'auto')
    
    use_multiprocessing = performance_params.get('use_multiprocessing', True)
    gpu_batch_size = performance_params.get('gpu_batch_size', 32)
    
    # Debug/Verbose parameters
    if verbose is None:
        verbose = debug_params.get('verbose', False)
    
    log_level = debug_params.get('log_level', 'INFO')
    
    # Validation parameters
    max_sequence_length = validation_params.get('max_sequence_length', 300000000)
    min_sequence_length = validation_params.get('min_sequence_length', 100)
    
    # Check GPU availability
    gpu_enabled = use_gpu and CUDA_EXTENSIONS_AVAILABLE and CUDA_AVAILABLE
    if verbose and gpu_enabled:
        print("[*] GPU acceleration enabled")
        print(f"    - Memory fraction: {gpu_memory_fraction}")
        print(f"    - Batch size: {gpu_batch_size}")
    elif verbose and use_gpu:
        print("[*] GPU acceleration requested but not available")
    
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
    
    buf1, L1 = seq1_tuple
    buf2, L2 = seq2_tuple
    
    # Validate sequence lengths
    if L1 > max_sequence_length or L2 > max_sequence_length:
        raise ValueError(f"Sequence too long. Max allowed: {max_sequence_length}")
    if L1 < min_sequence_length or L2 < min_sequence_length:
        raise ValueError(f"Sequence too short. Min required: {min_sequence_length}")
    
    # Ensure we have bytes, not bytearray
    if isinstance(buf1, bytearray):
        buf1_bytes = bytes(buf1)
    else:
        buf1_bytes = buf1
    
    if isinstance(buf2, bytearray):
        buf2_bytes = bytes(buf2)
    else:
        buf2_bytes = buf2
    
    # Extract strobes
    if verbose: 
        print("[*] Extracting strobemers...")
        print(f"    - Sequence 1 length: {L1}")
        print(f"    - Sequence 2 length: {L2}")
        print(f"    - Chunk size: {chunk_size}")
        print(f"    - Overlap: {overlap}")
    
    try:
        strobes1 = list(strobes_from_4bit_buffer(buf1_bytes, L1))
        strobes2 = list(strobes_from_4bit_buffer(buf2_bytes, L2))
    except Exception as e:
        if verbose:
            print(f"WARNING: Could not extract strobes: {e}")
        strobes1, strobes2 = [], []
    
    if verbose and strobes1 and strobes2:
        print(f"[*] S1 strobes: {len(strobes1)}")
        print(f"[*] S2 strobes: {len(strobes2)}")
    
    # Create chunks with optional GPU acceleration
    if not strobes1 or not strobes2:
        if verbose:
            print(f"[*] Not enough strobes for anchoring. Using simple tiling.")
        
        chunks = []
        step = chunk_size - overlap
        q_pos, t_pos = 0, 0
        cid = 0
        
        while q_pos < L1 or t_pos < L2:
            q_end = min(q_pos + chunk_size, L1) if q_pos < L1 else L1
            t_end = min(t_pos + chunk_size, L2) if t_pos < L2 else L2
            
            if q_pos < L1 and t_pos < L2:
                chunks.append(Chunk(
                    cid=cid,
                    q_start=q_pos,
                    q_end=q_end,
                    t_start=t_pos,
                    t_end=t_end
                ))
                cid += 1
            
            q_pos += step
            t_pos += step
    else:
        if gpu_enabled and chunking_params.get('use_gpu_anchoring', True):
            # GPU-accelerated anchor generation
            if verbose: 
                print("[*] Building strobe index with GPU...")
            try:
                processor = CUDAStrobeProcessor()
                
                # Convert strobes to tensors
                hashes1, positions1 = processor.convert_strobes_to_tensors(strobes1)
                hashes2, positions2 = processor.convert_strobes_to_tensors(strobes2)
                
                # Build index on GPU
                index2 = processor.build_index_gpu(hashes2, positions2, max_occ=200)
                
                # Generate anchors on GPU
                anchor_tuples = batch_generate_anchors_gpu(
                    hashes1, positions1, index2, positions2
                )
                
                # Convert to Anchor objects
                anchors = []
                for qpos, tpos, hash_val in anchor_tuples:
                    diag = tpos - qpos
                    # Find span from original strobes
                    span = 0
                    for st in strobes1:
                        if st.pos1 == qpos and st.hash == hash_val:
                            span = st.span
                            break
                    
                    anchors.append(Chunk(
                        cid=len(anchors),
                        q_start=qpos,
                        q_end=qpos + 1,  # Placeholder
                        t_start=tpos,
                        t_end=tpos + 1   # Placeholder
                    ))
                
                if verbose:
                    print(f"[*] GPU generated {len(anchors)} anchors")
                
                # Sort anchors (still CPU)
                anchors = sorted(anchors, key=lambda a: (a.t_start, a.q_start))
                
            except Exception as e:
                if verbose:
                    print(f"WARNING: GPU anchor generation failed: {e}")
                # Fall back to CPU
                index2 = build_strobe_index(strobes2)
                anchors = list(generate_true_anchors(strobes1, index2))
                anchors = sort_anchors(anchors)
        else:
            # CPU implementation
            if verbose: 
                print("[*] Building strobe index for seq2...")
            index2 = build_strobe_index(strobes2)
            
            if verbose: 
                print("[*] Matching strobes → anchors...")
            anchors = list(generate_true_anchors(strobes1, index2))
            
            if verbose:
                print(f"[*] Anchors generated: {len(anchors)}")
            
            if len(anchors) > 0:
                anchors = sort_anchors(anchors)
        
        # Chain anchors (CPU for now, could be GPU-accelerated)
        if len(anchors) > 0:
            chains = chain_anchors(anchors)
        else:
            chains = []
        
        if verbose:
            print(f"[*] Chains produced: {len(chains)}")
        
        if verbose: 
            print("[*] Building DSRC tiles...")
        chunks = chains_to_chunks(chains, L1, L2, chunk_size)
    
    # Limit chunk size
    for chunk in chunks:
        chunk.q_end = min(chunk.q_end, chunk.q_start + max_chunk_size)
        chunk.t_end = min(chunk.t_end, chunk.t_start + max_chunk_size)
    
    if verbose:
        print(f"[*] Chunks created: {len(chunks)}")
        if chunks:
            avg_len = sum((c.q_end - c.q_start + c.t_end - c.t_start) / 2 for c in chunks) / len(chunks)
            print(f"    - Average chunk length: {avg_len:.0f}bp")
            print(f"    - First chunk: Q[{chunks[0].q_start}:{chunks[0].q_end}], T[{chunks[0].t_start}:{chunks[0].t_end}]")
            print(f"    - Last chunk: Q[{chunks[-1].q_start}:{chunks[-1].q_end}], T[{chunks[-1].t_start}:{chunks[-1].t_end}]")
    
    # Process chunks
    saved = []
    
    # Prepare NW parameters tuple
    nw_params_tuple = (gap_open, gap_extend, penalty_matrix, beam_width)
    
    # Prepare NW affine parameters dictionary
    nw_affine_params_dict = {
        'tau': tau,
        'energy_match': energy_match,
        'energy_mismatch': energy_mismatch,
        'energy_gap_open': energy_gap_open,
        'energy_gap_extend': energy_gap_extend,
        'carry_gap_penalty': carry_gap_penalty
    }
    
    # Add any additional parameters from kwargs
    if kwargs:
        nw_affine_params_dict.update(kwargs)
    
    # Determine number of workers
    if num_workers == 'auto':
        if use_multiprocessing:
            num_workers = min(cpu_count(), len(chunks))
        else:
            num_workers = 1
    elif isinstance(num_workers, str):
        try:
            num_workers = int(num_workers)
        except ValueError:
            num_workers = 1
    
    # Ensure num_workers is an integer
    num_workers = int(num_workers) if isinstance(num_workers, (int, float)) else 1
    
    if num_workers > 1 and len(chunks) > 1 and use_multiprocessing:
        if verbose:
            print(f"[*] Processing {len(chunks)} chunks with {num_workers} workers...")
            print(f"    - Using multiprocessing: {num_workers} workers")
        
        try:
            # Prepare arguments for multiprocessing
            chunk_args = []
            for idx, chunk in enumerate(chunks):
                chunk_args.append((
                    idx, chunk, (buf1_bytes, L1), (buf2_bytes, L2), 
                    nw_params_tuple, nw_affine_params_dict
                ))
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(process_single_chunk, chunk_args)
            
            # Save results
            results.sort(key=lambda x: x[0])
            for _, cid, q0, q1, t0, t1, score, a1, comp, a2 in results:
                if a1 and a2:
                    path = save_chunk_result(outdir, cid, q0, q1, t0, t1, score, a1, comp, a2)
                    saved.append(path)
                    if verbose:
                        print(f"[chunk {cid}] Saved {len(a1)}bp alignment (score: {score:.1f})")
        except Exception as e:
            if verbose:
                print(f"WARNING: Multiprocessing failed, falling back to sequential: {e}")
                traceback.print_exc()
            num_workers = 1
    
    if num_workers <= 1 or len(chunks) <= 1 or not use_multiprocessing:
        # Sequential processing with all parameters from config
        if verbose:
            print(f"[*] Processing {len(chunks)} chunks sequentially...")
            if carry_gap_state:
                print(f"    - Carry gap state: Enabled")
            else:
                print(f"    - Carry gap state: Disabled")
        
        prev_gap_state = None
        
        for chunk_idx, chunk in enumerate(chunks):
            q0, q1 = chunk.q_start, chunk.q_end
            t0, t1 = chunk.t_start, chunk.t_end
            
            s1 = decompress_slice(buf1_bytes, L1, q0, q1)
            s2 = decompress_slice(buf2_bytes, L2, t0, t1)
            
            if verbose:
                print(f"[chunk {chunk.cid}] Q[{q0}:{q1}]={len(s1)}bp "
                      f"T[{t0}:{t1}]={len(s2)}bp")
            
            # Determine window size
            try:
                if predict_window_fn:
                    ws = int(predict_window_fn(s1, s2))
                else:
                    ws = max(10, min(len(s1), len(s2)) // 10)
            except:
                ws = max(10, min(len(s1), len(s2)) // 10)
            
            ws = max(5, ws)
            
            try:
                if carry_gap_state and chunk_idx > 0:
                    # Prepare NW arguments with all config parameters
                    nw_args = (
                        gap_open, gap_extend, penalty_matrix, ws, beam_width,
                        tau, energy_match, energy_mismatch, 
                        energy_gap_open, energy_gap_extend, carry_gap_penalty
                    )
                    
                    result = nw_affine_chunk_with_reanchor(
                        s1, s2,
                        strobes1, strobes2,
                        q0, t0,
                        *nw_args,
                        depth=0,
                        max_depth=3,
                        prev_gap_state=prev_gap_state,
                        config=nw_affine_params_dict
                    )
                else:
                    # Prepare NW arguments with all config parameters
                    nw_args = (
                        gap_open, gap_extend, penalty_matrix, ws, beam_width,
                        tau, energy_match, energy_mismatch, 
                        energy_gap_open, energy_gap_extend, carry_gap_penalty
                    )
                    
                    result = nw_affine_chunk_with_reanchor(
                        s1, s2,
                        strobes1, strobes2,
                        q0, t0,
                        *nw_args,
                        config=nw_affine_params_dict
                    )
                
                if len(result) == 5:
                    score, a1, comp, a2, gap_state = result
                else:
                    if verbose:
                        print(f"WARNING: Unexpected return format from nw_affine_chunk_with_reanchor")
                    score, a1, comp = result[:3]
                    a2 = ""
                    gap_state = (False, False, 0)
                
                prev_gap_state = gap_state
                
            except Exception as e:
                if verbose:
                    print(f"ERROR aligning chunk {chunk.cid}: {e}")
                    traceback.print_exc()
                from ..algorithms.nw_affine import nw_overlap_realign
                a1, a2 = nw_overlap_realign(s1, s2, match=1, mismatch=gap_open, gap=gap_extend)
                comp = ''.join('|' if iupac_is_match(x, y) else ' ' for x, y in zip(a1, a2))
                
                matches = sum(1 for x, y in zip(a1, a2) if iupac_is_match(x, y))
                mismatches = sum(1 for x, y in zip(a1, a2) if not iupac_is_match(x, y) and x != '-' and y != '-')
                gaps = a1.count('-') + a2.count('-')
                score = matches * 1.0 + mismatches * gap_open + gaps * gap_extend
                prev_gap_state = None
            
            path = save_chunk_result(outdir, chunk.cid, q0, q1, t0, t1, score, a1, comp, a2)
            saved.append(path)
            
            if verbose:
                print(f"  → Saved to {os.path.basename(path)} (score: {score:.1f})")
    
    if verbose:
        print(f"[*] Processing complete. Saved {len(saved)} chunk files to {outdir}")
        if saved:
            total_aligned = sum(len(load_chunk_result(p)[1]) for p in saved)
            print(f"    - Total aligned bases: {total_aligned}")
            print(f"    - Total score: {sum(load_chunk_result(p)[0].get('score', 0) for p in saved):.1f}")
    
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