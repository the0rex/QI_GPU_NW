import os
import sys
import yaml
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from multiprocessing import cpu_count
import argparse

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file. Fails if file does not exist."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML format in {config_path}")

    config['_source'] = str(config_path.resolve())
    return config

def parse_num_workers(num_workers_spec):
    """Parse num_workers specification."""
    if num_workers_spec == 'auto':
        return max(1, cpu_count() - 1)
    elif isinstance(num_workers_spec, str) and num_workers_spec.isdigit():
        return int(num_workers_spec)
    elif isinstance(num_workers_spec, int):
        return max(1, num_workers_spec)
    else:
        return 1

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(config['io']['logs_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_level_str = config['debug'].get('log_level', 'INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    logger = logging.getLogger('alignment_pipeline')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_dir / 'pipeline.log')
    fh.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_window_predictor(predictor_type: str = 'gpu'):
    """Get window size predictor function based on type."""
    
    def simple_window_predictor(seq1, seq2):
        """Simple heuristic predictor - returns integer window size."""
        len_diff_ratio = abs(len(seq1) - len(seq2)) / max(len(seq1), len(seq2))
        if len_diff_ratio > 0.3:
            base_window = max(100, min(len(seq1), len(seq2)) // 5)
        else:
            base_window = max(50, min(len(seq1), len(seq2)) // 10)
        
        return max(base_window, abs(len(seq1) - len(seq2)) + 10)
    
    if predictor_type == 'simple':
        return simple_window_predictor
    
    try:
        # Set model paths before importing
        from pathlib import Path
        
        # Find the predictor_models directory
        current_dir = Path(__file__).parent
        predictor_models_dir = current_dir.parent / "src" / "alignment_pipeline" / "algorithms" / "predictor_models"
        
        if predictor_models_dir.exists():
            # Set environment variables for the predict_window module
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['MODEL_PATH'] = str(predictor_models_dir / 'model_Adam.h5')
            os.environ['X_SCALER_PATH'] = str(predictor_models_dir / 'x_scaler.pkl')
            os.environ['Y_SCALER_PATH'] = str(predictor_models_dir / 'y_scaler.pkl')
        
        # Now import the predictor
        from alignment_pipeline.algorithms.predict_window import predict_window
        
        logger = logging.getLogger('alignment_pipeline')
        logger.info("Using neural network window predictor")
        
        # Return the function as-is (it returns integer window size)
        return predict_window
        
    except Exception as e:
        logger = logging.getLogger('alignment_pipeline')
        logger.warning(f"Could not load window predictor: {e}. Using simple.")
        return simple_window_predictor

def check_gpu_availability():
    """Check GPU availability and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            return {
                'available': True,
                'count': gpu_count,
                'name': gpu_name,
                'memory_gb': gpu_memory
            }
        else:
            return {'available': False}
    except ImportError:
        return {'available': False, 'error': 'PyTorch not installed'}

def list_chromosomes(fasta_file: str, limit: int = 0, range_spec: str = None) -> List[str]:
    """
    List all chromosome names in a FASTA file with optional range filtering.
    
    Args:
        fasta_file: Path to FASTA file
        limit: Maximum number of chromosomes to list (0 for all)
        range_spec: Range specification like "1:10", "chr1:chr10", "X:Y"
    
    Returns:
        List of chromosome names
    """
    from alignment_pipeline.io.fasta_reader import read_fasta_file
    
    try:
        sequences = read_fasta_file(fasta_file)
        chrom_names = list(sequences.keys())
        
        # Sort naturally (chr1, chr2, ..., chr10, etc.)
        def natural_sort_key(name):
            import re
            parts = re.split(r'(\d+)', name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]
        
        chrom_names.sort(key=natural_sort_key)
        
        # Apply range filter if specified
        if range_spec:
            chrom_names = filter_chromosomes_by_range(chrom_names, range_spec)
        
        if limit > 0 and len(chrom_names) > limit:
            return chrom_names[:limit]
        return chrom_names
        
    except Exception as e:
        raise ValueError(f"Failed to read FASTA file: {e}")

def filter_chromosomes_by_range(chrom_names: List[str], range_spec: str) -> List[str]:
    """
    Filter chromosome names by a range specification.
    
    Args:
        chrom_names: List of chromosome names (sorted)
        range_spec: Range specification like "1:10", "chr1:chr10", "X:Y"
    
    Returns:
        Filtered list of chromosome names
    """
    if ':' not in range_spec:
        raise ValueError(f"Invalid range specification: {range_spec}. Use format 'start:end'")
    
    start_str, end_str = range_spec.split(':', 1)
    start_str = start_str.strip()
    end_str = end_str.strip()
    
    # Find indices for start and end
    start_idx = None
    end_idx = None
    
    # Try different matching strategies
    for i, chrom in enumerate(chrom_names):
        # Case-insensitive matching
        chrom_lower = chrom.lower()
        
        # Try exact match first
        if chrom == start_str or chrom_lower == start_str.lower():
            start_idx = i
        if chrom == end_str or chrom_lower == end_str.lower():
            end_idx = i
        
        # Try partial match (for "1" matching "chr1")
        if start_idx is None and (start_str in chrom or chrom_lower.startswith(start_str.lower())):
            start_idx = i
        if end_idx is None and (end_str in chrom or chrom_lower.startswith(end_str.lower())):
            end_idx = i
        
        # Try numeric matching (for "1" matching "1" in "chr1")
        if start_idx is None and start_str.isdigit():
            # Extract numbers from chromosome name
            numbers = re.findall(r'\d+', chrom)
            if numbers and numbers[0] == start_str:
                start_idx = i
        if end_idx is None and end_str.isdigit():
            numbers = re.findall(r'\d+', chrom)
            if numbers and numbers[0] == end_str:
                end_idx = i
    
    if start_idx is None:
        raise ValueError(f"Start chromosome '{start_str}' not found in chromosome list")
    if end_idx is None:
        raise ValueError(f"End chromosome '{end_str}' not found in chromosome list")
    
    # Make sure indices are in order
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    
    # Return inclusive range
    return chrom_names[start_idx:end_idx + 1]

def list_chromosomes_simple(fasta_file: str, limit: int = 0, range_spec: str = None) -> List[str]:
    """
    Simple FASTA parser without BioPython dependency.
    """
    chrom_names = []
    
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Extract sequence ID (first word after '>')
                    seq_id = line[1:].split()[0]
                    chrom_names.append(seq_id)
        
        # Sort naturally
        def natural_sort_key(name):
            import re
            parts = re.split(r'(\d+)', name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]
        
        chrom_names.sort(key=natural_sort_key)
        
        # Apply range filter if specified
        if range_spec:
            chrom_names = filter_chromosomes_by_range(chrom_names, range_spec)
        
        if limit > 0 and len(chrom_names) > limit:
            return chrom_names[:limit]
        return chrom_names
        
    except Exception as e:
        raise ValueError(f"Failed to read FASTA file: {e}")

def extract_combined_chromosomes_from_fasta(fasta_file: str, chromosome_spec: str) -> Tuple[str, List[str]]:
    """
    Extract combined chromosome sequence from a FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        chromosome_spec: Chromosome specification (e.g., "chr1+chr2", "chrX+chrY", "chr1")
    
    Returns:
        Tuple of (concatenated_sequence, list_of_chromosome_names)
    """
    from alignment_pipeline.io.fasta_reader import read_fasta_file
    
    sequences = read_fasta_file(fasta_file)
    
    # Split by '+' to handle combined chromosomes
    chrom_parts = [part.strip() for part in chromosome_spec.split('+')]
    extracted_sequences = []
    found_chroms = []
    
    for chrom_part in chrom_parts:
        # Try different naming conventions for each part
        possible_names = [
            chrom_part,
            f"chr{chrom_part}",
            chrom_part.lstrip('chr'),
            f"chromosome{chrom_part}",
            f"chromosome_{chrom_part}",
            f"Chr{chrom_part}",
            f"CHR{chrom_part}"
        ]
        
        found = False
        for name in possible_names:
            if name in sequences:
                extracted_sequences.append(sequences[name])
                found_chroms.append(name)
                found = True
                break
        
        if not found:
            # Try to match partial names (case-insensitive)
            chrom_part_lower = chrom_part.lower()
            for seq_name in sequences.keys():
                seq_name_lower = seq_name.lower()
                if chrom_part_lower in seq_name_lower or seq_name_lower in chrom_part_lower:
                    extracted_sequences.append(sequences[seq_name])
                    found_chroms.append(seq_name)
                    found = True
                    break
            
            if not found:
                available_chroms = list(sequences.keys())
                raise ValueError(f"Chromosome '{chrom_part}' not found in {fasta_file}. "
                               f"Available chromosomes (first 10): {available_chroms[:10]}...")
    
    # Concatenate sequences with a separator (optional)
    separator = ""  # Default empty separator
    combined_sequence = separator.join(extracted_sequences)
    
    return combined_sequence, found_chroms

def extract_chromosome_from_fasta(fasta_file: str, chromosome: str) -> str:
    """
    Extract a specific chromosome sequence from a FASTA file.
    Supports both single and combined chromosomes.
    
    Args:
        fasta_file: Path to FASTA file
        chromosome: Chromosome name/ID to extract (e.g., 'chr1', 'chr1+chr2', 'chrX+chrY')
    
    Returns:
        DNA sequence as string
    """
    # Check if it's a combined chromosome specification
    if '+' in chromosome:
        combined_seq, chrom_names = extract_combined_chromosomes_from_fasta(fasta_file, chromosome)
        logger = logging.getLogger('alignment_pipeline')
        logger.info(f"Extracted combined chromosomes {chrom_names} from {fasta_file}")
        return combined_seq
    else:
        # Single chromosome extraction (backward compatible)
        from alignment_pipeline.io.fasta_reader import read_fasta_file
        
        sequences = read_fasta_file(fasta_file)
        
        # Try different naming conventions
        possible_names = [
            chromosome,
            f"chr{chromosome}",
            chromosome.lstrip('chr'),
            f"chromosome{chromosome}",
            f"chromosome_{chromosome}",
            f"Chr{chromosome}",
            f"CHR{chromosome}"
        ]
        
        for name in possible_names:
            if name in sequences:
                logger = logging.getLogger('alignment_pipeline')
                logger.info(f"Found chromosome '{name}' in {fasta_file}")
                return sequences[name]
        
        # If not found, try case-insensitive partial match
        chromosome_lower = chromosome.lower()
        for seq_name, seq in sequences.items():
            if chromosome_lower in seq_name.lower() or seq_name.lower() in chromosome_lower:
                logger = logging.getLogger('alignment_pipeline')
                logger.info(f"Found chromosome '{seq_name}' (matched '{chromosome}') in {fasta_file}")
                return seq
        
        # If not found, list available chromosomes
        available_chroms = list(sequences.keys())
        raise ValueError(f"Chromosome '{chromosome}' not found in {fasta_file}. "
                       f"Available chromosomes (first 10): {available_chroms[:10]}...")

def create_chromosome_fasta_files(
    fasta1: str, chrom1: str, 
    fasta2: str, chrom2: str,
    output_dir: Path
) -> Tuple[str, str, List[str], List[str]]:
    """
    Create temporary FASTA files with extracted chromosomes.
    Supports combined chromosomes.
    
    Returns:
        Tuple of (temp_fasta1_path, temp_fasta2_path, chrom1_names, chrom2_names)
    """
    from alignment_pipeline.io.fasta_reader import write_fasta
    
    # Extract sequences (handles both single and combined chromosomes)
    seq1, chrom1_list = extract_combined_chromosomes_from_fasta(fasta1, chrom1)
    seq2, chrom2_list = extract_combined_chromosomes_from_fasta(fasta2, chrom2)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive names for combined chromosomes
    chrom1_name = "_".join(chrom1_list) if len(chrom1_list) > 1 else chrom1_list[0]
    chrom2_name = "_".join(chrom2_list) if len(chrom2_list) > 1 else chrom2_list[0]
    
    # Create temporary FASTA files
    temp_fasta1 = output_dir / f"temp_{Path(fasta1).stem}_{chrom1_name}.fa"
    temp_fasta2 = output_dir / f"temp_{Path(fasta2).stem}_{chrom2_name}.fa"
    
    # Write sequences
    write_fasta(str(temp_fasta1), {chrom1_name: seq1})
    write_fasta(str(temp_fasta2), {chrom2_name: seq2})
    
    return str(temp_fasta1), str(temp_fasta2), chrom1_list, chrom2_list

def align_chromosomes(
    fasta1: str, chrom1: str,
    fasta2: str, chrom2: str,
    config: Dict[str, Any],
    logger: logging.Logger
) -> int:
    """
    Align specific chromosomes from two FASTA files.
    Supports combined chromosomes.
    
    Args:
        fasta1: Path to first FASTA file
        chrom1: Chromosome name(s) from first FASTA (can be combined with '+')
        fasta2: Path to second FASTA file
        chrom2: Chromosome name(s) from second FASTA (can be combined with '+')
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from alignment_pipeline.diagnostics.validation import validate_inputs
    
    logger.info(f"Aligning chromosomes:")
    logger.info(f"  File1: {fasta1} -> Chromosome(s): {chrom1}")
    logger.info(f"  File2: {fasta2} -> Chromosome(s): {chrom2}")
    
    # Create temporary FASTA files with extracted chromosomes
    # Use default temp_dir if not in config
    temp_dir = Path(config.get('io', {}).get('temp_dir', 'Data/temp_chromosomes'))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        temp_fasta1, temp_fasta2, chrom1_list, chrom2_list = create_chromosome_fasta_files(
            fasta1, chrom1, fasta2, chrom2, temp_dir
        )
        
        logger.info(f"Extracted chromosomes from {fasta1}: {chrom1_list}")
        logger.info(f"Extracted chromosomes from {fasta2}: {chrom2_list}")
        
        # Update config to use temporary files
        config['io']['default_fasta1'] = Path(temp_fasta1).name
        config['io']['default_fasta2'] = Path(temp_fasta2).name
        config['io']['fasta_dir'] = str(temp_dir)
        
        # Validate the extracted chromosome sequences
        if config['validation']['validate_inputs']:
            from alignment_pipeline.io.fasta_reader import validate_fasta_file
            
            valid1, msg1 = validate_fasta_file(temp_fasta1)
            valid2, msg2 = validate_fasta_file(temp_fasta2)
            
            if not valid1:
                logger.error(f"Invalid temporary FASTA file 1: {msg1}")
                return 1
            if not valid2:
                logger.error(f"Invalid temporary FASTA file 2: {msg2}")
                return 1
            
            # Check overall compatibility
            is_valid, errors = validate_inputs(temp_fasta1, temp_fasta2, config)
            if not is_valid:
                for error in errors:
                    logger.error(f"Validation error: {error}")
                return 1
        
        # Now run the normal pipeline with the temporary files
        # Create a descriptive name for the alignment
        chrom1_name = "+".join(chrom1_list)
        chrom2_name = "+".join(chrom2_list)
        return run_alignment_pipeline(config, logger, chrom1_name, chrom2_name)
        
    except Exception as e:
        logger.error(f"Failed to align chromosomes: {e}", exc_info=True)
        return 1
    finally:
        # Clean up temporary files if not in debug mode
        if not config['debug']['save_intermediate']:
            import shutil
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

def run_alignment_pipeline(
    config: Dict[str, Any],
    logger: logging.Logger,
    chrom1: str = None,
    chrom2: str = None
) -> int:
    """
    Core alignment pipeline logic.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        chrom1: Optional chromosome name(s) 1 (for naming output)
        chrom2: Optional chromosome name(s) 2 (for naming output)
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # First, lazy import all modules
    try:
        from alignment_pipeline import lazy_import
        lazy_import()
    except ImportError as e:
        print(f"ERROR: Could not lazy import modules: {e}")
        return 1
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Add the predictor_models directory to the system path
    predictor_models_dir = Path(__file__).parent.parent / "src" / "alignment_pipeline" / "algorithms" / "predictor_models"
    if predictor_models_dir.exists():
        sys.path.insert(0, str(predictor_models_dir))
        # You might need to set environment variables for the predict_window module
        os.environ['MODEL_PATH'] = str(predictor_models_dir / 'model_Adam.h5')
        os.environ['X_SCALER_PATH'] = str(predictor_models_dir / 'x_scaler.pkl')
        os.environ['Y_SCALER_PATH'] = str(predictor_models_dir / 'y_scaler.pkl')
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    if gpu_info.get('available', False):
        logger.info(f"GPU available: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
        config['performance']['use_gpu'] = True
    else:
        logger.info("GPU not available, using CPU only")
        config['performance']['use_gpu'] = False
        config['alignment']['use_gpu_acceleration'] = False
        config['chunking']['use_gpu_anchoring'] = False
        config['chunking']['use_gpu_pruning'] = False
    
    # Check versions - now import it here instead of at top level
    from alignment_pipeline.diagnostics.version_checker import check_versions
    check_versions()
    
    # Performance monitoring - now import it here instead of at top level
    from alignment_pipeline.diagnostics.performance import PerformanceMonitor
    perf_monitor = PerformanceMonitor()
    perf_monitor.start()
    
    try:
        # Get file paths from config
        fasta_dir = Path(config['io']['fasta_dir'])
        fasta1 = fasta_dir / config['io']['default_fasta1']
        fasta2 = fasta_dir / config['io']['default_fasta2']
        
        logger.info("Aligning sequences:")
        logger.info(f"  Sequence 1: {fasta1}")
        logger.info(f"  Sequence 2: {fasta2}")
        
        # Validate inputs - now import it here instead of at top level
        if config['validation']['validate_inputs']:
            from alignment_pipeline.diagnostics.validation import validate_inputs
            from alignment_pipeline.io.fasta_reader import validate_fasta_file
            
            # Check individual files
            valid1, msg1 = validate_fasta_file(str(fasta1))
            valid2, msg2 = validate_fasta_file(str(fasta2))
            
            if not valid1:
                logger.error(f"Invalid FASTA file 1: {msg1}")
                return 1
            if not valid2:
                logger.error(f"Invalid FASTA file 2: {msg2}")
                return 1
            
            # Check overall compatibility
            is_valid, errors = validate_inputs(str(fasta1), str(fasta2), config)
            if not is_valid:
                for error in errors:
                    logger.error(f"Validation error: {error}")
                return 1
        
        # Step 1: Load and compress sequences - now import it here instead of at top level
        from alignment_pipeline.core.compression import load_two_fasta_sequences
        logger.info("Loading and compressing sequences...")
        seq1_tuple, seq2_tuple = load_two_fasta_sequences(str(fasta1), str(fasta2))
        
        # Get penalty matrix
        penalty_matrix = {}
        bases = "ACGT"
        for b1 in bases:
            for b2 in bases:
                if b1 == b2:
                    penalty_matrix[(b1, b2)] = 1
                else:
                    penalty_matrix[(b1, b2)] = -3
        
        # Get window predictor
        predictor_type = config['alignment'].get('window_size_predictor', 'gpu')
        predict_window_fn = get_window_predictor(predictor_type)
        
        # Parse num_workers
        num_workers_spec = config['performance'].get('num_workers', 'auto')
        num_workers = parse_num_workers(num_workers_spec)
        
        # Disable multiprocessing if requested
        use_multiprocessing = config['performance'].get('use_multiprocessing', True)
        if not use_multiprocessing:
            num_workers = 1
        
        # Step 2: Process chunks with GPU acceleration - now import it here instead of at top level
        from alignment_pipeline.core.chunking import (
            process_and_save_chunks_parallel,
            reload_and_merge_chunks
        )
        
        # Step 2: Process chunks with GPU acceleration
        logger.info("Processing chunks with GPU acceleration...")
        
        # Determine GPU usage
        use_gpu = config['performance'].get('use_gpu', True) and gpu_info.get('available', False)
        
        saved_chunks = process_and_save_chunks_parallel(
            seq1_tuple, seq2_tuple,
            config['io']['chunk_dir'],
            gap_open=config['alignment']['gap_open'],
            gap_extend=config['alignment']['gap_extend'],
            penalty_matrix=penalty_matrix,
            predict_window_fn=predict_window_fn,
            chunk_size=config['chunking']['default_chunk_size'],
            overlap=config['chunking']['overlap'],  # Pass overlap parameter
            beam_width=config['alignment']['beam_width'],
            verbose=config['debug']['verbose'],
            carry_gap_state=config['alignment']['carry_gap_state'],
            num_workers=num_workers,
            use_gpu=use_gpu
        )
        
        logger.info(f"Processed {len(saved_chunks)} chunks")
        
        if not saved_chunks:
            logger.error("No chunks were created. Pipeline failed.")
            return 1
        
        # Step 3: Merge chunks
        logger.info("Merging chunks...")
        total_score, a1, comp, a2 = reload_and_merge_chunks(
            config['io']['chunk_dir'],
            overlap_raw=config['chunking']['overlap'],
            cleanup=not config['debug']['save_intermediate'],
            verbose=config['debug']['verbose']
        )
        
        # Step 4: Save results - now import it here instead of at top level
        from alignment_pipeline.io.results_writer import save_all_results
        logger.info("Saving results...")
        
        # If chromosome names are provided, use them in output
        output_dir = config['io']['output_dir']
        if chrom1 and chrom2:
            base_name = f"{chrom1}_vs_{chrom2}"
            results = save_all_results(
                a1, a2, total_score, str(fasta1), str(fasta2),
                output_dir=output_dir,
                base_name=base_name,
                chrom1=chrom1,
                chrom2=chrom2,
                config=config
            )
        else:
            results = save_all_results(
                a1, a2, total_score, str(fasta1), str(fasta2),
                output_dir=output_dir,
                config=config
            )
        
        # Step 5: Optional visualizations
        if config.get('visualization', {}).get('enabled', True):
            try:
                from alignment_pipeline.visualization.visualizer import (
                    visualize_pipeline_results
                )
                logger.info("Creating visualizations...")
                
                # Customize visualization title if chromosomes specified
                if chrom1 and chrom2:
                    title = f"Chromosome Alignment: {chrom1} vs {chrom2}"
                else:
                    title = None
                    
                visualize_pipeline_results(
                    final_a1=a1,
                    final_a2=a2,
                    total_score=total_score,
                    fasta1=str(fasta1),
                    fasta2=str(fasta2),
                    output_dir=config['io']['visualization_dir'],
                    title=title
                )
            except ImportError as e:
                logger.warning(f"Visualization module not available: {e}")
        
        # Performance report
        perf_monitor.stop()
        report = perf_monitor.get_report()
        
        # Add GPU info to report
        if gpu_info.get('available', False):
            try:
                import torch
                gpu_memory_used = torch.cuda.max_memory_allocated(0) / 1e9  # GB
                report['gpu_info'] = {
                    'device_name': gpu_info['name'],
                    'total_memory_gb': gpu_info['memory_gb'],
                    'max_memory_used_gb': gpu_memory_used,
                    'memory_utilization': (gpu_memory_used / gpu_info['memory_gb']) * 100
                }
            except:
                report['gpu_info'] = gpu_info
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total time: {report['total_time_seconds']:.2f}s")
        logger.info(f"Peak memory: {report['peak_memory_mb']:.1f} MB")
        logger.info(f"Total score: {total_score}")
        
        if gpu_info.get('available', False) and 'gpu_info' in report:
            gpu_report = report['gpu_info']
            logger.info(f"GPU memory used: {gpu_report.get('max_memory_used_gb', 0):.2f} GB")
        
        # Print alignment snippet
        logger.info("\nAlignment snippet (first 100 columns):")
        logger.info(f"Seq1: {a1[:100]}")
        logger.info(f"Comp: {comp[:100]}")
        logger.info(f"Seq2: {a2[:100]}")
        
        # Save performance report
        report_path = Path(config['io']['logs_dir']) / 'performance_report.json'
        perf_monitor.save_report(report_path)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1

def main(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> int:
    """Main pipeline entry point with GPU support."""
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"ERROR loading configuration: {e}")
        return 1
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting alignment pipeline with GPU acceleration")
    logger.info(f"Configuration loaded from {config.get('_source', 'default')}")
    
    # Check if chromosome alignment is requested
    chrom1 = config.get('io', {}).get('chrom1')
    chrom2 = config.get('io', {}).get('chrom2')
    
    if chrom1 and chrom2:
        # Run chromosome-to-chromosome alignment
        fasta_dir = Path(config['io']['fasta_dir'])
        fasta1 = fasta_dir / config['io']['default_fasta1']
        fasta2 = fasta_dir / config['io']['default_fasta2']
        
        return align_chromosomes(
            str(fasta1), chrom1,
            str(fasta2), chrom2,
            config, logger
        )
    else:
        # Run normal pipeline
        return run_alignment_pipeline(config, logger)

# Alias for backward compatibility
run_pipeline = main

def handle_chromosome_listing(args):
    """
    Handle chromosome listing for a FASTA file.
    
    Args:
        args: Command-line arguments
    """
    if not args.list_chromosomes:
        return None
    
    fasta_file = args.list_chromosomes
    if not os.path.exists(fasta_file):
        print(f"ERROR: FASTA file not found: {fasta_file}")
        return 1
    
    try:
        # Determine which function to use
        try:
            from alignment_pipeline.io.fasta_reader import read_fasta_file
            use_simple = False
        except ImportError:
            use_simple = True
        
        if use_simple:
            chrom_names = list_chromosomes_simple(fasta_file, args.list_limit, args.chromosome_range)
        else:
            chrom_names = list_chromosomes(fasta_file, args.list_limit, args.chromosome_range)
        
        print(f"\nChromosomes in {fasta_file}:")
        print("-" * 60)
        
        if args.chromosome_range:
            print(f"Range: {args.chromosome_range}")
            print("-" * 60)
        
        # Display chromosome names
        for i, chrom in enumerate(chrom_names, 1):
            print(f"{i:4d}. {chrom}")
        
        print("-" * 60)
        print(f"Total: {len(chrom_names)} chromosomes in range")
        
        # Show examples for combined alignment
        if len(chrom_names) >= 2:
            print(f"\nExamples for alignment using this range:")
            print(f"  Single chromosome: --chrom1 '{chrom_names[0]}'")
            print(f"  Two chromosomes: --chrom1 '{chrom_names[0]}+{chrom_names[1]}'")
            if len(chrom_names) >= 3:
                print(f"  Three chromosomes: --chrom1 '{chrom_names[0]}+{chrom_names[1]}+{chrom_names[2]}'")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Failed to list chromosomes: {e}")
        return 1

if __name__ == "__main__":
    # Enhanced command-line arguments for chromosome operations
    parser = argparse.ArgumentParser(description='DNA Sequence Alignment Pipeline')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--chrom1', help='Chromosome name(s) from first FASTA (use + to combine, e.g., "chr1+chr2")')
    parser.add_argument('--chrom2', help='Chromosome name(s) from second FASTA (use + to combine, e.g., "chrX+chrY")')
    parser.add_argument('--list-chromosomes', metavar='FASTA_FILE', 
                       help='List all chromosome names in a FASTA file')
    parser.add_argument('--list-limit', type=int, default=0,
                       help='Limit number of chromosomes to display (0 for all)')
    parser.add_argument('--chromosome-range', type=str,
                       help='Range of chromosomes to display (e.g., "1:10", "chr1:chr10", "X:Y")')
    
    args = parser.parse_args()
    
    # Handle chromosome listing if requested
    if args.list_chromosomes:
        sys.exit(handle_chromosome_listing(args))
    
    # Prepare overrides if chromosomes specified
    overrides = None
    if args.chrom1 and args.chrom2:
        overrides = {
            'io': {
                'chrom1': args.chrom1,
                'chrom2': args.chrom2
            }
        }
    
    sys.exit(main(config_path=args.config, overrides=overrides))