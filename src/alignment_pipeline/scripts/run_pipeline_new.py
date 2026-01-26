#!/usr/bin/env python3
"""
Alignment Pipeline - Combined Main and CLI Script
Author: Rowel Facunla
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from multiprocessing import cpu_count

# Set TensorFlow logging level early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def setup_cuda():
    """Check CUDA availability and configure GPU settings."""
    try:
        import torch
        print(f"[MAIN] Checking CUDA...")
        print(f"[MAIN] PyTorch version: {torch.__version__}")
        
        CUDA_AVAILABLE = torch.cuda.is_available()
        if CUDA_AVAILABLE:
            # Make sure CUDA is initialized
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            
            print(f"[MAIN] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[MAIN] CUDA device count: {torch.cuda.device_count()}")
            
            # Test CUDA
            test_tensor = torch.tensor([1.0]).cuda()
            print(f"[MAIN] CUDA test tensor: {test_tensor.device}")
            
            # Only use GPU 0 to avoid conflicts
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            print("[MAIN] CUDA not available")
            print(f"[MAIN] CUDA version in PyTorch: {torch.version.cuda}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return False
    except ImportError:
        print("[MAIN] PyTorch not installed")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False
    except Exception as e:
        print(f"[MAIN] CUDA check error: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

# Add src to path if needed
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

def get_default_config() -> Dict[str, Any]:
    """Get default configuration from default_config.yaml or fallback."""
    default_config_path = Path(__file__).parent / "default_config.yaml"
    
    if default_config_path.exists():
        try:
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[MAIN] Loaded default configuration from {default_config_path}")
            return config
        except Exception as e:
            print(f"[MAIN] Error loading default_config.yaml: {e}, using built-in defaults")
    
    # Fallback to built-in default config
    return {
        'pipeline': {
            'name': 'Alignment Pipeline',
            'version': '1.0.0',
            'description': 'Default configuration with GPU support'
        },
        'io': {
            'fasta_dir': 'examples',
            'default_fasta1': 'test_fasta_1.fa',
            'default_fasta2': 'test_fasta_2.fa',
            'output_dir': 'Results',
            'chunk_dir': 'chunks_tmp',
            'visualization_dir': 'Results/visualizations',
            'logs_dir': 'Results/logs'
        },
        'alignment': {
            'gap_open': -30,
            'gap_extend': -0.5,
            'match_score': 5,
            'mismatch_score': -50,
            'window_size_predictor': 'gpu',
            'beam_width': 30,
            'carry_gap_state': True,
            'use_gpu_acceleration': True
        },
        'chunking': {
            'default_chunk_size': 5000,
            'overlap': 250,
            'min_chunk_size': 100,
            'max_chunk_size': 10000,
            'chunk_tolerance': 0.05,
            'use_gpu_anchoring': True,
            'use_gpu_pruning': True
        },
        'seeding': {
            'kmer_size': 21,
            'syncmer_size': 5,
            'syncmer_position': 2,
            'window_min': 20,
            'window_max': 70,
            'max_occurrences': 200
        },
        'models': {
            'window_predictor': 'model_Adam.h5',
            'torch_window_predictor': 'model_torch.pt',
            'x_scaler': 'x_scaler.pkl',
            'y_scaler': 'y_scaler.pkl'
        },
        'performance': {
            'use_multiprocessing': True,
            'num_workers': 'auto',
            'chunk_buffer_size': 10,
            'cache_enabled': True,
            'cache_maxsize': 2,
            'use_gpu': True,
            'gpu_batch_size': 32
        },
        'validation': {
            'validate_inputs': True,
            'check_memory': True,
            'max_sequence_length': 300_000_000,
            'min_sequence_length': 100,
            'check_gpu_memory': True
        },
        'debug': {
            'verbose': True,
            'save_intermediate': False,
            'log_level': 'INFO',
            'profile_performance': True,
            'profile_gpu': True
        }
    }

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use default."""
    if config_path is None:
        return get_default_config()
    
    try:
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            print(f"[WARNING] Config file {config_path} not found, using default")
            return get_default_config()
        
        with open(config_path_obj, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults to ensure all sections exist
        default_config = get_default_config()
        for section in default_config:
            if section not in config:
                config[section] = default_config[section]
            elif isinstance(default_config[section], dict):
                for key in default_config[section]:
                    if key not in config[section]:
                        config[section][key] = default_config[section][key]
        
        config['_source'] = str(config_path_obj)
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config from {config_path}: {e}")
        print("[INFO] Using default configuration")
        return get_default_config()

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
        """Simple heuristic predictor."""
        len_diff_ratio = abs(len(seq1) - len(seq2)) / max(len(seq1), len(seq2))
        if len_diff_ratio > 0.3:
            base_window = max(100, min(len(seq1), len(seq2)) // 5)
        else:
            base_window = max(50, min(len(seq1), len(seq2)) // 10)
        
        return max(base_window, abs(len(seq1) - len(seq2)) + 10)
    
    if predictor_type == 'simple':
        return simple_window_predictor
    
    try:
        # Try to import the actual predictor module
        from alignment_pipeline.algorithms.predict_window import predict_window
        if predictor_type == 'gpu':
            try:
                from alignment_pipeline.algorithms.predict_window import predict_window_gpu
                return predict_window_gpu
            except ImportError:
                print("[WARNING] GPU predictor not available, falling back to CPU")
                from alignment_pipeline.algorithms.predict_window import predict_window_cpu
                return predict_window_cpu
        elif predictor_type == 'tf':
            from alignment_pipeline.algorithms.predict_window import predict_window_cpu
            return predict_window_cpu
        else:
            # Auto-detect based on CUDA
            import torch
            if torch.cuda.is_available():
                try:
                    from alignment_pipeline.algorithms.predict_window import predict_window_gpu
                    return predict_window_gpu
                except ImportError:
                    return predict_window_cpu
            else:
                return predict_window_cpu
    except ImportError as e:
        print(f"[WARNING] Could not load window predictor module: {e}")
        print("[INFO] Using simple window predictor")
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

def validate_input_files(config: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
    """Validate that input files exist and return missing/valid files."""
    fasta_dir = Path(config['io']['fasta_dir'])
    fasta1_path = fasta_dir / config['io']['default_fasta1']
    fasta2_path = fasta_dir / config['io']['default_fasta2']
    
    missing_files = []
    valid_files = []
    
    if not fasta1_path.exists():
        missing_files.append(str(fasta1_path))
    else:
        valid_files.append(str(fasta1_path))
    
    if not fasta2_path.exists():
        missing_files.append(str(fasta2_path))
    else:
        valid_files.append(str(fasta2_path))
    
    return len(missing_files) == 0, missing_files, valid_files

def run_pipeline(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> int:
    """Main pipeline entry point with GPU support."""
    # Load configuration
    config = load_config(config_path)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting alignment pipeline")
    logger.info(f"Configuration loaded from {config.get('_source', 'default')}")
    
    # Check input files
    files_valid, missing_files, valid_files = validate_input_files(config)
    
    if not files_valid:
        logger.error("Missing input files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        logger.error("\nPlease provide valid FASTA files to align.")
        logger.error("Run align-pipeline --help for usage information.")
        return 1
    
    logger.info(f"Input files found: {valid_files}")
    
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
    
    # Check versions if available
    try:
        from alignment_pipeline.diagnostics.version_checker import check_versions
        check_versions()
    except ImportError:
        logger.warning("Version checker not available, skipping version checks")
    
    # Performance monitoring if available
    try:
        from alignment_pipeline.diagnostics.performance import PerformanceMonitor
        perf_monitor = PerformanceMonitor()
        perf_monitor.start()
        use_perf_monitor = True
    except ImportError:
        logger.warning("Performance monitor not available")
        use_perf_monitor = False
    
    try:
        # Get file paths from config
        fasta_dir = Path(config['io']['fasta_dir'])
        fasta1 = fasta_dir / config['io']['default_fasta1']
        fasta2 = fasta_dir / config['io']['default_fasta2']
        
        logger.info(f"Aligning sequences:")
        logger.info(f"  Sequence 1: {fasta1}")
        logger.info(f"  Sequence 2: {fasta2}")
        
        # Validate inputs if requested
        if config['validation']['validate_inputs']:
            try:
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
            except ImportError:
                logger.warning("Validation module not available, skipping validation")
        
        # Import required modules
        try:
            from alignment_pipeline.core.compression import load_two_fasta_sequences
            from alignment_pipeline.core.chunking import (
                process_and_save_chunks_parallel,
                reload_and_merge_chunks
            )
            from alignment_pipeline.io.results_writer import save_all_results
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            return 1
        
        # Step 1: Load and compress sequences
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
        
        # Step 2: Process chunks with GPU acceleration
        logger.info("Processing chunks...")
        
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
            overlap=config['chunking']['overlap'],
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
        
        # Step 4: Save results
        logger.info("Saving results...")
        save_all_results(a1, a2, total_score, str(fasta1), str(fasta2))
        
        # Step 5: Optional visualizations
        if config.get('visualization', {}).get('enabled', True):
            try:
                from alignment_pipeline.visualization.visualizer import (
                    visualize_pipeline_results
                )
                logger.info("Creating visualizations...")
                visualize_pipeline_results(
                    final_a1=a1,
                    final_a2=a2,
                    total_score=total_score,
                    fasta1=str(fasta1),
                    fasta2=str(fasta2),
                    output_dir=config['io']['visualization_dir']
                )
            except ImportError as e:
                logger.warning(f"Visualization module not available: {e}")
        
        # Performance report
        if use_perf_monitor:
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
            
            if gpu_info.get('available', False) and 'gpu_info' in report:
                gpu_report = report['gpu_info']
                logger.info(f"GPU memory used: {gpu_report.get('max_memory_used_gb', 0):.2f} GB")
        else:
            logger.info("Pipeline completed successfully!")
        
        logger.info(f"Total score: {total_score}")
        
        # Print alignment snippet
        logger.info(f"\nAlignment snippet (first 100 columns):")
        logger.info(f"Seq1: {a1[:100]}")
        logger.info(f"Comp: {comp[:100]}")
        logger.info(f"Seq2: {a2[:100]}")
        
        # Save performance report if available
        if use_perf_monitor:
            report_path = Path(config['io']['logs_dir']) / 'performance_report.json'
            perf_monitor.save_report(report_path)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DNA Sequence Alignment Pipeline - High performance alignment with chunked processing, energy-based quantum-inspired scoring, and FNN-enabled window predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --fasta1 seq1.fa --fasta2 seq2.fa
  %(prog)s --config my_config.yaml --verbose
  %(prog)s --version --diagnostics
        """
    )
    
    parser.add_argument(
        '--fasta1',
        type=str,
        help='Path to first FASTA file'
    )
    
    parser.add_argument(
        '--fasta2',
        type=str,
        help='Path to second FASTA file'
    )
    
    parser.add_argument(
        '--fasta-dir',
        type=str,
        default='',
        help='Directory containing FASTA files'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Chunk size for processing'
    )
    
    parser.add_argument(
        '--gap-open',
        type=float,
        help='Gap opening penalty'
    )
    
    parser.add_argument(
        '--gap-extend',
        type=float,
        help='Gap extension penalty'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of worker processes (0 = auto)'
    )
    
    parser.add_argument(
        '--no-multiprocessing',
        action='store_true',
        help='Disable multiprocessing'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='Results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--diagnostics',
        action='store_true',
        help='Run diagnostics before pipeline'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    
    # Debug
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--save-chunks',
        action='store_true',
        help='Save intermediate chunks'
    )
    
    return parser.parse_args()

def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    if args.version:
        try:
            from alignment_pipeline.diagnostics.version_checker import print_version_report
            print_version_report()
        except ImportError:
            print("Version checker not available")
            print("Alignment Pipeline - Unknown version")
        return 0
    
    if args.diagnostics:
        try:
            from alignment_pipeline.diagnostics.version_checker import print_version_report
            print_version_report()
        except ImportError:
            print("Version checker not available")
        
        print("\n" + "="*70)
        print("Running additional diagnostics...")
        
        try:
            from alignment_pipeline.diagnostics.validation import (
                check_system_requirements,
                check_sequence_compatibility
            )
            
            config = get_default_config()
            
            meets_reqs, warnings = check_system_requirements(config)
            if warnings:
                print("\nSystem warnings:")
                for warning in warnings:
                    print(f"  ⚠ {warning}")
            
            if args.fasta1 and args.fasta2:
                compatible, msg = check_sequence_compatibility(args.fasta1, args.fasta2)
                print(f"\nSequence compatibility: {msg}")
        except ImportError:
            print("Diagnostics modules not available")
        
        print("\n" + "="*70)
        return 0
    
    # Setup CUDA early
    CUDA_AVAILABLE = setup_cuda()
    
    # Build config overrides from command line arguments
    config_overrides = {}
    
    if args.fasta1:
        fasta1_path = Path(args.fasta1)
        if fasta1_path.exists():
            config_overrides['io'] = config_overrides.get('io', {})
            config_overrides['io']['fasta_dir'] = str(fasta1_path.parent)
            config_overrides['io']['default_fasta1'] = fasta1_path.name
        else:
            print(f"[ERROR] FASTA file not found: {args.fasta1}")
            return 1
    
    if args.fasta2:
        fasta2_path = Path(args.fasta2)
        if fasta2_path.exists():
            config_overrides['io'] = config_overrides.get('io', {})
            if 'fasta_dir' not in config_overrides.get('io', {}):
                config_overrides['io']['fasta_dir'] = str(fasta2_path.parent)
            config_overrides['io']['default_fasta2'] = fasta2_path.name
        else:
            print(f"[ERROR] FASTA file not found: {args.fasta2}")
            return 1
    
    if args.fasta_dir:
        config_overrides['io'] = config_overrides.get('io', {})
        config_overrides['io']['fasta_dir'] = args.fasta_dir
    
    if args.chunk_size:
        config_overrides['chunking'] = config_overrides.get('chunking', {})
        config_overrides['chunking']['default_chunk_size'] = args.chunk_size
    
    if args.gap_open:
        config_overrides['alignment'] = config_overrides.get('alignment', {})
        config_overrides['alignment']['gap_open'] = args.gap_open
    
    if args.gap_extend:
        config_overrides['alignment'] = config_overrides.get('alignment', {})
        config_overrides['alignment']['gap_extend'] = args.gap_extend
    
    if args.workers is not None:
        config_overrides['performance'] = config_overrides.get('performance', {})
        if args.workers == 0:
            config_overrides['performance']['num_workers'] = 'auto'
        else:
            config_overrides['performance']['num_workers'] = args.workers
    
    if args.no_multiprocessing:
        config_overrides['performance'] = config_overrides.get('performance', {})
        config_overrides['performance']['use_multiprocessing'] = False
    
    if args.output_dir:
        config_overrides['io'] = config_overrides.get('io', {})
        config_overrides['io']['output_dir'] = args.output_dir
    
    if args.verbose:
        config_overrides['debug'] = config_overrides.get('debug', {})
        config_overrides['debug']['verbose'] = True
    
    if args.debug:
        config_overrides['debug'] = config_overrides.get('debug', {})
        config_overrides['debug']['log_level'] = 'DEBUG'
    
    if args.save_chunks:
        config_overrides['debug'] = config_overrides.get('debug', {})
        config_overrides['debug']['save_intermediate'] = True
    
    try:
        return run_pipeline(config_path=args.config, overrides=config_overrides)
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())