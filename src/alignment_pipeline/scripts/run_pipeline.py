"""
Command-line interface for the alignment pipeline.
Author: Rowel Facunla
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="DNA Sequence Alignment Pipeline - High performance alignment with chunked processing, energy-based quantum-inspired scoring, and FNN-enabled window predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic alignment
  %(prog)s --fasta1 seq1.fa --fasta2 seq2.fa
  
  # Align specific chromosomes
  %(prog)s --fasta1 GRCh38.fasta --chrom1 chr1 --fasta2 Pan_tro_3.0.fasta --chrom2 chr2
  
  # Align combined chromosomes
  %(prog)s --fasta1 GRCh38.fasta --chrom1 "chr1+chr2" --fasta2 Pan_tro_3.0.fasta --chrom2 "chr1+chr2+chr3"
  
  # List chromosomes in a FASTA file
  %(prog)s --list-chromosomes Pan_tro_3.0.fasta
  
  # List chromosomes in a specific range
  %(prog)s --list-chromosomes GRCh38.fasta --chromosome-range "1:10"
  %(prog)s --list-chromosomes GRCh38.fasta --chromosome-range "chr1:chr10"
  %(prog)s --list-chromosomes GRCh38.fasta --chromosome-range "X:Y"
  
  # Other options
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
    
    # Chromosome alignment options
    parser.add_argument(
        '--chrom1',
        type=str,
        help='Chromosome name(s) from first FASTA file (e.g., "chr1", "chr1+chr2", "chrX+chrY")'
    )
    
    parser.add_argument(
        '--chrom2',
        type=str,
        help='Chromosome name(s) from second FASTA file (e.g., "chr2", "chr1+chr2+chr3", "chrX")'
    )
    
    # Chromosome listing options
    parser.add_argument(
        '--list-chromosomes',
        metavar='FASTA_FILE',
        type=str,
        help='List all chromosome names in a FASTA file'
    )
    
    parser.add_argument(
        '--list-limit',
        type=int,
        default=0,
        help='Limit number of chromosomes to display (0 for all)'
    )
    
    parser.add_argument(
        '--chromosome-range',
        type=str,
        help='Range of chromosomes to display (e.g., "1:10", "chr1:chr10", "X:Y")'
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
    
    # Parse the arguments - this will handle --help automatically and exit
    args = parser.parse_args()
    
    # Handle version check - minimal imports
    if args.version:
        try:
            from alignment_pipeline.diagnostics.version_checker import print_version_report
        except ImportError as e:
            # Try lazy import first
            try:
                from src import lazy_import_all
                lazy_import_all()
                from alignment_pipeline.diagnostics.version_checker import print_version_report
            except ImportError as e2:
                print(f"ERROR: Could not import version checker: {e}")
                return 1
        print_version_report()
        return 0
    
    # Handle chromosome listing if requested
    if args.list_chromosomes:
        try:
            from alignment_pipeline.pipeline.main_pipeline import handle_chromosome_listing
            return handle_chromosome_listing(args)
        except ImportError as e:
            print(f"ERROR: Could not import chromosome listing module: {e}")
            return 1
    
    # Handle diagnostics - lazy import
    if args.diagnostics:
        # First lazy import all modules
        try:
            from src import lazy_import_all
            lazy_import_all()
            
            import yaml
            from alignment_pipeline.diagnostics.version_checker import print_version_report
            from alignment_pipeline.diagnostics.validation import (
                check_system_requirements,
                check_sequence_compatibility
            )
        except ImportError as e:
            print(f"ERROR: Could not import diagnostics modules: {e}")
            return 1
    
    # If we reach here, we need to run the alignment pipeline
    # First import pathlib for building config_overrides
    from pathlib import Path
    
    # Build config_overrides based on arguments
    config_overrides = {}
    
    if args.fasta1:
        fasta1_path = Path(args.fasta1)
        if fasta1_path.exists():
            config_overrides['io'] = config_overrides.get('io', {})
            config_overrides['io']['fasta_dir'] = str(fasta1_path.parent)
            config_overrides['io']['default_fasta1'] = fasta1_path.name
        else:
            print(f"ERROR: FASTA file not found: {args.fasta1}")
            return 1
    
    if args.fasta2:
        fasta2_path = Path(args.fasta2)
        if fasta2_path.exists():
            config_overrides['io'] = config_overrides.get('io', {})
            if 'fasta_dir' not in config_overrides.get('io', {}):
                config_overrides['io']['fasta_dir'] = str(fasta2_path.parent)
            config_overrides['io']['default_fasta2'] = fasta2_path.name
        else:
            print(f"ERROR: FASTA file not found: {args.fasta2}")
            return 1
    
    if args.fasta_dir:
        config_overrides['io'] = config_overrides.get('io', {})
        config_overrides['io']['fasta_dir'] = args.fasta_dir
    
    # Add chromosome alignment overrides
    if args.chrom1 and args.chrom2:
        config_overrides['io'] = config_overrides.get('io', {})
        config_overrides['io']['chrom1'] = args.chrom1
        config_overrides['io']['chrom2'] = args.chrom2
    
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
    
    # Now import and run the pipeline - only if alignment is needed
    try:
        from alignment_pipeline.pipeline.main_pipeline import main as pipeline_main
    except ImportError as e:
        print(f"ERROR: Could not import pipeline module: {e}")
        print("Make sure the package is installed correctly.")
        return 1
    
    try:
        return pipeline_main(config_path=args.config, overrides=config_overrides)
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())