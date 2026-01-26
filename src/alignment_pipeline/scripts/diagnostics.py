#!/usr/bin/env python3
"""
Script to run diagnostics on the alignment pipeline.
Author: Rowel Facunla
"""

import argparse
import sys
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostics on the alignment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --full                     # Run full diagnostics
  %(prog)s --check-all                # Same as --full (alias)
  %(prog)s --check-fasta seq1.fa seq2.fa
  %(prog)s --benchmark                # Run performance benchmarks
  %(prog)s --config my_config.yaml    # Validate configuration
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full diagnostics including system tests'
    )
    
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Alias for --full (run full diagnostics)'
    )
    
    parser.add_argument(
        '--check-fasta',
        nargs=2,
        metavar=('FASTA1', 'FASTA2'),
        help='Check FASTA file compatibility'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarks'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file to validate'
    )
    
    parser.add_argument(
        '--gpu-check',
        action='store_true',
        help='Check GPU availability and configuration'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick diagnostics (version check only)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Always run version check (unless quick mode)
    if not args.quick:
        from alignment_pipeline.diagnostics.version_checker import print_version_report
        print_version_report()
    
    # Handle --check-all as alias for --full
    if args.check_all:
        args.full = True
    
    # Check configuration if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                from alignment_pipeline.diagnostics.validation import validate_configuration
                is_valid, errors = validate_configuration(config)
                
                print("\n" + "="*70)
                print("CONFIGURATION VALIDATION")
                print("="*70)
                
                if is_valid:
                    print("✅ Configuration is valid")
                else:
                    print("❌ Configuration errors:")
                    for error in errors:
                        print(f"  - {error}")
                
            except Exception as e:
                print(f"ERROR: Could not validate configuration: {e}")
        else:
            print(f"ERROR: Configuration file not found: {config_path}")
    
    # Check FASTA files if provided
    if args.check_fasta:
        fasta1, fasta2 = args.check_fasta
        
        from alignment_pipeline.diagnostics.validation import check_sequence_compatibility
        from alignment_pipeline.io.fasta_reader import validate_fasta_file
        
        print("\n" + "="*70)
        print("FASTA FILE CHECK")
        print("="*70)
        
        # Validate individual files
        for i, fasta in enumerate([fasta1, fasta2], 1):
            is_valid, msg = validate_fasta_file(fasta)
            status = "✅" if is_valid else "❌"
            print(f"{status} FASTA {i} ({fasta}): {msg}")
        
        # Check compatibility
        compatible, msg = check_sequence_compatibility(fasta1, fasta2)
        status = "✅" if compatible else "❌"
        print(f"\n{status} Sequence compatibility: {msg}")
    
    # Run full diagnostics (or --check-all)
    if args.full or args.check_all:
        print("\n" + "="*70)
        print("FULL DIAGNOSTICS")
        print("="*70)
        
        # Load default config for system checks
        default_config_path = Path(__file__).parent.parent / "alignment_pipeline" / "config" / "pipeline_config.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Check system requirements
        from alignment_pipeline.diagnostics.validation import check_system_requirements
        meets_reqs, warnings = check_system_requirements(config)
        
        if warnings:
            print("\nSystem requirements:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        else:
            print("\n✅ All system requirements met")
        
        # Check C++ extensions
        print("\nC++ extensions:")
        try:
            from alignment_pipeline.cpp_extensions import _compress_cpp
            print("  ✅ 4-bit compression extension available")
        except ImportError:
            print("  ❌ 4-bit compression extension not available")
        
        try:
            from alignment_pipeline.cpp_extensions import _syncmer_cpp
            print("  ✅ Syncmer/strobemer extension available")
        except ImportError:
            print("  ❌ Syncmer/strobemer extension not available")
        
        # Check disk space
        from alignment_pipeline.io.file_handler import check_disk_space
        has_space, available_gb, required_gb = check_disk_space('.', 10.0)
        if has_space:
            print(f"\n✅ Disk space: {available_gb:.1f} GB available")
        else:
            print(f"\n❌ Disk space: Only {available_gb:.1f} GB available, {required_gb:.1f} GB required")
    
    # Check GPU if requested
    if args.gpu_check:
        print("\n" + "="*70)
        print("GPU DIAGNOSTICS")
        print("="*70)
        
        try:
            # Try to import GPU test function
            from alignment_pipeline.scripts.gpu_test import test_gpu_availability
            test_gpu_availability()
        except ImportError:
            print("GPU test module not available. Checking basic GPU support...")
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
                    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                else:
                    print("❌ PyTorch CUDA not available")
            except ImportError:
                print("❌ PyTorch not installed")
    
    # Run benchmarks
    if args.benchmark:
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARKS")
        print("="*70)
        
        try:
            from alignment_pipeline.diagnostics.performance import benchmark_function
            
            # Define benchmark functions
            def benchmark_compression():
                from alignment_pipeline.core.compression import compress_4bit_sequence
                test_seq = "ACGT" * 1000  # 4000 bp
                compress_4bit_sequence(test_seq)
            
            def benchmark_alignment():
                from alignment_pipeline.algorithms.nw_affine import nw_overlap_realign
                seq1 = "ACGT" * 100
                seq2 = "ACGT" * 100
                nw_overlap_realign(seq1, seq2)
            
            # Run benchmarks
            print("\nBenchmarking 4-bit compression...")
            comp_results = benchmark_function(benchmark_compression, num_runs=5)
            print(f"  Average time: {comp_results['avg_time']*1000:.1f} ms")
            
            print("\nBenchmarking small alignment...")
            align_results = benchmark_function(benchmark_alignment, num_runs=5)
            print(f"  Average time: {align_results['avg_time']*1000:.1f} ms")
            
        except Exception as e:
            print(f"ERROR: Could not run benchmarks: {e}")
    
    if not args.quick:
        print("\n" + "="*70)
        print("Diagnostics complete")
        print("="*70)
        print("\nFor more detailed benchmarks, use: align-benchmark")
        print("For GPU testing, use: align-gpu-test")
        print("For configuration generation, use: align-config")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())