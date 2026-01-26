#!/usr/bin/env python3
"""
Performance benchmarking script for alignment pipeline.
Author: Rowel Facunla
"""

import sys
import os
import json
import time
import statistics
import argparse
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

def benchmark_alignment_small(use_gpu: bool = False) -> Dict[str, Any]:
    """Benchmark small sequence alignment."""
    print("Testing small sequences (100-1000 bp)...")
    
    from alignment_pipeline.algorithms.nw_affine import nw_affine_chunk
    from alignment_pipeline.core.compression import compress_4bit_sequence, decompress_slice
    
    results = []
    
    # Test sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        # Create test sequences
        seq1 = "ACGT" * (size // 4)
        seq2 = seq1[:size-10] + "TTTT" + seq1[size-6:]
        
        # Setup parameters
        penalty_matrix = {('A', 'A'): 1, ('C', 'C'): 1, ('G', 'G'): 1, ('T', 'T'): 1}
        
        # Time alignment
        start_time = time.time()
        
        try:
            if use_gpu:
                # Try GPU-accelerated version
                from alignment_pipeline.cuda_extensions import CUDABeamPruner
                pruner = CUDABeamPruner()
                
                score, a1, comp, a2, gap_state = nw_affine_chunk(
                    seq1, seq2,
                    gap_open=-2,
                    gap_extend=-0.5,
                    penalty_matrix=penalty_matrix,
                    window_size=min(100, size // 2),
                    beam_width=50
                )
            else:
                score, a1, comp, a2, gap_state = nw_affine_chunk(
                    seq1, seq2,
                    gap_open=-2,
                    gap_extend=-0.5,
                    penalty_matrix=penalty_matrix,
                    window_size=min(100, size // 2),
                    beam_width=50
                )
            
            elapsed = time.time() - start_time
            
            results.append({
                'size': size,
                'time_seconds': elapsed,
                'score': score,
                'alignment_length': len(a1),
                'success': True
            })
            
            print(f"  Size {size:4d} bp: {elapsed:.4f}s")
            
        except Exception as e:
            results.append({
                'size': size,
                'time_seconds': None,
                'error': str(e),
                'success': False
            })
            print(f"  Size {size:4d} bp: Failed - {e}")
    
    return {
        'test_name': 'small_alignment',
        'use_gpu': use_gpu,
        'results': results
    }

def benchmark_compression(use_gpu: bool = False) -> Dict[str, Any]:
    """Benchmark compression operations."""
    print("Testing compression/decompression...")
    
    from alignment_pipeline.core.compression import compress_4bit_sequence, decompress_slice
    
    results = []
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        seq = "ACGT" * (size // 4)
        
        # Compression benchmark
        start = time.time()
        compressed, length = compress_4bit_sequence(seq)
        compress_time = time.time() - start
        
        # Decompression benchmark
        start = time.time()
        decompressed = decompress_slice(compressed, length, 0, length)
        decompress_time = time.time() - start
        
        # Verify
        success = decompressed == seq
        
        results.append({
            'size': size,
            'compress_time': compress_time,
            'decompress_time': decompress_time,
            'compression_ratio': len(compressed) / len(seq.encode('utf-8')),
            'success': success
        })
        
        print(f"  Size {size:6d} bp: Compress={compress_time:.4f}s, Decompress={decompress_time:.4f}s, Ratio={len(compressed)/len(seq.encode('utf-8')):.2f}")
    
    return {
        'test_name': 'compression',
        'use_gpu': use_gpu,
        'results': results
    }

def benchmark_chunking(use_gpu: bool = False) -> Dict[str, Any]:
    """Benchmark chunking operations."""
    print("Testing chunking operations...")
    
    from alignment_pipeline.algorithms.anchor_chaining import chains_to_chunks, chain_anchors
    from alignment_pipeline.algorithms.anchor_chaining import Anchor
    
    results = []
    
    # Create test anchors
    num_anchors_list = [100, 1000, 10000]
    
    for num_anchors in num_anchors_list:
        anchors = []
        for i in range(num_anchors):
            pos = i * 10
            anchors.append(Anchor(
                qpos=pos,
                tpos=pos,
                span=100,
                hash=i,
                diag=0
            ))
        
        # Time chaining
        start = time.time()
        chains = chain_anchors(anchors)
        chain_time = time.time() - start
        
        # Time chunking
        start = time.time()
        chunks = chains_to_chunks(chains, L1=num_anchors*10, L2=num_anchors*10, chunk_size=5000)
        chunk_time = time.time() - start
        
        results.append({
            'num_anchors': num_anchors,
            'chain_time': chain_time,
            'chunk_time': chunk_time,
            'num_chains': len(chains),
            'num_chunks': len(chunks),
            'success': True
        })
        
        print(f"  {num_anchors:6d} anchors: Chain={chain_time:.4f}s, Chunk={chunk_time:.4f}s, Chains={len(chains)}, Chunks={len(chunks)}")
    
    return {
        'test_name': 'chunking',
        'use_gpu': use_gpu,
        'results': results
    }

def benchmark_gpu_specific(use_gpu: bool = False) -> Dict[str, Any]:
    """Benchmark GPU-specific operations."""
    if not use_gpu:
        return {'test_name': 'gpu_operations', 'use_gpu': False, 'skipped': True}
    
    print("Testing GPU-specific operations...")
    
    results = []
    
    try:
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("  CUDA not available, skipping GPU benchmarks")
            return {
                'test_name': 'gpu_operations',
                'use_gpu': False,
                'skipped': True,
                'error': 'CUDA not available'
            }
        
        # Test tensor operations
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            # CPU operation
            start = time.time()
            cpu_tensor = torch.randn(size, size)
            cpu_result = torch.matmul(cpu_tensor, cpu_tensor.T)
            cpu_time = time.time() - start
            
            # GPU operation
            start = time.time()
            gpu_tensor = torch.randn(size, size).cuda()
            gpu_result = torch.matmul(gpu_tensor, gpu_tensor.T)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            results.append({
                'size': size,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'success': True
            })
            
            print(f"  Size {size:6d}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.2f}x")
    
    except Exception as e:
        results.append({
            'error': str(e),
            'success': False
        })
        print(f"  GPU operations failed: {e}")
    
    return {
        'test_name': 'gpu_operations',
        'use_gpu': use_gpu,
        'results': results
    }

def benchmark_end_to_end(test_data_dir: str, use_gpu: bool = False) -> Dict[str, Any]:
    """Benchmark end-to-end pipeline."""
    print("Testing end-to-end pipeline...")
    
    results = []
    
    # Look for test FASTA files
    test_dir = Path(test_data_dir)
    fasta_files = list(test_dir.glob("*.fa")) + list(test_dir.glob("*.fasta"))
    
    if len(fasta_files) < 2:
        print(f"  Need at least 2 FASTA files in {test_data_dir}")
        return {
            'test_name': 'end_to_end',
            'use_gpu': use_gpu,
            'results': [],
            'error': 'Insufficient test files'
        }
    
    # Use first two files
    fasta1, fasta2 = fasta_files[:2]
    
    try:
        from alignment_pipeline.pipeline.main_pipeline import main as pipeline_main
        
        # Run pipeline
        start = time.time()
        
        return_code = pipeline_main(
            config_path=None,
            overrides={
                'io': {
                    'default_fasta1': str(fasta1.name),
                    'default_fasta2': str(fasta2.name),
                    'fasta_dir': str(test_data_dir),
                    'output_dir': 'benchmark_results'
                },
                'alignment': {
                    'use_gpu_acceleration': use_gpu
                },
                'performance': {
                    'use_gpu': use_gpu,
                    'num_workers': 1  # Single worker for consistent benchmarking
                },
                'debug': {
                    'verbose': False,
                    'save_intermediate': False
                }
            }
        )
        
        elapsed = time.time() - start
        success = return_code == 0
        
        # Get sequence lengths
        from Bio import SeqIO
        seq1_len = len(next(SeqIO.parse(fasta1, "fasta")).seq)
        seq2_len = len(next(SeqIO.parse(fasta2, "fasta")).seq)
        
        results.append({
            'file1': str(fasta1.name),
            'file2': str(fasta2.name),
            'seq1_length': seq1_len,
            'seq2_length': seq2_len,
            'time_seconds': elapsed,
            'success': success,
            'return_code': return_code
        })
        
        print(f"  Files: {fasta1.name} ({seq1_len} bp) vs {fasta2.name} ({seq2_len} bp)")
        print(f"  Time: {elapsed:.2f}s, Success: {success}")
        
    except Exception as e:
        results.append({
            'error': str(e),
            'success': False
        })
        print(f"  End-to-end test failed: {e}")
    
    return {
        'test_name': 'end_to_end',
        'use_gpu': use_gpu,
        'results': results
    }

def run_comprehensive_benchmark(output_dir: str = "benchmark_results", 
                              use_gpu: bool = False,
                              test_data: str = None) -> Dict[str, Any]:
    """Run comprehensive benchmark suite."""
    
    print("=" * 70)
    print(f"ALIGNMENT PIPELINE BENCHMARK {'(GPU ACCELERATED)' if use_gpu else '(CPU ONLY)'}")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"GPU Mode: {use_gpu}")
    print("=" * 70)
    
    all_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'use_gpu': use_gpu,
        'python_version': sys.version,
        'platform': sys.platform,
        'benchmarks': []
    }
    
    # Run benchmarks
    benchmarks = [
        benchmark_alignment_small,
        benchmark_compression,
        benchmark_chunking,
        benchmark_gpu_specific,
    ]
    
    if test_data:
        benchmarks.append(lambda: benchmark_end_to_end(test_data, use_gpu))
    
    for benchmark_func in benchmarks:
        try:
            result = benchmark_func(use_gpu)
            all_results['benchmarks'].append(result)
        except Exception as e:
            print(f"Benchmark {benchmark_func.__name__} failed: {e}")
            all_results['benchmarks'].append({
                'test_name': benchmark_func.__name__,
                'error': str(e),
                'success': False
            })
    
    # Calculate summary statistics
    total_time = 0
    successful_tests = 0
    total_tests = 0
    
    for benchmark in all_results['benchmarks']:
        if benchmark.get('results'):
            for result in benchmark['results']:
                total_tests += 1
                if result.get('success', False):
                    successful_tests += 1
                    if 'time_seconds' in result and result['time_seconds']:
                        total_time += result['time_seconds']
    
    all_results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
        'total_time_seconds': total_time
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 70)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / f"benchmark_{'gpu' if use_gpu else 'cpu'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_results

def compare_gpu_vs_cpu(output_dir: str = "benchmark_results", 
                      test_data: str = None,
                      iterations: int = 3) -> Dict[str, Any]:
    """Compare GPU vs CPU performance."""
    
    print("=" * 70)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("=" * 70)
    
    comparison_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'iterations': iterations,
        'cpu_results': [],
        'gpu_results': [],
        'speedups': {}
    }
    
    # Run CPU benchmarks
    print("\nRunning CPU benchmarks...")
    cpu_times = []
    for i in range(iterations):
        print(f"\nCPU Iteration {i+1}/{iterations}:")
        results = run_comprehensive_benchmark(
            output_dir=os.path.join(output_dir, f"cpu_iteration_{i+1}"),
            use_gpu=False,
            test_data=test_data
        )
        cpu_times.append(results['summary']['total_time_seconds'])
        comparison_results['cpu_results'].append(results)
    
    # Run GPU benchmarks
    print("\nRunning GPU benchmarks...")
    gpu_times = []
    for i in range(iterations):
        print(f"\nGPU Iteration {i+1}/{iterations}:")
        results = run_comprehensive_benchmark(
            output_dir=os.path.join(output_dir, f"gpu_iteration_{i+1}"),
            use_gpu=True,
            test_data=test_data
        )
        gpu_times.append(results['summary']['total_time_seconds'])
        comparison_results['gpu_results'].append(results)
    
    # Calculate statistics
    if cpu_times and gpu_times:
        avg_cpu = statistics.mean(cpu_times)
        avg_gpu = statistics.mean(gpu_times)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
        
        comparison_results['speedups'] = {
            'average_cpu_time': avg_cpu,
            'average_gpu_time': avg_gpu,
            'average_speedup': speedup,
            'min_speedup': min([c/g for c, g in zip(cpu_times, gpu_times) if g > 0]),
            'max_speedup': max([c/g for c, g in zip(cpu_times, gpu_times) if g > 0]),
            'cpu_times': cpu_times,
            'gpu_times': gpu_times
        }
        
        # Print comparison
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON RESULTS")
        print("=" * 70)
        print(f"Average CPU Time: {avg_cpu:.2f}s")
        print(f"Average GPU Time: {avg_gpu:.2f}s")
        print(f"Average Speedup: {speedup:.2f}x")
        print(f"Minimum Speedup: {comparison_results['speedups']['min_speedup']:.2f}x")
        print(f"Maximum Speedup: {comparison_results['speedups']['max_speedup']:.2f}x")
        print("=" * 70)
        
        if speedup > 1:
            print(f"✅ GPU acceleration provides {speedup:.1f}x speedup over CPU")
        else:
            print("⚠ GPU acceleration does not provide speedup (check GPU setup)")
    
    # Save comparison results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    comparison_file = output_path / f"gpu_vs_cpu_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nComparison results saved to: {comparison_file}")
    
    return comparison_results

def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="Alignment Pipeline Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --cpu                     # Run CPU benchmarks
  %(prog)s --gpu                     # Run GPU benchmarks
  %(prog)s --compare --iterations 5  # Compare GPU vs CPU
  %(prog)s --test-data examples/     # Use specific test data
  %(prog)s --sequences examples/     # Alias for --test-data (deprecated)
        """
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Run CPU-only benchmarks'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Run GPU-accelerated benchmarks'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare GPU vs CPU performance'
    )
    
    # Accept both --test-data and --sequences for backward compatibility
    parser.add_argument(
        '--test-data',
        type=str,
        default='examples',
        help='Directory containing test FASTA files'
    )
    
    parser.add_argument(
        '--sequences',
        type=str,
        dest='test_data',  # Map to test_data
        help='Alias for --test-data (deprecated)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Directory to save benchmark results'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of iterations for comparison tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check for deprecated --sequences argument
    if hasattr(args, 'sequences') and args.sequences and not args.test_data:
        args.test_data = args.sequences
        print(f"⚠ Note: --sequences is deprecated, use --test-data instead")
    
    # If no mode specified, run both CPU and comparison
    if not any([args.cpu, args.gpu, args.compare]):
        args.compare = True
    
    results = {}
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run requested benchmarks
    if args.cpu:
        print("Running CPU benchmarks...")
        results['cpu'] = run_comprehensive_benchmark(
            output_dir=os.path.join(args.output_dir, 'cpu'),
            use_gpu=False,
            test_data=args.test_data
        )
    
    if args.gpu:
        print("Running GPU benchmarks...")
        results['gpu'] = run_comprehensive_benchmark(
            output_dir=os.path.join(args.output_dir, 'gpu'),
            use_gpu=True,
            test_data=args.test_data
        )
    
    if args.compare:
        print("Running GPU vs CPU comparison...")
        results['comparison'] = compare_gpu_vs_cpu(
            output_dir=os.path.join(args.output_dir, 'comparison'),
            test_data=args.test_data,
            iterations=args.iterations
        )
    
    # Print final recommendations
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    if 'comparison' in results and 'speedups' in results['comparison']:
        speedup = results['comparison']['speedups'].get('average_speedup', 1)
        if speedup > 1.5:
            print(f"✅ STRONG RECOMMENDATION: Use GPU acceleration ({speedup:.1f}x speedup)")
        elif speedup > 1.0:
            print(f"⚠ MODERATE RECOMMENDATION: GPU provides some speedup ({speedup:.1f}x)")
        else:
            print("⚠ RECOMMENDATION: Use CPU-only mode (GPU not providing speedup)")
    else:
        print("Benchmark results available in:", args.output_dir)
    
    print("\nTo use the pipeline:")
    print("  align-pipeline --fasta1 examples/test_fasta_1.fa --fasta2 examples/test_fasta_2.fa")
    print("\nFor GPU acceleration:")
    print("  align-pipeline --config config/pipeline_config_gpu.yaml")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())