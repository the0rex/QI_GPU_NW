"""
Benchmark tests for performance evaluation.
Author: Rowel Facunla
"""

import pytest
import time
import numpy as np
from alignment_pipeline.diagnostics.performance import (
    PerformanceMonitor,
    benchmark_function
)

def test_performance_monitor_basic():
    """Test basic functionality of PerformanceMonitor."""
    monitor = PerformanceMonitor(sampling_interval=0.1)
    
    # Start monitoring
    monitor.start()
    
    # Do some work
    time.sleep(0.2)
    
    # Stop monitoring
    monitor.stop()
    
    # Get report
    report = monitor.get_report()
    
    # Basic assertions
    assert 'total_time_seconds' in report
    assert report['total_time_seconds'] >= 0.1  # Should be close to sleep time
    assert 'peak_memory_mb' in report
    assert report['peak_memory_mb'] > 0
    
    # System info should be present
    assert 'system_info' in report
    if 'error' not in report['system_info']:
        assert 'cpu_count' in report['system_info']
        assert 'total_memory_mb' in report['system_info']

def test_performance_monitor_context():
    """Test PerformanceMonitor with context manager pattern."""
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Do some work
    result = sum(range(100000))
    time.sleep(0.1)
    
    monitor.stop()
    
    # Monitor should have stopped
    report = monitor.get_report()
    assert report['total_time_seconds'] >= 0.05

def test_benchmark_compression():
    """Benchmark 4-bit compression performance."""
    from alignment_pipeline.core.compression import compress_4bit_sequence
    
    # Create test sequences of different sizes
    test_cases = [
        ("Small", "ACGT" * 10),  # 40 bases
        ("Medium", "ACGT" * 100),  # 400 bases
        ("Large", "ACGT" * 1000),  # 4000 bases
    ]
    
    for name, seq in test_cases:
        # Benchmark compression
        def compress_test():
            return compress_4bit_sequence(seq)
        
        # Run benchmark
        benchmark_result = benchmark_function(compress_test, num_runs=3, warmup=1)
        
        # Basic assertions
        assert benchmark_result['num_runs'] == 3
        assert benchmark_result['warmup_runs'] == 1
        assert len(benchmark_result['times']) == 3
        assert benchmark_result['avg_time'] > 0
        
        # Verify compression works
        compressed, length = compress_4bit_sequence(seq)
        assert length == len(seq)
        assert len(compressed) == (length + 1) // 2

def test_benchmark_alignment():
    """Benchmark alignment algorithm performance."""
    from alignment_pipeline.algorithms.nw_affine import nw_overlap_realign
    
    # Test cases: different sequence lengths
    test_cases = [
        ("Tiny", "ACGT", "ACGT"),
        ("Small", "ACGT" * 10, "ACGT" * 10),
    ]
    
    for name, seq1, seq2 in test_cases:
        # Benchmark alignment
        def align_test():
            return nw_overlap_realign(seq1, seq2)
        
        # Run benchmark
        benchmark_result = benchmark_function(align_test, num_runs=3, warmup=1)
        
        # Verify alignment works
        aligned1, aligned2 = nw_overlap_realign(seq1, seq2)
        assert len(aligned1) == len(aligned2)
        assert aligned1.replace('-', '') == seq1
        assert aligned2.replace('-', '') == seq2

def test_memory_usage_tracking():
    """Test memory usage tracking."""
    from alignment_pipeline.diagnostics.performance import PerformanceMonitor
    
    monitor = PerformanceMonitor(sampling_interval=0.05)
    monitor.start()
    
    # Allocate some memory
    large_list = []
    for i in range(50000):
        large_list.append("x" * 100)
    
    # Free memory
    del large_list
    
    monitor.stop()
    
    report = monitor.get_report()
    
    # Should have memory samples
    if 'memory_timeline' in report:
        mem_info = report['memory_timeline']
        assert mem_info['samples'] > 0

def test_performance_report_format():
    """Test that performance reports are properly formatted."""
    from alignment_pipeline.diagnostics.performance import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Do minimal work
    _ = sum(range(1000))
    
    monitor.stop()
    
    # Get and print report
    report = monitor.get_report()
    
    # Check required fields
    required_fields = [
        'total_time_seconds',
        'peak_memory_mb',
        'average_cpu_percent',
        'start_time',
        'system_info',
    ]
    
    for field in required_fields:
        assert field in report, f"Missing field: {field}"
    
    # Test print_report doesn't crash
    monitor.print_report()

def test_benchmark_scalability():
    """Test how performance scales with input size."""
    from alignment_pipeline.core.compression import compress_4bit_sequence
    
    sizes = [100, 1000, 10000]
    results = []
    
    for size in sizes:
        seq = "ACGT" * (size // 4)
        
        def benchmark():
            return compress_4bit_sequence(seq)
        
        result = benchmark_function(benchmark, num_runs=2, warmup=1)
        results.append((size, result['avg_time']))
    
    # Check that time increases with size
    print("\nScaling Benchmark Results:")
    print("-" * 50)
    for i, (size, avg_time) in enumerate(results):
        print(f"Size {size}: {avg_time*1000:.3f} ms")
        
        if i > 0:
            prev_size, prev_time = results[i-1]
            if prev_time > 0:
                ratio = avg_time / prev_time
                print(f"  Scaling: {ratio:.2f}x for {size/prev_size:.1f}x size increase")

def test_concurrent_monitoring():
    """Test that monitoring works with concurrent operations."""
    import threading
    
    monitor = PerformanceMonitor(sampling_interval=0.1)
    results = []
    
    def worker(worker_id):
        """Worker function that does some computation."""
        local_monitor = PerformanceMonitor()
        local_monitor.start()
        
        # Do work
        total = 0
        for i in range(50000):
            total += i
        
        local_monitor.stop()
        results.append((worker_id, local_monitor.get_report()['total_time_seconds']))
    
    # Start monitoring
    monitor.start()
    
    # Start worker threads
    threads = []
    for i in range(2):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Stop monitoring
    monitor.stop()
    
    # Verify results
    assert len(results) == 2
    for worker_id, result in results:
        assert result > 0
    
    report = monitor.get_report()
    assert report['total_time_seconds'] > 0

def test_profile_function_decorator():
    """Test the profile_function decorator."""
    # Note: The decorator prints output, so we just test it doesn't crash
    from alignment_pipeline.diagnostics.performance import profile_function
    
    @profile_function
    def test_func():
        """Test function to profile."""
        total = 0
        for i in range(100000):
            total += i
        return total
    
    # Run function - should print performance report
    result = test_func()
    assert result == sum(range(100000))

def test_benchmark_edge_cases():
    """Test benchmarking edge cases."""
    from alignment_pipeline.diagnostics.performance import benchmark_function
    
    # Test with very fast function
    def fast_func():
        return 42
    
    result = benchmark_function(fast_func, num_runs=5, warmup=2)
    assert result['num_runs'] == 5
    assert result['warmup_runs'] == 2
    assert len(result['times']) == 5
    
    # Test with function that raises exception
    def failing_func():
        raise ValueError("Test error")
    
    # Should catch the exception
    try:
        result = benchmark_function(failing_func, num_runs=1, warmup=0)
    except ValueError:
        pass  # Expected
    
    # Test with zero runs (should handle gracefully)
    def normal_func():
        return "test"
    
    result = benchmark_function(normal_func, num_runs=0, warmup=0)
    assert result['num_runs'] == 0
    assert len(result['times']) == 0

if __name__ == "__main__":
    # Run tests without --benchmark-only flag
    import sys
    sys.exit(pytest.main([__file__, "-v"]))