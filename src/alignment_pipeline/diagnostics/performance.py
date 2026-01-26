"""
Performance monitoring and profiling.
Author: Rowel Facunla
"""

import time
import psutil
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    start_time: float
    end_time: Optional[float] = None
    peak_memory_mb: float = 0.0
    cpu_percent: List[float] = None
    disk_io: Dict = None
    
    @property
    def total_time(self) -> float:
        """Total execution time in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

class PerformanceMonitor:
    """Monitor performance metrics during pipeline execution."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.metrics = PerformanceMetrics(start_time=time.time())
        self.monitoring = False
        self.thread = None
        self.cpu_samples = []
        self.memory_samples = []
        
    def start(self):
        """Start performance monitoring."""
        self.metrics = PerformanceMetrics(start_time=time.time())
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.metrics.end_time = time.time()
        
        # Calculate peak memory
        if self.memory_samples:
            self.metrics.peak_memory_mb = max(self.memory_samples)
        
        # Calculate average CPU
        if self.cpu_samples:
            self.metrics.cpu_percent = self.cpu_samples
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Sample CPU and memory
                cpu_percent = process.cpu_percent(interval=None)
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)
                
                time.sleep(self.sampling_interval)
            except (psutil.NoSuchProcess, KeyboardInterrupt):
                break
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        process = psutil.Process()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.metrics.start_time,
            'memory_mb': process.memory_info().rss / (1024 * 1024),
            'cpu_percent': process.cpu_percent(interval=None),
            'threads': process.num_threads(),
        }
    
    def get_report(self) -> Dict:
        """Get comprehensive performance report."""
        report = {
            'total_time_seconds': self.metrics.total_time,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'average_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            'start_time': datetime.fromtimestamp(self.metrics.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.metrics.end_time).isoformat() if self.metrics.end_time else None,
            'system_info': self._get_system_info(),
        }
        
        # Add memory timeline if available
        if self.memory_samples:
            report['memory_timeline'] = {
                'samples': len(self.memory_samples),
                'min_mb': min(self.memory_samples),
                'max_mb': max(self.memory_samples),
                'avg_mb': sum(self.memory_samples) / len(self.memory_samples),
            }
        
        return report
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'total_memory_mb': psutil.virtual_memory().total / (1024 * 1024),
                'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024),
                'disk_usage': psutil.disk_usage('.')._asdict(),
            }
        except Exception:
            return {'error': 'Could not get system info'}
    
    def save_report(self, filepath: str):
        """Save performance report to file."""
        report = self.get_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_report(self):
        """Print performance report to console."""
        report = self.get_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Total time: {report['total_time_seconds']:.2f} seconds")
        print(f"Peak memory: {report['peak_memory_mb']:.1f} MB")
        print(f"Average CPU: {report['average_cpu_percent']:.1f}%")
        
        if 'memory_timeline' in report:
            mem = report['memory_timeline']
            print(f"\nMemory usage:")
            print(f"  Samples: {mem['samples']}")
            print(f"  Minimum: {mem['min_mb']:.1f} MB")
            print(f"  Maximum: {mem['max_mb']:.1f} MB")
            print(f"  Average: {mem['avg_mb']:.1f} MB")
        
        sys_info = report['system_info']
        if 'error' not in sys_info:
            print(f"\nSystem information:")
            print(f"  CPU cores: {sys_info['cpu_count']}")
            print(f"  Total memory: {sys_info['total_memory_mb']:.0f} MB")
            print(f"  Available memory: {sys_info['available_memory_mb']:.0f} MB")
        
        print("="*60)

def profile_function(func):
    """Decorator to profile a function."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            monitor.stop()
            monitor.print_report()
        
        return result
    
    return wrapper

def benchmark_function(func, num_runs: int = 3, warmup: int = 1):
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        
    Returns:
        Benchmark results dictionary
    """
    import timeit
    import math  # For standard deviation calculation
    
    # Warmup runs
    for _ in range(warmup):
        func()
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)
    
    # Calculate statistics
    if len(times) == 0:
        return {
            'num_runs': num_runs,
            'warmup_runs': warmup,
            'times': times,
            'min_time': 0,
            'max_time': 0,
            'avg_time': 0,
            'std_time': 0,
        }
    
    avg_time = sum(times) / len(times)
    
    # Calculate standard deviation manually (no numpy dependency)
    if len(times) > 1:
        variance = sum((t - avg_time) ** 2 for t in times) / (len(times) - 1)
        std_time = math.sqrt(variance)
    else:
        std_time = 0.0
    
    return {
        'num_runs': num_runs,
        'warmup_runs': warmup,
        'times': times,
        'min_time': min(times) if times else 0,
        'max_time': max(times) if times else 0,
        'avg_time': avg_time,
        'std_time': std_time,
    }

__all__ = [
    'PerformanceMonitor',
    'profile_function',
    'benchmark_function',
]