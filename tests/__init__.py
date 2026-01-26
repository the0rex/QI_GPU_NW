import sys
from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

__version__ = "1.0.0"
__author__ = "Rowel Facunla"
__description__ = "Test suite for Alignment Pipeline"

try:
    from .test_alignment import (
        test_nw_overlap_realign,
        test_nw_overlap_realign_with_gaps,
        test_detect_gap_explosion,
        test_iupac_matching,
        test_simple_alignment,
    )
    
    from .test_compression import (
        test_compress_4bit_sequence,
        test_compress_even_length,
        test_compress_odd_length,
        test_decompress_slice,
        test_decompress_base,
        test_iupac_compression,
        test_compression_memory_efficiency,
        test_load_two_fasta_sequences,
        test_compression_consistency,
        test_error_handling,
        test_performance_small_sequences,
    )
    
    from .test_chunking import (
        test_anchor_creation,
        test_chunk_creation,
        test_enforce_equal_span,
        test_sort_anchors,
        test_chain_anchors,
        test_safe_regression,
        test_map_q_to_t,
        test_tile_chain,
        test_fallback_tiles,
        test_chains_to_chunks,
        test_chains_to_chunks_no_chains,
        test_chains_to_chunks_single_chain,
        test_chains_to_chunks_with_gap,
        test_debug_functions,
        test_anchor_chaining_edge_cases,
        test_chunk_span_consistency,
    )
    
    from .test_integration import (
        test_compression_integration,
        test_fasta_loading_integration,
        test_alignment_algorithm_integration,
        test_chunking_integration,
        test_seeding_integration,
        test_pipeline_config_integration,
        test_diagnostics_integration,
        test_visualization_integration,
        test_end_to_end_small_alignment,
        test_error_handling_integration,
    )
    
    TEST_MODULES_AVAILABLE = True
    
except ImportError as e:
    TEST_MODULES_AVAILABLE = False
    _import_error = str(e)

TEST_CATEGORIES = {
    'alignment': 'Alignment algorithm tests',
    'compression': '4-bit compression tests',
    'chunking': 'Anchor chaining and chunking tests',
    'integration': 'Integration and end-to-end tests',
    'diagnostics': 'Diagnostic and validation tests',
    'performance': 'Performance benchmark tests',
}

def list_tests():
    """List all available tests by category."""
    if not TEST_MODULES_AVAILABLE:
        print(f"Test modules not available: {_import_error}")
        return
    
    print("Available Tests")
    print("=" * 60)
    
    for category, description in TEST_CATEGORIES.items():
        print(f"\n{category.upper()}: {description}")
        print("-" * 40)
        
        # List tests for each category
        if category == 'alignment':
            tests = [
                'test_nw_overlap_realign',
                'test_nw_overlap_realign_with_gaps',
                'test_detect_gap_explosion',
                'test_iupac_matching',
                'test_simple_alignment',
            ]
        elif category == 'compression':
            tests = [
                'test_compress_4bit_sequence',
                'test_compress_even_length',
                'test_compress_odd_length',
                'test_decompress_slice',
                'test_decompress_base',
                'test_iupac_compression',
                'test_compression_memory_efficiency',
                'test_load_two_fasta_sequences',
                'test_compression_consistency',
                'test_error_handling',
                'test_performance_small_sequences',
            ]
        elif category == 'chunking':
            tests = [
                'test_anchor_creation',
                'test_chunk_creation',
                'test_enforce_equal_span',
                'test_sort_anchors',
                'test_chain_anchors',
                'test_safe_regression',
                'test_map_q_to_t',
                'test_tile_chain',
                'test_fallback_tiles',
                'test_chains_to_chunks',
                'test_chains_to_chunks_no_chains',
                'test_chains_to_chunks_single_chain',
                'test_chains_to_chunks_with_gap',
                'test_debug_functions',
                'test_anchor_chaining_edge_cases',
                'test_chunk_span_consistency',
            ]
        elif category == 'integration':
            tests = [
                'test_compression_integration',
                'test_fasta_loading_integration',
                'test_alignment_algorithm_integration',
                'test_chunking_integration',
                'test_seeding_integration',
                'test_pipeline_config_integration',
                'test_diagnostics_integration',
                'test_visualization_integration',
                'test_end_to_end_small_alignment',
                'test_error_handling_integration',
            ]
        else:
            tests = []
        
        for test in tests:
            print(f"  • {test}")
    
    print("\n" + "=" * 60)
    print("Run tests with: pytest tests/ -v")
    print("Run specific test: pytest tests/test_alignment.py::test_nw_overlap_realign -v")

def run_all_tests():
    """Run all tests using pytest."""
    import subprocess
    import sys
    
    print("Running all tests...")
    print("=" * 60)
    
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    return result.returncode

def run_category(category: str):
    """Run tests in a specific category."""
    import subprocess
    import sys
    
    if category not in TEST_CATEGORIES:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(TEST_CATEGORIES.keys())}")
        return 1
    
    print(f"Running {category} tests...")
    print("=" * 60)
    
    if category == 'alignment':
        test_file = "tests/test_alignment.py"
    elif category == 'compression':
        test_file = "tests/test_compression.py"
    elif category == 'chunking':
        test_file = "tests/test_chunking.py"
    elif category == 'integration':
        test_file = "tests/test_integration.py"
    else:
        print(f"No specific test file for category: {category}")
        return 1
    
    result = subprocess.run([sys.executable, "-m", "pytest", test_file, "-v"])
    return result.returncode

def create_test_data():
    """Create test data for running tests."""
    import tempfile
    import os
    
    test_dir = tempfile.mkdtemp(prefix="alignment_pipeline_test_")
    print(f"Created test directory: {test_dir}")
    
    # Create test FASTA files
    fasta1 = os.path.join(test_dir, "test1.fa")
    fasta2 = os.path.join(test_dir, "test2.fa")
    
    with open(fasta1, 'w') as f:
        f.write(">test_sequence_1\n")
        f.write("ACGTACGTACGTACGTACGTACGTACGTACGT\n")
        f.write("GATCGATCGATCGATCGATCGATCGATCGATC\n")
    
    with open(fasta2, 'w') as f:
        f.write(">test_sequence_2\n")
        f.write("ACGTACGTACGTACGTACGTACGTACGTACGT\n")
        f.write("GATCGATCGATCGATCGATCGATCGATCGATC\n")
        f.write("TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC\n")
    
    print("Created test FASTA files:")
    print(f"  {fasta1}")
    print(f"  {fasta2}")
    
    return test_dir

def benchmark_tests():
    """Run performance benchmarks."""
    import time
    import pytest
    
    print("Running performance benchmarks...")
    print("=" * 60)
    
    benchmarks = [
        ("test_compression_integration", test_compression_integration),
        ("test_alignment_algorithm_integration", test_alignment_algorithm_integration),
        ("test_chunking_integration", test_chunking_integration),
    ]
    
    results = {}
    for name, test_func in benchmarks:
        print(f"\nBenchmarking {name}...")
        start_time = time.time()
        
        try:
            # Run test 10 times for better timing
            for _ in range(10):
                test_func()
            elapsed = (time.time() - start_time) / 10
            results[name] = {
                'status': 'PASS',
                'time_per_run': elapsed,
                'time_total': elapsed * 10,
            }
            print(f"  ✓ Average time: {elapsed:.3f}s")
        except Exception as e:
            results[name] = {
                'status': 'FAIL',
                'error': str(e),
            }
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Benchmark Results:")
    for name, result in results.items():
        if result['status'] == 'PASS':
            print(f"  {name:40} {result['time_per_run']:.3f}s per run")
        else:
            print(f"  {name:40} FAILED: {result['error']}")
    
    return results

# Export for easy access
__all__ = [
    # Functions
    'list_tests',
    'run_all_tests',
    'run_category',
    'create_test_data',
    'benchmark_tests',
    
    # Test categories
    'TEST_CATEGORIES',
    
    # Module info
    '__version__',
    '__author__',
    '__description__',
]

if __name__ == "__main__":
    # If module is run directly, list tests
    list_tests()
    print("\nUse: python -m alignment_pipeline.tests <command>")
    print("Commands: list, run_all, run <category>, benchmark, create_data")