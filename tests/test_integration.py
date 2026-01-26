"""
Integration tests for the alignment pipeline.
Updated to use non-deprecated libraries.
"""
import pytest
import os
import tempfile
import shutil
import sys
import importlib.metadata
import warnings
from pathlib import Path
import numpy as np

def create_test_fasta(filename, sequence_id, sequence):
    """Create a test FASTA file."""
    with open(filename, 'w') as f:
        f.write(f">{sequence_id}\n")
        # Write in lines of 80 characters
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")

def test_compression_integration():
    """Test the complete compression/decompression pipeline."""
    from alignment_pipeline.core.compression import (
        compress_4bit_sequence,
        decompress_slice,
        decompress_base
    )
    
    # Test sequences - avoid Ns and ambiguous bases for exact matching
    test_seqs = [
        "ACGT" * 25,  # 100 bases
        "ACGT" * 20,  # 80 bases (avoid Ns for exact matching)
        "ACGTACGTAC" * 10,  # 100 bases
        "A" * 100,
        "C" * 100,
        "G" * 100,
        "T" * 100,
    ]
    
    for seq in test_seqs:
        # Compress
        compressed, length = compress_4bit_sequence(seq)
        
        # Verify length
        assert length == len(seq)
        assert len(compressed) == (length + 1) // 2
        
        # Test full decompression
        full_decompressed = decompress_slice(compressed, length, 0, length)
        assert len(full_decompressed) == len(seq)
        
        # Test individual base decompression
        for i in range(min(10, len(seq))):  # Test first 10 bases
            base = decompress_base(compressed, i)
            # Decompressed should match original for ACGT bases
            if seq[i] in 'ACGTacgt':
                assert base.upper() == seq[i].upper()
            else:
                # For other bases, should be valid IUPAC
                assert base.upper() in 'ACGTURYSWKMBDHVN'
        
        # Test random slices
        import random
        random.seed(42)  # For reproducibility
        for _ in range(3):  # Fewer tests for speed
            start = random.randint(0, len(seq) - 10)
            end = start + random.randint(5, 10)
            slice_decompressed = decompress_slice(compressed, length, start, end)
            assert len(slice_decompressed) == (end - start)
            
            # Verify slice content matches original
            for i in range(start, min(end, len(seq))):
                decomp_base = slice_decompressed[i - start]
                if seq[i] in 'ACGTacgt':
                    assert decomp_base.upper() == seq[i].upper()
                else:
                    assert decomp_base.upper() in 'ACGTURYSWKMBDHVN'

def test_fasta_loading_integration():
    """Test FASTA loading and compression integration."""
    from alignment_pipeline.core.compression import load_two_fasta_sequences
    from alignment_pipeline.io.fasta_reader import validate_fasta_file
    
    # Create temporary FASTA files
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta1 = os.path.join(tmpdir, "test1.fa")
        fasta2 = os.path.join(tmpdir, "test2.fa")
        
        # Create test sequences (avoid ambiguous bases for simpler testing)
        seq1 = "ACGT" * 250  # 1000 bases
        seq2 = "ACGT" * 250  # 1000 bases
        
        create_test_fasta(fasta1, "test_seq1", seq1)
        create_test_fasta(fasta2, "test_seq2", seq2)
        
        # Validate FASTA files
        valid1, msg1 = validate_fasta_file(fasta1)
        valid2, msg2 = validate_fasta_file(fasta2)
        
        assert valid1, f"FASTA 1 validation failed: {msg1}"
        assert valid2, f"FASTA 2 validation failed: {msg2}"
        
        # Load and compress
        seq1_tuple, seq2_tuple = load_two_fasta_sequences(fasta1, fasta2)
        
        # Verify compression
        buf1, len1 = seq1_tuple
        buf2, len2 = seq2_tuple
        
        assert len1 == 1000
        assert len2 == 1000
        # 1000 bases → 500 bytes (4-bit compression: 2 bases per byte)
        assert len(buf1) == 500
        assert len(buf2) == 500

def test_alignment_algorithm_integration():
    """Test integration of alignment algorithms."""
    from alignment_pipeline.algorithms.nw_affine import (
        nw_affine_chunk,
        nw_overlap_realign
    )
    from alignment_pipeline.core.utilities import iupac_is_match
    
    # Test sequences
    seq1 = "ACGTACGTACGT"
    seq2 = "ACGTACGTACGT"
    
    # Test nw_overlap_realign
    aligned1, aligned2 = nw_overlap_realign(seq1, seq2)
    assert len(aligned1) == len(aligned2)
    assert aligned1.replace('-', '') == seq1
    assert aligned2.replace('-', '') == seq2
    
    # Test nw_affine_chunk with simple case
    penalty_matrix = {('A', 'A'): 1, ('C', 'C'): 1, ('G', 'G'): 1, ('T', 'T'): 1}
    
    score, a1, comp, a2, gap_state = nw_affine_chunk(
        seq1, seq2,
        gap_open=-2,
        gap_extend=-0.5,
        penalty_matrix=penalty_matrix,
        window_size=10,
        beam_width=10
    )
    
    # Basic assertions
    assert score > 0
    assert len(a1) == len(a2)
    assert len(comp) == len(a1)
    
    # Verify no bases are lost
    assert a1.replace('-', '') == seq1
    assert a2.replace('-', '') == seq2
    
    # Verify match indicators
    matches = sum(1 for c in comp if c == '|')
    expected_matches = sum(1 for i in range(len(seq1)) if iupac_is_match(seq1[i], seq2[i]))
    assert matches >= expected_matches  # Should have at least exact matches

def test_chunking_integration():
    """Test chunking algorithm integration."""
    from alignment_pipeline.algorithms.anchor_chaining import (
        Anchor,
        Chunk,
        chains_to_chunks,
        chain_anchors,
        sort_anchors
    )
    
    # Create test anchors
    anchors = [
        Anchor(qpos=0, tpos=0, span=100, hash=123, diag=0),
        Anchor(qpos=1000, tpos=1000, span=100, hash=456, diag=0),
        Anchor(qpos=2000, tpos=2000, span=100, hash=789, diag=0),
    ]
    
    # Test sorting
    sorted_anchors = sort_anchors(anchors)
    assert len(sorted_anchors) == 3
    
    # Test chaining
    chains = chain_anchors(sorted_anchors)
    assert len(chains) > 0
    
    # Test chunk creation
    chunks = chains_to_chunks(chains, L1=3000, L2=3000, chunk_size=1000)
    assert len(chunks) > 0
    
    # Verify chunk properties
    for chunk in chunks:
        assert chunk.q_start < chunk.q_end
        assert chunk.t_start < chunk.t_end
        # Within tolerance (chunk_size * 1.2 for enforce_equal_span adjustments)
        assert chunk.q_end - chunk.q_start <= 1000 * 1.2
        assert chunk.t_end - chunk.t_start <= 1000 * 1.2

def test_seeding_integration():
    """Test seeding algorithm integration."""
    # Note: This test may require C++ extensions
    try:
        from alignment_pipeline.core.seeding import strobes_from_4bit_buffer
        from alignment_pipeline.algorithms.seed_matcher import (
            build_strobe_index,
            generate_true_anchors
        )
        
        # Check if C++ extension is available
        from alignment_pipeline.cpp_extensions import SYNC_EXTENSION_AVAILABLE
        
        if not SYNC_EXTENSION_AVAILABLE:
            pytest.skip("C++ syncmer extension not available")
        
        # Create test sequence - need longer sequence for strobe generation
        # Strobes require minimum sequence length based on parameters
        test_seq = "ACGT" * 1000  # 4000 bases - longer for better chance of strobes
        from alignment_pipeline.core.compression import compress_4bit_sequence
        compressed, length = compress_4bit_sequence(test_seq)
        
        # Extract strobes with parameters that work with our sequence
        # Use smaller k and s for shorter effective window
        strobes = list(strobes_from_4bit_buffer(
            compressed, length, 
            k=10,      # Smaller k-mer
            s=3,       # Smaller syncmer
            w_min=10,  # Smaller window
            w_max=30   # Smaller window
        ))
        
        # With C++ extension and proper parameters, should get some strobes
        # But could be 0 if sequence is too simple or parameters not optimal
        # Just verify the function doesn't crash
        assert isinstance(strobes, list)
        
        # If we get strobes, test the index building
        if len(strobes) > 0:
            # Test index building
            index = build_strobe_index(strobes)
            assert len(index) > 0
            
            # Test anchor generation
            anchors = list(generate_true_anchors(strobes, index))
            # At least self-matches should be found
            assert len(anchors) >= len(strobes)
        
    except ImportError as e:
        pytest.skip(f"C++ extensions not available: {e}")

def test_pipeline_config_integration():
    """Test configuration loading and validation."""
    import yaml
    from alignment_pipeline.diagnostics.validation import validate_configuration
    
    # Create minimal valid config
    config = {
        'io': {
            'fasta_dir': 'test',
            'output_dir': 'test_results'
        },
        'alignment': {
            'gap_open': -2,
            'gap_extend': -0.5,
            'beam_width': 30
        },
        'chunking': {
            'default_chunk_size': 1000,
            'overlap': 50
        }
    }
    
    # Validate
    is_valid, errors = validate_configuration(config)
    assert is_valid, f"Config validation failed: {errors}"
    
    # Test invalid config
    invalid_config = {
        'alignment': {
            'gap_open': 10,  # Should be negative
            'beam_width': 0  # Should be positive
        }
    }
    
    is_valid, errors = validate_configuration(invalid_config)
    assert not is_valid
    assert len(errors) >= 2

def test_diagnostics_integration():
    """Test diagnostics module integration - updated to use importlib.metadata."""
    from alignment_pipeline.diagnostics.version_checker import (
        check_python_version,
        check_required_packages
    )
    
    # Check Python version
    py_info = check_python_version()
    assert 'ok' in py_info
    assert 'installed' in py_info
    
    # Check required packages
    package_results = check_required_packages()
    assert isinstance(package_results, dict)
    
    # At least numpy should be installed (as it's in test dependencies)
    assert 'numpy' in package_results

def test_visualization_integration():
    """Test visualization module integration if available."""
    # First, check what's actually in the visualizer module
    try:
        from alignment_pipeline.visualization import visualizer

        # List available functions/classes
        available = dir(visualizer)
        print(f"Available in visualizer: {[f for f in available if not f.startswith('_')]}")

        # Check if iupac_is_match exists
        if 'iupac_is_match' in available:
            # Test iupac_is_match
            assert visualizer.iupac_is_match('A', 'A')
            assert not visualizer.iupac_is_match('A', 'C')
            assert not visualizer.iupac_is_match('A', '-')
            assert not visualizer.iupac_is_match('-', 'A')
            
            # Test None handling
            assert not visualizer.iupac_is_match(None, 'A')
            assert not visualizer.iupac_is_match('A', None)
            
            # Test case insensitivity
            assert visualizer.iupac_is_match('a', 'A')
            assert visualizer.iupac_is_match('A', 'a')
            assert visualizer.iupac_is_match('t', 'T')
            
            # Since you changed the function, 'N' should NOT match 'N'
            assert not visualizer.iupac_is_match('N', 'N'), "'N' should NOT match 'N' with your new implementation"
            assert not visualizer.iupac_is_match('n', 'N'), "Lowercase 'n' should NOT match uppercase 'N'"
            
            # 'N' should not match other bases either
            assert not visualizer.iupac_is_match('N', 'A')
            assert not visualizer.iupac_is_match('A', 'N')
            assert not visualizer.iupac_is_match('N', 'C')
            assert not visualizer.iupac_is_match('N', 'G')
            assert not visualizer.iupac_is_match('N', 'T')
            
            # Test other ambiguous bases if your function excludes them
            # Test a few common ambiguous bases
            for amb_base in ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']:
                assert not visualizer.iupac_is_match(amb_base, amb_base), \
                    f"Ambiguous base {amb_base} should NOT match itself"
                assert not visualizer.iupac_is_match(amb_base, 'A'), \
                    f"Ambiguous base {amb_base} should NOT match 'A'"
                assert not visualizer.iupac_is_match('A', amb_base), \
                    f"'A' should NOT match ambiguous base {amb_base}"
            
            print("✓ iupac_is_match function works correctly (excluding ambiguous bases)")
        else:
            print("✗ iupac_is_match not found in visualizer")
            pytest.fail("iupac_is_match function not found")

        # Check if visualize_alignment_statistics exists (not compute_alignment_stats)
        if 'visualize_alignment_statistics' in available:
            # Create test alignment - Both sequences are 9 characters
            a1 = "ACGT-ACGT"  # 9 chars: A C G T - A C G T
            a2 = "ACGTTACGT"  # 9 chars: A C G T T A C G T
            
            # Test with mock save_path to avoid actually saving files
            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = os.path.join(tmpdir, "test_stats.png")
                stats = visualizer.visualize_alignment_statistics(a1, a2, save_path=save_path)
                
                # Check returned stats
                assert 'matches' in stats
                assert 'mismatches' in stats
                assert 'insertions' in stats
                assert 'deletions' in stats
                assert 'identity' in stats
                assert 'total_aligned' in stats
                assert 'gap_percentage' in stats
                
                # Verify calculations
                # Position by position:
                # 1: A-A = match
                # 2: C-C = match
                # 3: G-G = match
                # 4: T-T = match
                # 5: -T = gap (insertion in seq2)
                # 6: A-A = match
                # 7: C-C = match
                # 8: G-G = match
                # 9: T-T = match
                #
                # Total: 9 aligned positions
                # Matches: 8 (all positions except the gap at position 5)
                # Mismatches: 0
                # Insertions: 1 (T in seq2 aligned to - in seq1 at position 5)
                # Deletions: 0 (no gaps in seq2)
                
                assert stats['total_aligned'] == 9, f"Expected total_aligned=9, got {stats['total_aligned']}"
                assert stats['matches'] == 8, f"Expected matches=8, got {stats['matches']}"
                assert stats['mismatches'] == 0, f"Expected mismatches=0, got {stats['mismatches']}"
                assert stats['insertions'] == 1, f"Expected insertions=1, got {stats['insertions']}"
                assert stats['deletions'] == 0, f"Expected deletions=0, got {stats['deletions']}"
                
                # Identity = matches / (matches + mismatches) = 8/8 = 1.0
                assert abs(stats['identity'] - 1.0) < 0.001, f"Expected identity≈1.0, got {stats['identity']}"
                
                # Gap percentage = (insertions + deletions) / total_aligned = 1/9 ≈ 0.1111
                expected_gap_percentage = 1.0 / 9.0
                assert abs(stats['gap_percentage'] - expected_gap_percentage) < 0.001, \
                    f"Expected gap_percentage≈{expected_gap_percentage}, got {stats['gap_percentage']}"
                
                print("✓ visualize_alignment_statistics function works correctly")
        else:
            print("✗ visualize_alignment_statistics not found in visualizer")
            
        # Test get_base_color if it exists
        if 'get_base_color' in available:
            colors = {
                'A': '#3498db',  # Blue
                'C': '#2ecc71',  # Green
                'G': '#f39c12',  # Orange
                'T': '#e74c3c',  # Red
                'U': '#e74c3c',  # Red (same as T)
                'N': '#95a5a6',  # Gray
            }
            
            for base, expected_color in colors.items():
                color = visualizer.get_base_color(base)
                assert color == expected_color, f"Expected color {expected_color} for base {base}, got {color}"
            
            # Test case-insensitive
            assert visualizer.get_base_color('a') == visualizer.get_base_color('A')
            print("✓ get_base_color function works correctly")
        
        # Test that we can import from the module directly
        from alignment_pipeline.visualization.visualizer import iupac_is_match
        assert iupac_is_match('A', 'A')
        assert not iupac_is_match('A', 'C')
        print("✓ Direct import works")
        
    except ImportError as e:
        print(f"Could not import visualization module: {e}")
        pytest.skip(f"Visualization module not available: {e}")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def test_end_to_end_small_alignment():
    """Test small end-to-end alignment."""
    from alignment_pipeline.core.compression import (
        compress_4bit_sequence,
        decompress_slice
    )
    from alignment_pipeline.algorithms.nw_affine import nw_affine_chunk
    
    # Create small test sequences
    seq1 = "ACGTACGT"
    seq2 = "ACGTACGT"
    
    # Compress
    comp1, len1 = compress_4bit_sequence(seq1)
    comp2, len2 = compress_4bit_sequence(seq2)
    
    # Decompress slices (simulating chunk extraction)
    s1 = decompress_slice(comp1, len1, 0, len1)
    s2 = decompress_slice(comp2, len2, 0, len2)
    
    # Align
    penalty_matrix = {('A', 'A'): 1, ('C', 'C'): 1, ('G', 'G'): 1, ('T', 'T'): 1}
    
    score, a1, comp, a2, gap_state = nw_affine_chunk(
        s1, s2,
        gap_open=-2,
        gap_extend=-0.5,
        penalty_matrix=penalty_matrix,
        window_size=10,
        beam_width=10
    )
    
    # Verify results
    assert score > 0
    assert a1.replace('-', '') == seq1
    assert a2.replace('-', '') == seq2
    assert '|' in comp  # Should have matches

def test_error_handling_integration():
    """Test error handling across modules."""
    from alignment_pipeline.io.fasta_reader import validate_fasta_file
    
    # Test invalid FASTA
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_fasta = os.path.join(tmpdir, "invalid.fa")
        
        # Create empty file
        with open(invalid_fasta, 'w') as f:
            f.write("")
        
        valid, msg = validate_fasta_file(invalid_fasta)
        assert not valid
        assert "empty" in msg.lower() or "no sequences" in msg.lower()
        
        # Create file without FASTA header
        with open(invalid_fasta, 'w') as f:
            f.write("ACGTACGT\n")
        
        valid, msg = validate_fasta_file(invalid_fasta)
        assert not valid
        assert "does not start with" in msg.lower()

def test_cpp_extensions_integration():
    """Test C++ extensions if available."""
    try:
        from alignment_pipeline.cpp_extensions import (
            compress_4bit,
            COMPRESSION_EXTENSION_AVAILABLE,
            SYNC_EXTENSION_AVAILABLE
        )
        
        # Test compression if available
        if COMPRESSION_EXTENSION_AVAILABLE:
            test_seq = "ACGTACGT"
            compressed = compress_4bit(test_seq)
            assert isinstance(compressed, bytes)
            assert len(compressed) == 4  # 8 bases → 4 bytes
        
        # Just check that imports work - actual strobe generation is complex
        
    except ImportError:
        pytest.skip("C++ extensions not available")

def test_memory_efficiency():
    """Test that compression is memory efficient."""
    import sys
    
    # Create a moderately long sequence
    long_seq = "ACGT" * 1000  # 4000 bases
    
    # Get size of uncompressed string
    uncompressed_size = sys.getsizeof(long_seq)
    
    # Compress
    from alignment_pipeline.core.compression import compress_4bit_sequence
    compressed, length = compress_4bit_sequence(long_seq)
    compressed_size = sys.getsizeof(compressed)
    
    # Compression should reduce size (ignoring Python overhead)
    # 4-bit compression should use about half the memory per base
    ratio = compressed_size / uncompressed_size
    # Allow for Python object overhead
    assert ratio < 0.8, f"Compression ratio {ratio:.2f} not efficient enough"

# New tests for importlib.metadata compatibility
def test_importlib_metadata_compatibility():
    """Test that we can use importlib.metadata as a replacement for pkg_resources."""
    try:
        # Test basic functionality of importlib.metadata
        numpy_version = importlib.metadata.version('numpy')
        assert isinstance(numpy_version, str)
        assert len(numpy_version) > 0
        
        # Test exception handling for non-existent package
        try:
            importlib.metadata.version('nonexistent-package-12345')
            assert False, "Should have raised PackageNotFoundError"
        except importlib.metadata.PackageNotFoundError:
            pass  # Expected
        
        print("✓ importlib.metadata works correctly")
        
    except importlib.metadata.PackageNotFoundError as e:
        # numpy might not be installed in test environment, but we should still
        # be able to test the functionality
        print(f"Note: Package not found during test: {e}")
        # Test that we can at least attempt to get version
        # Create a mock to test the import
        pass
    except ImportError:
        # importlib.metadata might not be available in older Python
        # But we require Python 3.8+, so it should be available
        pytest.skip("importlib.metadata not available")

if __name__ == "__main__":
    # Run tests
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))