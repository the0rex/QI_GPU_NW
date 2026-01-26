import pytest
from alignment_pipeline.core.compression import (
    compress_4bit_sequence,
    decompress_slice,
    decompress_base
)

def test_compression_roundtrip():
    """Test that compression and decompression are inverses."""
    test_sequences = [
        "ACGT",
        "ACGTACGT",
        "ACGT" * 100,
        "NNNN",
        "RRYYSSWW",
        "ACGTN" * 50,
    ]
    
    for seq in test_sequences:
        compressed, length = compress_4bit_sequence(seq)
        
        assert length == len(seq)
        assert len(compressed) == (length + 1) // 2
        
        # Test decompression of individual bases
        for i in range(length):
            decompressed_base = decompress_base(compressed, i)
            # N and ambiguous bases might not roundtrip exactly
            if seq[i] not in 'Nn':
                assert decompressed_base.upper() == seq[i].upper()
        
        # Test slice decompression
        decompressed = decompress_slice(compressed, length, 0, length)
        assert len(decompressed) == len(seq)
        
        # Check each position (allow for N ambiguity)
        for i, (orig, decomp) in enumerate(zip(seq.upper(), decompressed.upper())):
            if orig != 'N':
                assert decomp == orig or decomp in 'ACGT'

def test_compression_odd_length():
    """Test compression of sequences with odd length."""
    seq = "ACG"  # Length 3
    compressed, length = compress_4bit_sequence(seq)
    
    assert length == 3
    assert len(compressed) == 2  # (3 + 1) // 2 = 2
    
    # Decompress and verify
    decompressed = decompress_slice(compressed, length, 0, length)
    assert len(decompressed) == 3
    assert decompressed.upper() == seq.upper()

def test_decompress_slice():
    """Test slice decompression."""
    seq = "ACGTACGTACGT"
    compressed, length = compress_4bit_sequence(seq)
    
    # Test full slice
    full = decompress_slice(compressed, length, 0, length)
    assert full.upper() == seq.upper()
    
    # Test partial slices
    slice1 = decompress_slice(compressed, length, 2, 6)
    assert slice1.upper() == seq[2:6].upper()
    
    slice2 = decompress_slice(compressed, length, 0, 4)
    assert slice2.upper() == seq[0:4].upper()
    
    slice3 = decompress_slice(compressed, length, length-4, length)
    assert slice3.upper() == seq[-4:].upper()

def test_iupac_compression():
    """Test compression of IUPAC ambiguous bases."""
    # Test all IUPAC codes
    iupac_bases = "ACGTURYSWKMBDHVN"
    
    for base in iupac_bases:
        compressed, length = compress_4bit_sequence(base)
        assert length == 1
        
        decompressed = decompress_base(compressed, 0)
        # Decompressed should be valid IUPAC
        assert decompressed in iupac_bases + iupac_bases.lower()

def test_compression_memory():
    """Test that compression saves memory."""
    import sys
    
    long_seq = "ACGT" * 10000  # 40,000 bases
    seq_size = sys.getsizeof(long_seq)
    
    compressed, length = compress_4bit_sequence(long_seq)
    compressed_size = sys.getsizeof(compressed)
    
    # Compressed should be smaller (4 bits per base vs 8 bits per char)
    # Allow some overhead for Python objects
    assert compressed_size < seq_size * 0.6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])