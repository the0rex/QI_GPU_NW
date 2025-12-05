from qi_align.stitch.overlap_dp import local_overlap_align

def test_overlap_align_basic():
    A = b"ACGTACGT"
    B = b"ACGTACGT"
    cigar = local_overlap_align(A, B)
    assert "M" in cigar
    assert len(cigar) > 0
