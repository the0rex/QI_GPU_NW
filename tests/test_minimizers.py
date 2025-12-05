from qi_align.sketch.minimizer import extract_minimizers

def test_minimizers_basic():
    seq = b"ACGTACGTACGT"
    mins = list(extract_minimizers(seq, k=3, w=4))
    assert len(mins) > 0
    # minimizer hash must be int
    for h,pos in mins:
        assert isinstance(h, int)
        assert isinstance(pos, int)
