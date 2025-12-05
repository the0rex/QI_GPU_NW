from qi_align.sketch.strobemer import extract_strobemers

def test_strobemers_basic():
    seq = b"ACGTACGTACGTACGT"
    stro = list(extract_strobemers(seq, k=3, w=8))
    assert len(stro) > 0
    for h,pos in stro:
        assert isinstance(h, int)
        assert isinstance(pos, int)
