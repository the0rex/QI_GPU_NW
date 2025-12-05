from qi_align.stitch.stitcher import stitch_cigars_global

def test_stitch_two_chunks():
    c1 = "MMMMMM"
    c2 = "MMMMMM"
    result = stitch_cigars_global([c1, c2], overlap=2)
    assert "M" in result
    assert len(result) >= 10  # stitched CIGAR should be larger
