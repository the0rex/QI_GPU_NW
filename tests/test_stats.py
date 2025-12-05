from qi_align.stats.compute_stats import compute_alignment_stats

def test_compute_stats_basic():
    stats = compute_alignment_stats("MMMIID")
    assert stats["matches"] == 3
    assert stats["gaps"] == 3  # 2 I + 1 D
    assert stats["identity"] == 3/6
