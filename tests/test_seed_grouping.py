from qi_align.chain.seed_grouping import group_seeds_by_diagonal

def test_grouping():
    seeds = [(10,15), (11,17), (200,203)]
    groups = group_seeds_by_diagonal(seeds, diag_band=10)
    assert len(groups) >= 1
