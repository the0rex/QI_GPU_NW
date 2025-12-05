from qi_align.chain.seed_filter import filter_repetitive_seeds

def test_filter():
    seeds = [
        (5, 1, 1),
        (6, 1, 2),
        (7, 1, 3),  # repetitive at pos1
        (8, 10, 10)
    ]
    filtered = filter_repetitive_seeds(seeds, max_freq=2)
    # all seeds with pos1=1 should be removed
    assert (8,10,10) in filtered
