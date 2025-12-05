from qi_align.sketch.external_sort import external_seed_sort

def test_external_sort_small():
    seeds = [
        (5,1,2),
        (1,2,3),
        (8,3,4),
        (2,3,1)
    ]
    out = list(external_seed_sort(seeds, chunk_size=2))
    assert out == sorted(seeds)
