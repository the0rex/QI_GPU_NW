from qi_align.chain.chaining import chain_global

def test_basic_chain():
    seeds = [
        (10, 100),
        (20, 200),
        (30, 300),
        (25, 290),
    ]
    chain = chain_global(seeds)
    assert chain == [(10,100), (20,200), (30,300)]
