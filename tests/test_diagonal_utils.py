from qi_align.chain.diagonal_utils import diagonal, is_monotonic_chain

def test_diag():
    assert diagonal(10,20) == 10

def test_monotonic():
    assert is_monotonic_chain([(1,2),(2,3),(5,6)])
    assert not is_monotonic_chain([(1,2),(2,1)])
