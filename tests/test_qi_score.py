from qi_align.align.qi_score import build_qi_matrix

def test_qi_matrix_shape():
    M = build_qi_matrix()
    assert M.shape == (5,5)
