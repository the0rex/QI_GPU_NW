from qi_align.stats.cigar import parse_cigar, count_ops

def test_parse_symbolic():
    c = "MMMIID"
    parsed = parse_cigar(c)
    assert parsed == [("M",3),("I",2),("D",1)]

def test_parse_numeric():
    c = "10M2I5D"
    parsed = parse_cigar(c)
    assert parsed == [("M",10),("I",2),("D",5)]

def test_count_ops():
    ops = count_ops("MMIID")
    assert ops["M"] == 2
    assert ops["I"] == 2
    assert ops["D"] == 1
