from qi_align.chunk.memory_budget import MemoryBudget

def test_memory_budget():
    mb = MemoryBudget(max_gb=1000)  # never exceeded
    mb.record_chunk()
    assert mb.warned == False
