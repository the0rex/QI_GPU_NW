# ================================================================
# chunker.py
# Ultra-low-RAM chunk generation from global seed chain
# ================================================================
from dataclasses import dataclass
from typing import List, Iterator
from .disk_array import DiskArray
from .memory_budget import MemoryBudget

@dataclass
class Chunk:
    ref_start: int
    ref_end: int
    qry_start: int
    qry_end: int

# ----------------------------------------------------------------
# Split chain segments into manageable tiles
# ----------------------------------------------------------------
def _split_segment(x1, y1, x2, y2, chunk_size, overlap):
    """
    Given a segment (x1,y1) → (x2,y2), yields chunk coordinate windows.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = max(dx, dy)

    if length <= 0:
        return

    # number of chunks in this segment
    n = length // chunk_size + 1

    for i in range(n):
        rx1 = x1 + i * chunk_size
        rx2 = min(rx1 + chunk_size + overlap, x2)

        qy1 = y1 + i * chunk_size
        qy2 = min(qy1 + chunk_size + overlap, y2)

        yield Chunk(rx1, rx2, qy1, qy2)


# ----------------------------------------------------------------
# Main chunk generation entrypoint
# ----------------------------------------------------------------
def generate_chunks_from_chain(chain: List[tuple],
                               chunk_size=50_000,
                               overlap=3_000,
                               ram_limit_gb=4):
    """
    Convert global seed chain into disk-backed chunk list.

    Returns an iterator over Chunk objects.
    """

    if len(chain) < 2:
        return []

    # ensure chain is monotonically increasing
    for i in range(1, len(chain)):
        x1, y1 = chain[i-1]
        x2, y2 = chain[i]
        assert x2 > x1 and y2 > y1, "Chain must be strictly monotonic"

    # RAM budgeting
    budget = MemoryBudget(ram_limit_gb)
    disk_chunks = DiskArray()

    # Process chain segment-by-segment
    for (x1,y1), (x2,y2) in zip(chain[:-1], chain[1:]):
        for ch in _split_segment(x1, y1, x2, y2, chunk_size, overlap):
            # append to disk-backed storage
            disk_chunks.append(ch)
            budget.record_chunk()

    # yield chunks lazily
    for ch in disk_chunks:
        yield ch
