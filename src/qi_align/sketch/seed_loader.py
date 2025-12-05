# ================================================================
# seed_loader.py
# Load and merge pre-written seed files
# ================================================================
import struct
import heapq

def load_seeds_merged(paths):
    """
    Merge multiple sorted seed binary files.
    Format: <hash, pos1, pos2> packed as 3x uint64
    """
    fps = [open(p, "rb") for p in paths]
    heap = []

    def read_seed(fp):
        data = fp.read(24)  # 3 * 8 bytes
        if not data:
            return None
        return struct.unpack("<QQQ", data)

    # prime heap
    for idx, fp in enumerate(fps):
        s = read_seed(fp)
        if s:
            h, p1, p2 = s
            heapq.heappush(heap, (h, p1, p2, idx))

    while heap:
        h, p1, p2, idx = heapq.heappop(heap)
        yield (h, p1, p2)
        nxt = read_seed(fps[idx])
        if nxt:
            h2, q1, q2 = nxt
            heapq.heappush(heap, (h2, q1, q2, idx))

    for fp in fps:
        fp.close()
