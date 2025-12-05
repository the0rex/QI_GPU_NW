# ================================================================
# external_sort.py
# External merge sort for seeds (low-RAM)
# ================================================================
import heapq
import tempfile
import os

def write_sorted_chunk(seeds, tmpfiles):
    """Writes sorted list of seeds to a temp file."""
    seeds.sort()
    f = tempfile.NamedTemporaryFile(delete=False)
    for h, p1, p2 in seeds:
        f.write(f"{h},{p1},{p2}\n".encode())
    tmpfiles.append(f.name)
    f.close()


def external_seed_sort(seed_iter, chunk_size=5_000_000):
    """
    seed_iter yields (hash, pos1, pos2) tuples.
    Returns generator that yields seeds in sorted order.
    """
    tmpfiles = []
    buf = []

    # create sorted temp files
    for seed in seed_iter:
        buf.append(seed)
        if len(buf) >= chunk_size:
            write_sorted_chunk(buf, tmpfiles)
            buf = []

    if buf:
        write_sorted_chunk(buf, tmpfiles)

    # multi-way merge
    fps = [open(f) for f in tmpfiles]
    heap = []

    # load first line from each file
    for i, fp in enumerate(fps):
        line = fp.readline()
        if line:
            h, p1, p2 = map(int, line.strip().split(","))
            heapq.heappush(heap, (h, p1, p2, i))

    # merge
    while heap:
        h, p1, p2, idx = heapq.heappop(heap)
        yield (h, p1, p2)
        line = fps[idx].readline()
        if line:
            h2, p12, p22 = map(int, line.strip().split(","))
            heapq.heappush(heap, (h2, p12, p22, idx))

    # cleanup
    for fp in fps:
        fp.close()
    for f in tmpfiles:
        os.remove(f)
