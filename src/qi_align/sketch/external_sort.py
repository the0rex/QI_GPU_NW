# ================================================================
# external_sort.py  (FIXED VERSION)
# Disk-backed seed sorting with safe None handling
# ================================================================
import tempfile
import heapq
from pathlib import Path

def external_seed_sort(seed_iter, chunk_size=500000):
    """
    Sort seeds by hash with minimal RAM.
    Seeds are (h, pos_ref, pos_qry).
    pos_ref or pos_qry may be None.
    """

    temp_files = []

    # -------------------------------
    # 1. Write sorted chunks to disk
    # -------------------------------
    buf = []
    for h, p1, p2 in seed_iter:
        # write None as empty string
        buf.append((h, "" if p1 is None else p1, "" if p2 is None else p2))

        if len(buf) >= chunk_size:
            buf.sort(key=lambda x: x[0])
            tf = tempfile.NamedTemporaryFile(delete=False, mode="w")
            for r in buf:
                tf.write(f"{r[0]},{r[1]},{r[2]}\n")
            tf.close()
            temp_files.append(tf.name)
            buf = []

    # flush last chunk
    if buf:
        buf.sort(key=lambda x: x[0])
        tf = tempfile.NamedTemporaryFile(delete=False, mode="w")
        for r in buf:
            tf.write(f"{r[0]},{r[1]},{r[2]}\n")
        tf.close()
        temp_files.append(tf.name)

    # -------------------------------
    # 2. Merge sorted chunks
    # -------------------------------
    iters = []
    for fname in temp_files:
        f = open(fname, "r")
        iters.append(_seed_iter_file(f))

    for h, p1, p2 in heapq.merge(*iters, key=lambda x: x[0]):
        # Convert "" back to None
        yield (h,
               None if p1 == "" else int(p1),
               None if p2 == "" else int(p2))

    # cleanup
    for fname in temp_files:
        Path(fname).unlink()


def _seed_iter_file(f):
    """Yield parsed rows from one temp file."""
    for line in f:
        h, p1, p2 = line.strip().split(",")

        # Do not cast empty strings to int
        yield (int(h), p1, p2)
