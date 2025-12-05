# ================================================================
# minimizer.py
# minimap2-style minimizer extraction
# ================================================================
import hashlib
from collections import deque

def _hash_kmer(kmer: bytes) -> int:
    return int(hashlib.md5(kmer).hexdigest(), 16) & ((1 << 61) - 1)


def extract_minimizers(seq: bytes, k: int = 19, w: int = 10):
    """
    Yield (hash, position) minimizers from sequence.
    This is standard minimap2-style window minimizer extraction.
    """
    n = len(seq)
    if n < k:
        return

    window = deque()  # list of (hash, pos)
    kmer_hashes = []

    # compute all kmer hashes
    for i in range(n - k + 1):
        kmer = seq[i:i+k]
        h = _hash_kmer(kmer)
        kmer_hashes.append((h, i))

    for i in range(len(kmer_hashes)):
        # push new
        h, pos = kmer_hashes[i]
        window.append((h, pos))

        # pop if window too large
        if len(window) > w:
            window.popleft()

        # once filled, output minimizer
        if len(window) == w:
            yield min(window)  # (hash, position)
