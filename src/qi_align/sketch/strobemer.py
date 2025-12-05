# ================================================================
# strobemer.py
# 3-way Randstrobes v2 (very high sensitivity)
# ================================================================
import hashlib
import random

def _h(x: bytes) -> int:
    return int(hashlib.sha1(x).hexdigest(), 16) & ((1 << 61) - 1)


def extract_strobemers(seq: bytes, k: int = 15, w: int = 50):
    """
    Yield 3-way strobemers: (hash, pos)
    stro = k1 | k2 | k3
    """
    n = len(seq)
    if n < k*3:
        return

    for i in range(n - k):
        k1 = seq[i:i+k]

        win_start = i + 1
        win_end   = min(n - k, i + w)
        if win_end <= win_start:
            continue

        # select k2, k3 from next window
        j = random.randint(win_start, win_end)
        j2 = random.randint(win_start, win_end)

        k2 = seq[j:j+k]
        k3 = seq[j2:j2+k]

        stro = k1 + k2 + k3
        yield (_h(stro), i)
