# ================================================================
# cigar.py
# Parse CIGAR strings (symbolic or numeric) and count operations.
# ================================================================
import re
from collections import Counter

NUMERIC_CIGAR_RE = re.compile(r"(\d+)([MID])")


def parse_cigar(cigar: str):
    """
    Parse a CIGAR string.
    Automatically detect symbolic vs numeric style.

    Returns list of (op, length).
    """
    # Numeric SAM-style CIGAR
    if any(ch.isdigit() for ch in cigar):
        parts = NUMERIC_CIGAR_RE.findall(cigar)
        return [(op, int(n)) for n, op in parts]

    # Symbolic CIGAR (each character is an op)
    out = []
    last = None
    count = 0

    for ch in cigar:
        if ch not in ("M", "I", "D"):
            raise ValueError(f"Invalid CIGAR op: {ch}")

        if last is None:
            last = ch
            count = 1
        elif ch == last:
            count += 1
        else:
            out.append((last, count))
            last = ch
            count = 1

    if last is not None:
        out.append((last, count))

    return out


def count_ops(cigar: str):
    """
    Return counts of matches, mismatches (not available here; mismatches
    are included in M), insertions, deletions, and total aligned length.
    Useful for global statistics.
    """
    ops = parse_cigar(cigar)
    stats = Counter()

    for op, length in ops:
        stats[op] += length
        stats["total"] += length

    return stats

def compress_cigar(cigar: str) -> str:
    """
    Convert long symbolic CIGAR (e.g., 'MMMMIIDDM') to
    SAM-style compressed CIGAR (e.g., '4M2I2D1M').
    """
    if not cigar:
        return ""

    out = []
    last = cigar[0]
    cnt = 1

    for ch in cigar[1:]:
        if ch == last:
            cnt += 1
        else:
            out.append(f"{cnt}{last}")
            last = ch
            cnt = 1

    out.append(f"{cnt}{last}")
    return "".join(out)
