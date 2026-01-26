"""
Author: Rowel Facunla
"""

from collections import defaultdict
from .anchor_chaining import Anchor

MAX_OCC = 200   # user-selected value

# ================================================================
# Build index for seq2
# ================================================================
def build_strobe_index(strobe_iter):
    """
    Construct minimap2-style occurrence table:
        hash -> [positions]
    """
    index = defaultdict(list)

    for st in strobe_iter:
        index[st.hash].append(st.pos1)

    # Apply occurrence filtering (minimap2's "freq cutoff")
    filtered = {}
    for h, positions in index.items():
        if len(positions) <= MAX_OCC:
            filtered[h] = positions
    return filtered

# ================================================================
# Generate true anchors by hash matching
# ================================================================
def generate_true_anchors(seq1_strobes, seq2_index):
    """
    seq1_strobes: iterator of strobemers from seq1
    seq2_index: hash -> list of positions in seq2

    For every strobe in seq1 that appears in seq2,
    generate anchors:
        Anchor(qpos, tpos, span, hash, diag)
    """

    for st1 in seq1_strobes:
        h = st1.hash
        if h not in seq2_index:
            continue

        t_positions = seq2_index[h]

        for tpos in t_positions:
            qpos = st1.pos1
            diag = tpos - qpos
            yield Anchor(
                qpos=qpos,
                tpos=tpos,
                span=st1.span,
                hash=h,
                diag=diag
            )

__all__ = [
    'build_strobe_index',
    'generate_true_anchors',
    'MAX_OCC'
]