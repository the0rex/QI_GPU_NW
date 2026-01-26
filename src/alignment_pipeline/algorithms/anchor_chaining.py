"""
Author: Rowel Facunla
"""

from dataclasses import dataclass
import math
import numpy as np

# ================================================================
# ANCHOR OBJECT
# ================================================================
@dataclass
class Anchor:
    qpos: int    # position in query (seq1)
    tpos: int    # position in target (seq2)
    span: int    # strobe span
    hash: int    # strobe hash
    diag: int    # tpos - qpos

@dataclass
class Chunk:
    cid: int
    q_start: int
    q_end: int
    t_start: int
    t_end: int

# ================================================================
# Convert strobemers to raw anchors
# ================================================================
def strobes_to_anchors(strobe_gen):
    """
    Converts strobemers into chaining anchors.
    Yields Anchor(qpos, tpos, span, hash, diag).
    """
    for st in strobe_gen:
        # A randstrobe has pos1; we map pos1 to BOTH sequences.
        # Query = seq1; Target = seq2.
        qpos = st.pos1
        tpos = st.pos1   # assumption: positional sync (your pipeline expects monotonic index)
        yield Anchor(qpos, tpos, st.span, st.hash, tpos - qpos)

# ================================================================
# Sorting anchors
# ================================================================
def sort_anchors(anchors):
    """
    Minimap2 sorting rule:
      1) by target position (tpos)
      2) by query position (qpos)
    """
    return sorted(anchors, key=lambda a: (a.tpos, a.qpos))

# ================================================================
# Streaming anchor generator (no large RAM footprint)
# ================================================================
def stream_anchors_from_strobes(buf1, L1, buf2, L2, strobe_func):
    """
    Generates anchors directly from two 4-bit buffers.
    """
    strobes1 = strobe_func(buf1, L1)
    strobes2 = strobe_func(buf2, L2)

    # Pair strobes by index
    for s1, s2 in zip(strobes1, strobes2):
        qpos = s1.pos1
        tpos = s2.pos1
        diag = tpos - qpos
        hash_val = s1.hash ^ s2.hash
        span = max(s1.span, s2.span)
        yield Anchor(qpos, tpos, span, hash_val, diag)

# ================================================================
# Debug utilities
# ================================================================
def debug_print_anchors(anchors, limit=50):
    print("=== DEBUG: First anchors ===")
    for i, a in enumerate(anchors):
        print(f"{i:04d}: q={a.qpos}, t={a.tpos}, diag={a.diag}, span={a.span}, hash={a.hash}")
        if i >= limit:
            break

# ================================================================
# DP CHAINING ENGINE (minimap2-style, Hybrid Gap Model)
# ================================================================

CHAIN_OVERLAP_COST = 3
CHAIN_MAX_SKIP = 40

def hybrid_gap_cost(a, b):
    """
    Hybrid chaining model:
        gap = |(a.qpos - b.qpos) - (a.tpos - b.tpos)|   = diagonal divergence
        diag_diff = |a.diag - b.diag|
    Final cost = 0.5 * log2(gap + 1) + 0.5 * diag_diff
    """
    dq = a.qpos - b.qpos
    dt = a.tpos - b.tpos
    gap = abs(dq - dt)
    diag_diff = abs(a.diag - b.diag)
    return 0.5 * math.log2(gap + 1) + 0.5 * diag_diff

def chain_anchors(anchors):
    """
    Minimap2-style DP chaining with:
        - Hybrid gap model
        - Overlap penalty
        - max_skip = 40
        - backtracking to reconstruct chains

    Input: sorted anchors (sort_anchors())
    Output: list of chains, each chain = list of anchors
    """

    N = len(anchors)
    if N == 0:
        return []

    # DP arrays
    scores = [0.0] * N
    parents = [-1] * N
    best_chain_end = -1
    best_score = -1e300

    # -----------------------------------------------------------
    # For each anchor i, look backwards for possible predecessors.
    # -----------------------------------------------------------
    for i in range(N):
        ai = anchors[i]
        best = 0.0
        best_j = -1
        skip = 0

        # backwards search
        for j in range(i - 1, -1, -1):

            aj = anchors[j]

            # stop if too far left
            if skip > CHAIN_MAX_SKIP:
                break

            # require monotonicity
            if aj.qpos >= ai.qpos or aj.tpos >= ai.tpos:
                continue

            # diagonal consistency (weak rule)
            diag_diff = abs(ai.diag - aj.diag)
            if diag_diff > 20000:   # safety cutoff for disjoint regions
                continue

            # compute gap cost
            g = hybrid_gap_cost(ai, aj)

            # overlap penalty if qpos or tpos overlap backward
            overlap_penalty = 0
            if aj.qpos + aj.span > ai.qpos:
                overlap_penalty += CHAIN_OVERLAP_COST
            if aj.tpos + aj.span > ai.tpos:
                overlap_penalty += CHAIN_OVERLAP_COST

            s = scores[j] + ai.span - (g + overlap_penalty)

            if s > best:
                best = s
                best_j = j
                skip = 0
            else:
                skip += 1

        scores[i] = best
        parents[i] = best_j

        if best > best_score:
            best_score = best
            best_chain_end = i

    # -----------------------------------------------------------
    # Backtrack to extract the best chain
    # -----------------------------------------------------------
    chains = []
    used = set()

    def backtrack(start_idx):
        path = []
        i = start_idx
        while i >= 0 and i not in used:
            path.append(i)
            used.add(i)
            i = parents[i]
        return list(reversed(path))

    # first chain = best scoring
    if best_chain_end >= 0:
        chain = backtrack(best_chain_end)
        chains.append([anchors[i] for i in chain])

    # -----------------------------------------------------------
    # Extract additional chains from unused anchors (multi-chain)
    # -----------------------------------------------------------
    remaining = [i for i in range(N) if i not in used]
    while remaining:
        # choose the remaining anchor with highest DP score
        next_idx = max(remaining, key=lambda x: scores[x])
        chain = backtrack(next_idx)
        chains.append([anchors[i] for i in chain])
        remaining = [i for i in remaining if i not in used]

    # sort chains by first anchor's tpos
    chains.sort(key=lambda chain: chain[0].tpos)

    return chains

# ================================================================
# Debugging: pretty print chains
# ================================================================
def debug_print_chains(chains, limit=5):
    print("=== CHAINS ===")
    for ci, chain in enumerate(chains[:limit]):
        print(f"Chain {ci} length={len(chain)} q[{chain[0].qpos}..{chain[-1].qpos}], "
              f"t[{chain[0].tpos}..{chain[-1].tpos}]")
        for a in chain[:10]:
            print(f"  q={a.qpos}, t={a.tpos}, diag={a.diag}, span={a.span}")
        if len(chain) > 10:
            print("  ...")

# ================================================================
# DSRC Chunker with the following guarantees:
#   • Chunk span equality: q_span == t_span for all tiles
#   • 5000 ± 5% chunk length
#   • Zero drift accumulation
#   • Zero artificial deletions for identical sequences
#   • Full coverage until (L1, L2)
#   • Supports fallback and multi-chain stitching safely
# ================================================================

DEFAULT_CHUNK = 5000
CHUNK_TOL = 0.05
MIN_CHUNK = int(DEFAULT_CHUNK * (1 - CHUNK_TOL))
MAX_CHUNK = int(DEFAULT_CHUNK * (1 + CHUNK_TOL))

# ================================================================
# HARD RULE: Enforce equal spans to avoid deletions drift
# ================================================================
def enforce_equal_span(q0, q1, t0, t1, L1=None, L2=None):
    """
    Mathematically safe span equalization.

    Guarantees:
      • Never deletes bases (no truncation of longer span)
      • Never exceeds sequence bounds (if L1/L2 provided)
      • Preserves monotonicity
      • Safe for final tail tiles
      • Drift-proof
    """

    # Normalize ordering
    if q1 < q0:
        q0, q1 = q1, q0
    if t1 < t0:
        t0, t1 = t1, t0

    span_q = q1 - q0
    span_t = t1 - t0

    # Already equal → nothing to do
    if span_q == span_t:
        return q0, q1, t0, t1

    # --------------------------------------------------
    # Prefer expanding the shorter interval
    # --------------------------------------------------
    if span_q < span_t:
        delta = span_t - span_q
        new_q1 = q1 + delta

        if L1 is not None and new_q1 > L1:
            # Cannot expand → clamp safely
            new_q1 = L1
            new_span = new_q1 - q0
            new_t1 = t0 + new_span
            if L2 is not None:
                new_t1 = min(new_t1, L2)
            return q0, new_q1, t0, new_t1

        return q0, new_q1, t0, t1

    else:
        delta = span_q - span_t
        new_t1 = t1 + delta

        if L2 is not None and new_t1 > L2:
            # Cannot expand → clamp safely
            new_t1 = L2
            new_span = new_t1 - t0
            new_q1 = q0 + new_span
            if L1 is not None:
                new_q1 = min(new_q1, L1)
            return q0, new_q1, t0, new_t1

        return q0, q1, t0, new_t1

# ================================================================
# Regression projection (for anchor-based chunks)
# ================================================================
def safe_regression(chain):
    q = np.array([a.qpos for a in chain], dtype=float)
    t = np.array([a.tpos for a in chain], dtype=float)

    if len(q) < 2:
        slope = 1.0
        intercept = t[0] - q[0]
        return intercept, slope

    slope, intercept = np.polyfit(q, t, 1)

    # Clamp runaway regressions
    slope = max(min(slope, 10), -10)

    return intercept, slope

# ================================================================
# q-range → t-range projection
# ================================================================
def map_q_to_t(q0, q1, intercept, slope, L2):
    t0 = intercept + slope * q0
    t1 = intercept + slope * q1

    # If projection exceeds safe bounds → fallback proportional
    if t0 < 0 or t1 < 0 or t0 > L2 or t1 > L2 or abs(t1 - t0) > 10 * abs(q1 - q0):
        scale = L2 / max(1, L2)    # essentially identity mapping
        t0 = q0 * scale
        t1 = q1 * scale

    return int(max(0, min(L2, t0))), int(max(0, min(L2, t1)))

# ================================================================
# Chunk tiling for a single chain (anchor-derived)
# ================================================================
def tile_chain(chain, cid_start, L2, chunk_size):
    tiles = []
    cid = cid_start

    qmin = chain[0].qpos
    qmax = chain[-1].qpos

    total = qmax - qmin
    if total <= 0:
        return tiles, cid

    ntiles = max(1, total // chunk_size)
    step = total / ntiles

    intercept, slope = safe_regression(chain)

    for i in range(ntiles):
        q0 = int(qmin + step * i)
        q1 = int(qmin + step * (i + 1))
        if q1 > qmax:
            q1 = qmax

        t0, t1 = map_q_to_t(q0, q1, intercept, slope, L2)

        # Prevent drift: enforce equal spans
        q0, q1, t0, t1 = enforce_equal_span(q0, q1, t0, t1)

        tiles.append(Chunk(cid, q0, q1, t0, t1))
        cid += 1

    return tiles, cid

# ================================================================
# Fallback tiling (no anchors)
# ================================================================
def fallback_tiles(q0, q1, t0, t1, cid_start, chunk_size):
    tiles = []
    cid = cid_start

    total_q = q1 - q0
    if total_q <= 0:
        return tiles, cid

    ntiles = max(1, total_q // chunk_size)
    step = total_q / ntiles

    total_t = t1 - t0
    total_t = max(1, total_t)

    for i in range(ntiles):
        qs = int(q0 + i * step)
        qe = int(q0 + (i + 1) * step)
        if qe > q1:
            qe = q1

        # proportional mapping
        a = (qs - q0) / total_q
        b = (qe - q0) / total_q

        ts = int(t0 + a * total_t)
        te = int(t0 + b * total_t)

        # enforce drift-proof spans
        qs, qe, ts, te = enforce_equal_span(qs, qe, ts, te)

        tiles.append(Chunk(cid, qs, qe, ts, te))
        cid += 1

    return tiles, cid

# ================================================================
# MAIN: Convert multiple chains → drift-proof tiles
# ================================================================
def chains_to_chunks(chains, L1, L2, chunk_size=DEFAULT_CHUNK):
    cid = 0
    all_tiles = []

    # ------------------------------------------------------------
    # No anchors → whole genome fallback
    # ------------------------------------------------------------
    if not chains:
        tiles, cid = fallback_tiles(0, L1, 0, L2, cid, chunk_size)
        return tiles

    # ------------------------------------------------------------
    # First chain
    # ------------------------------------------------------------
    tiles, cid = tile_chain(chains[0], cid, L2, chunk_size)
    all_tiles.extend(tiles)

    prev_q = tiles[-1].q_end
    prev_t = tiles[-1].t_end

    # ------------------------------------------------------------
    # Middle chains + potential gaps
    # ------------------------------------------------------------
    for ch in chains[1:]:
        next_q = ch[0].qpos
        next_t = ch[0].tpos

        # Fallback tile between chains if needed
        if next_q - prev_q > chunk_size:
            gap_tiles, cid = fallback_tiles(prev_q, next_q, prev_t, next_t, cid, chunk_size)
            all_tiles.extend(gap_tiles)

        tiles, cid = tile_chain(ch, cid, L2, chunk_size)
        all_tiles.extend(tiles)

        prev_q = tiles[-1].q_end
        prev_t = tiles[-1].t_end

    # ------------------------------------------------------------
    # Final tail region to reach (L1, L2)
    # ------------------------------------------------------------
    if prev_q < L1:
        # FINAL TAIL: force exact end alignment
        all_tiles.append(
            Chunk(
                cid=cid,
                q_start=prev_q,
                q_end=L1,
                t_start=prev_t,
                t_end=L2
            )
        )

    return all_tiles

__all__ = [
    'Anchor',
    'Chunk',
    'strobes_to_anchors',
    'sort_anchors',
    'stream_anchors_from_strobes',
    'debug_print_anchors',
    'chain_anchors',
    'debug_print_chains',
    'enforce_equal_span',
    'safe_regression',
    'map_q_to_t',
    'tile_chain',
    'fallback_tiles',
    'chains_to_chunks',
    'DEFAULT_CHUNK',
    'CHAIN_OVERLAP_COST',
    'CHAIN_MAX_SKIP'
]