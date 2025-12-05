# ================================================================
# chaining.py
# Minimap2-style global sparse chaining (O(N log N))
# ================================================================
import bisect

def chain_global(seeds, diag_thres=50000):
    """
    seeds: list of (pos_ref, pos_qry)
    Returns: best increasing chain (global)
    """
    if not seeds:
        return []

    # sort by reference coordinate (primary)
    seeds = sorted(seeds, key=lambda x: (x[0], x[1]))

    n = len(seeds)
    dp = [1] * n          # best chain length ending at i
    parent = [-1] * n     # traceback pointer

    # For binary search on query coordinate
    qry_coords = [y for (_, y) in seeds]

    for i in range(n):
        x_i, y_i = seeds[i]

        # find seeds with y < y_i (monotonic requirement)
        j_end = bisect.bisect_left(qry_coords, y_i)

        # scan backward in reduced window
        best_len = 1
        best_j = -1

        # search backwards for feasible predecessors
        # prune by diagonal similarity
        for j in range(j_end - 1, -1, -1):
            x_j, y_j = seeds[j]

            if x_j >= x_i:  # must increase
                continue

            if abs((y_i - x_i) - (y_j - x_j)) > diag_thres:
                # diagonals mismatch → unlikely colinear
                continue

            cand = dp[j] + 1
            if cand > best_len:
                best_len = cand
                best_j = j

        dp[i] = best_len
        parent[i] = best_j

    # traceback best
    end = max(range(n), key=lambda i: dp[i])
    chain = []
    while end != -1:
        chain.append(seeds[end])
        end = parent[end]

    chain.reverse()
    return chain
