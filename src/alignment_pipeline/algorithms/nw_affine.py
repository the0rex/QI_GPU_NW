"""
Author: Rowel Facunla
Optimized version with performance improvements
"""

from ..core.utilities import iupac_is_match, iupac_partial_overlap
import math

def nw_affine_chunk(
    seq1, seq2,
    gap_open, gap_extend,
    penalty_matrix,
    window_size,
    beam_width=100,
    tau=3.0,
    energy_match=-5.0,
    energy_mismatch=40.0,
    energy_gap_open=30.0,
    energy_gap_extend=0.5,
    prev_gap_state=None,
    carry_gap_penalty=True
):
    """
    Optimized beam-search NW with affine gaps.
    Key optimizations:
    1. Precomputed energy constants
    2. Local variable caching
    3. Reduced dictionary lookups
    4. Optimized traceback
    """
    try:
        # Pre-uppercase sequences once
        seq1 = seq1.upper()
        seq2 = seq2.upper()
        m, n = len(seq1), len(seq2)
        
        # Precompute tau reciprocal to avoid division in hot loop
        inv_tau = 1.0 / tau
        
        # Precompute energy values as local variables
        ematch = energy_match
        emismatch = energy_mismatch
        egap_open = energy_gap_open
        egap_extend = energy_gap_extend
        
        # Initialize gap state
        initial_gap_penalty = 0.0
        if prev_gap_state is not None and carry_gap_penalty:
            in_gap_seq1, in_gap_seq2, gap_len = prev_gap_state
            if in_gap_seq1 or in_gap_seq2:
                initial_gap_penalty = egap_extend * gap_len

        # Optimized beam representation: use lists instead of dicts for beam storage
        # beam[j] = (log_amp, energy, prev_i, prev_j, move, gap_length)
        # Use array for faster lookups when window is small
        beam_dict = {0: (0.0, initial_gap_penalty, None, None, None, 0)}
        trace = {}
        
        # Preallocate common variables
        best_log = -math.inf
        best_E2 = 0.0
        best_gap_len = 0

        # Cache sequence access
        seq1_chars = seq1
        seq2_chars = seq2

        # =========================
        # DP Loop - Optimized
        # =========================
        for i in range(1, m + 1):
            next_beam = {}
            next_gap_lengths = {}
            
            # Calculate window bounds
            if i >= m - window_size:
                j_start = 1
                j_end = n
            else:
                j_start = max(1, i - window_size)
                j_end = min(n, i + window_size)
            
            # Cache current seq1 character
            s1_char = seq1_chars[i-1]
            
            for j in range(j_start, j_end + 1):
                best = None
                best_log = -math.inf
                
                # ----- Diagonal -----
                if (j - 1) in beam_dict:
                    loga, E, _, _, pm, gap_len = beam_dict[j - 1]
                    # Inline energy calculation for diagonal
                    b2 = seq2_chars[j-1]
                    if iupac_is_match(s1_char, b2):
                        dE = ematch
                    elif iupac_partial_overlap(s1_char, b2):
                        dE = 0.5
                    else:
                        dE = emismatch
                    
                    E2 = E + dE
                    log2 = -E2 * inv_tau  # Multiplication faster than division
                    if log2 > best_log:
                        best = (log2, E2, i-1, j-1, 'D', 0)
                        best_log = log2

                # ----- Up (gap in seq2) -----
                if j in beam_dict:
                    loga, E, _, _, pm, gap_len = beam_dict[j]
                    if pm == 'U':
                        # EXTENDING gap
                        gap_penalty = egap_extend
                        new_gap_len = gap_len + 1
                    else:
                        # OPENING new gap
                        gap_penalty = egap_open
                        new_gap_len = 1
                    
                    E2 = E + gap_penalty
                    log2 = -E2 * inv_tau
                    if log2 > best_log:
                        best = (log2, E2, i-1, j, 'U', new_gap_len)
                        best_log = log2

                # ----- Left (gap in seq1) -----
                if (j - 1) in next_beam:
                    loga, E, _, _, pm, gap_len = next_beam[j - 1]
                    if pm == 'L':
                        # EXTENDING gap
                        gap_penalty = egap_extend
                        new_gap_len = gap_len + 1
                    else:
                        # OPENING new gap
                        gap_penalty = egap_open
                        new_gap_len = 1
                    
                    E2 = E + gap_penalty
                    log2 = -E2 * inv_tau
                    if log2 > best_log:
                        best = (log2, E2, i, j-1, 'L', new_gap_len)
                        best_log = log2

                if best:
                    next_beam[j] = best
                    next_gap_lengths[j] = best[5]  # gap_len is last element
                    trace[(i, j)] = best[4]  # move is 4th element

            # Beam pruning with early exit for small beams
            beam_size = len(next_beam)
            if beam_size > beam_width:
                # Use nsmallest with negative log for max-heap behavior
                import heapq
                top_items = heapq.nlargest(beam_width, 
                                         next_beam.items(), 
                                         key=lambda x: x[1][0])
                next_beam = dict(top_items)
                # Only keep gap lengths for kept items
                kept_keys = {j for j, _ in top_items}
                next_gap_lengths = {j: next_gap_lengths[j] for j in kept_keys}

            beam_dict = next_beam
            
            if not beam_dict:
                beam_dict = {0: (0.0, initial_gap_penalty, None, None, None, 0)}
                window_size = max(window_size * 2, min(m, n))

        # -----------------------------------------
        # Find best reachable end state - Optimized
        # -----------------------------------------
        final_gap_state = (False, False, 0)
        if n in beam_dict:
            _, E, _, _, last_move, last_gap_len = beam_dict[n]
            i, j = m, n
            if last_move == 'U':
                final_gap_state = (False, True, last_gap_len)
            elif last_move == 'L':
                final_gap_state = (True, False, last_gap_len)
        else:
            # Find closest end with single pass
            best_j = -1
            best_score = -math.inf
            best_entry = None
            
            for j, entry in beam_dict.items():
                logA = entry[0]
                if logA > best_score:
                    best_score = logA
                    best_j = j
                    best_entry = entry
            
            if best_j >= 0:
                i, j = m, best_j
                E = best_entry[1]
                last_move = best_entry[4]
                last_gap_len = best_entry[5]
                if last_move == 'U':
                    final_gap_state = (False, True, last_gap_len)
                elif last_move == 'L':
                    final_gap_state = (True, False, last_gap_len)
            else:
                # Fallback - create diagonal path
                E = 0.0
                i, j = m, n
                min_len = min(m, n)
                for k in range(1, min_len + 1):
                    trace[(k, k)] = 'D'

        # =========================
        # TRACEBACK - Optimized
        # =========================
        # Preallocate with estimated size
        est_size = max(m, n) * 2
        a1_list = [''] * est_size
        a2_list = [''] * est_size
        comp_list = [''] * est_size
        idx = est_size - 1
        
        # Use while loops for traceback
        while i > 0 and j > 0:
            move = trace.get((i, j), 'D')
            
            if move == 'D':
                c1 = seq1[i-1]
                c2 = seq2[j-1]
                a1_list[idx] = c1
                a2_list[idx] = c2
                comp_list[idx] = '|' if iupac_is_match(c1, c2) else ' '
                i -= 1
                j -= 1
            elif move == 'U':
                a1_list[idx] = seq1[i-1]
                a2_list[idx] = '-'
                comp_list[idx] = ' '
                i -= 1
            else:  # 'L'
                a1_list[idx] = '-'
                a2_list[idx] = seq2[j-1]
                comp_list[idx] = ' '
                j -= 1
            idx -= 1

        # Handle remaining bases
        remaining_i = i
        remaining_j = j
        
        if remaining_i > 0:
            # Copy remaining seq1 in reverse order
            for k in range(remaining_i - 1, -1, -1):
                a1_list[idx] = seq1[k]
                a2_list[idx] = '-'
                comp_list[idx] = ' '
                idx -= 1
        
        if remaining_j > 0:
            # Copy remaining seq2 in reverse order
            for k in range(remaining_j - 1, -1, -1):
                a1_list[idx] = '-'
                a2_list[idx] = seq2[k]
                comp_list[idx] = ' '
                idx -= 1

        # Slice to get actual content and reverse in one operation
        start_idx = idx + 1
        a1 = ''.join(a1_list[start_idx:])
        a2 = ''.join(a2_list[start_idx:])
        comp = ''.join(comp_list[start_idx:])

        # Verify invariants - only in debug mode
        if __debug__:
            seq1_no_gaps = a1.replace('-', '')
            seq2_no_gaps = a2.replace('-', '')
            if len(seq1_no_gaps) != len(seq1) or len(seq2_no_gaps) != len(seq2):
                print(f"DEBUG: Length mismatch: seq1 {len(seq1_no_gaps)} vs {len(seq1)}, "
                      f"seq2 {len(seq2_no_gaps)} vs {len(seq2)}")

        # Calculate score - optimized loop
        score = 0
        in_gap = False
        gap_type = None
        
        # Cache gap penalties
        gap_open_penalty = gap_open
        gap_extend_penalty = gap_extend
        
        for x, y in zip(a1, a2):
            if x != '-' and y != '-':
                if iupac_is_match(x, y):
                    score += 1
                else:
                    score += gap_open_penalty
                in_gap = False
            elif x == '-' and y != '-':
                # Insertion in seq1
                if not in_gap or gap_type != 'I':
                    score += gap_open_penalty
                    in_gap = True
                    gap_type = 'I'
                else:
                    score += gap_extend_penalty
            elif x != '-' and y == '-':
                # Deletion in seq2
                if not in_gap or gap_type != 'D':
                    score += gap_open_penalty
                    in_gap = True
                    gap_type = 'D'
                else:
                    score += gap_extend_penalty
        
        return score, a1, comp, a2, final_gap_state
        
    except Exception as e:
        print(f"WARNING: Beam search failed, using simple NW: {e}")
        # Fallback - with optimizations
        return _nw_fallback(seq1, seq2, gap_open, gap_extend)

def _nw_fallback(seq1, seq2, gap_open, gap_extend):
    """Optimized fallback NW implementation."""
    m, n = len(seq1), len(seq2)
    
    # Use smaller data types if possible
    import array
    INF_NEG = -10**9
    
    # Initialize matrices
    M = [array.array('f', [INF_NEG] * (n+1)) for _ in range(m+1)]
    I = [array.array('f', [INF_NEG] * (n+1)) for _ in range(m+1)]
    D = [array.array('f', [INF_NEG] * (n+1)) for _ in range(m+1)]
    
    # Simple traceback storage
    trace = [[0] * (n+1) for _ in range(m+1)]
    
    M[0][0] = 0
    for i in range(1, m+1):
        I[i][0] = gap_open + (i-1) * gap_extend
    for j in range(1, n+1):
        D[0][j] = gap_open + (j-1) * gap_extend
    
    # DP with optimized inner loop
    for i in range(1, m+1):
        s1_char = seq1[i-1]
        M_row = M[i]
        I_row = I[i]
        D_row = D[i]
        M_prev = M[i-1]
        I_prev = I[i-1]
        D_prev = D[i-1]
        trace_row = trace[i]
        
        for j in range(1, n+1):
            # Match score
            match_score = 1 if iupac_is_match(s1_char, seq2[j-1]) else gap_open
            
            # M state
            from_M = M_prev[j-1] + match_score
            from_I = I_prev[j-1] + match_score
            from_D = D_prev[j-1] + match_score
            
            if from_M >= from_I and from_M >= from_D:
                M_row[j] = from_M
                trace_state = 0
            elif from_I >= from_D:
                M_row[j] = from_I
                trace_state = 1
            else:
                M_row[j] = from_D
                trace_state = 2
            
            # I state
            open_I = max(M_row[j-1], D_row[j-1]) + gap_open
            extend_I = I_row[j-1] + gap_extend
            if open_I >= extend_I:
                I_row[j] = open_I
                trace_state |= (0 if M_row[j-1] >= D_row[j-1] else 2) << 2
            else:
                I_row[j] = extend_I
                trace_state |= 1 << 2
            
            # D state
            open_D = max(M_prev[j], I_prev[j]) + gap_open
            extend_D = D_prev[j] + gap_extend
            if open_D >= extend_D:
                D_row[j] = open_D
                trace_state |= (0 if M_prev[j] >= I_prev[j] else 1) << 4
            else:
                D_row[j] = extend_D
                trace_state |= 2 << 4
            
            trace_row[j] = trace_state
    
    # Traceback
    final_score = max(M[m][n], I[m][n], D[m][n])
    if final_score == M[m][n]:
        state = 0
    elif final_score == I[m][n]:
        state = 1
    else:
        state = 2
    
    a1, a2 = [], []
    i, j = m, n
    
    while i > 0 or j > 0:
        trace_val = trace[i][j] if i <= m and j <= n else 0
        
        if state == 0:
            prev_state = trace_val & 3
            a1.append(seq1[i-1])
            a2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif state == 1:
            prev_state = (trace_val >> 2) & 3
            a1.append('-')
            a2.append(seq2[j-1])
            j -= 1
        else:
            prev_state = (trace_val >> 4) & 3
            a1.append(seq1[i-1])
            a2.append('-')
            i -= 1
        state = prev_state
    
    a1.reverse()
    a2.reverse()
    
    a1_str = ''.join(a1)
    a2_str = ''.join(a2)
    comp = ''.join('|' if iupac_is_match(x, y) else ' ' for x, y in zip(a1_str, a2_str))
    
    return final_score, a1_str, comp, a2_str, (False, False, 0)

# Keep other functions as-is unless they show up as bottlenecks in profiling
def nw_affine_chunk_with_reanchor(
    s1, s2,
    full_strobes1, full_strobes2,
    q0, t0,
    *nw_args,
    depth=0,
    max_depth=3,
    prev_gap_state=None
):
    """Optimized version with pre-check and early returns."""
    # Early return for trivial cases
    if not s1 or not s2:
        return 0.0, '', '', '', (False, False, 0)
    
    # Extract parameters with defaults
    if len(nw_args) >= 5:
        gap_open, gap_extend, penalty_matrix, window_size, beam_width = nw_args[:5]
    else:
        gap_open, gap_extend = -2, -0.5
        penalty_matrix = {}
        window_size = max(10, min(len(s1), len(s2)) // 10)
        beam_width = 30
    
    # Early return if sequences are very short
    if len(s1) < 20 or len(s2) < 20:
        try:
            result = nw_affine_chunk(
                s1, s2,
                gap_open, gap_extend,
                penalty_matrix,
                window_size,
                beam_width,
                prev_gap_state=prev_gap_state,
                carry_gap_penalty=True
            )
            return result[:5] if len(result) >= 5 else (0.0, '', '', '', (False, False, 0))
        except Exception:
            return 0.0, '', '', '', (False, False, 0)
    
    try:
        result = nw_affine_chunk(
            s1, s2,
            gap_open, gap_extend,
            penalty_matrix,
            window_size,
            beam_width,
            prev_gap_state=prev_gap_state,
            carry_gap_penalty=True
        )
        
        if len(result) == 5:
            score, a1, comp, a2, final_gap_state = result
        elif len(result) >= 3:
            score, a1, comp = result[:3]
            a2 = ""
            final_gap_state = (False, False, 0)
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
            
    except Exception as e:
        # Use optimized fallback
        a1, a2 = _nw_overlap_realign(s1, s2, 1, gap_open, gap_extend)
        comp = ''.join('|' if iupac_is_match(x, y) else ' ' for x, y in zip(a1, a2))
        
        matches = sum(1 for x, y in zip(a1, a2) if iupac_is_match(x, y))
        mismatches = sum(1 for x, y in zip(a1, a2) if not iupac_is_match(x, y) and x != '-' and y != '-')
        gaps = a1.count('-') + a2.count('-')
        score = matches * 1.0 + mismatches * gap_open + gaps * gap_extend
        final_gap_state = (False, False, 0)

    if depth >= max_depth:
        return score, a1, comp, a2, final_gap_state

    if not _detect_gap_explosion_fast(a1, a2):
        return score, a1, comp, a2, final_gap_state

    anchors = local_reanchor_chunk(
        len(s1), len(s2),
        full_strobes1, full_strobes2,
        q0, t0
    )

    if not anchors or len(anchors) < 3:
        return score, a1, comp, a2, final_gap_state

    mid = anchors[len(anchors)//2]
    split_q = mid.q_pos - q0
    split_t = mid.t_pos - t0

    if split_q <= 10 or split_t <= 10 or split_q >= len(s1) - 10 or split_t >= len(s2) - 10:
        return score, a1, comp, a2, final_gap_state

    left_result = nw_affine_chunk_with_reanchor(
        s1[:split_q], s2[:split_t],
        full_strobes1, full_strobes2,
        q0, t0,
        *nw_args,
        depth=depth+1,
        prev_gap_state=prev_gap_state
    )
    
    left_score, left_a1, left_comp, left_a2, left_gap_state = left_result

    right_result = nw_affine_chunk_with_reanchor(
        s1[split_q:], s2[split_t:],
        full_strobes1, full_strobes2,
        q0+split_q, t0+split_t,
        *nw_args,
        depth=depth+1,
        prev_gap_state=left_gap_state
    )
    
    right_score, right_a1, right_comp, right_a2, right_gap_state = right_result

    return (
        left_score + right_score,
        left_a1 + right_a1,
        left_comp + right_comp,
        left_a2 + right_a2,
        right_gap_state
    )

def _nw_overlap_realign(seqA, seqB, match=1, mismatch=-1, gap=-1):
    """Optimized overlap realignment."""
    m, n = len(seqA), len(seqB)
    
    # Use two rows for DP
    prev = [j * gap for j in range(n+1)]
    curr = [0] * (n+1)
    
    # Traceback stored as single byte per cell
    trace = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(1, m+1):
        curr[0] = i * gap
        a_char = seqA[i-1]
        
        for j in range(1, n+1):
            b_char = seqB[j-1]
            
            d = prev[j-1] + (match if iupac_is_match(a_char, b_char) else mismatch)
            u = prev[j] + gap
            l = curr[j-1] + gap
            
            if d >= u and d >= l:
                curr[j] = d
                trace[i][j] = 0  # D
            elif u >= l:
                curr[j] = u
                trace[i][j] = 1  # U
            else:
                curr[j] = l
                trace[i][j] = 2  # L
        
        # Swap rows
        prev, curr = curr, prev
        curr[0] = (i+1) * gap

    # Traceback
    i, j = m, n
    a1, a2 = [], []
    
    while i > 0 or j > 0:
        if i == 0:
            a1.append('-')
            a2.append(seqB[j-1])
            j -= 1
        elif j == 0:
            a1.append(seqA[i-1])
            a2.append('-')
            i -= 1
        else:
            move = trace[i][j]
            if move == 0:  # D
                a1.append(seqA[i-1])
                a2.append(seqB[j-1])
                i -= 1
                j -= 1
            elif move == 1:  # U
                a1.append(seqA[i-1])
                a2.append('-')
                i -= 1
            else:  # L
                a1.append('-')
                a2.append(seqB[j-1])
                j -= 1
    
    a1.reverse()
    a2.reverse()
    
    return ''.join(a1), ''.join(a2)

def _detect_gap_explosion_fast(a1, a2, max_gap_run=30, max_gap_frac=0.15):
    """Fast gap detection using iteration."""
    gap_run = 0
    max_run = 0
    gap_cols = 0
    total = len(a1)
    
    # Handle empty alignment - no gaps by definition
    if total == 0:
        return False
    
    # Use zip and iterate once
    for x, y in zip(a1, a2):
        if x == '-' or y == '-':
            gap_cols += 1
            gap_run += 1
            if gap_run > max_run:
                max_run = gap_run
        else:
            gap_run = 0
    
    return max_run >= max_gap_run or (gap_cols / total) >= max_gap_frac

def local_reanchor_chunk(
    s1_len, s2_len,
    strobes1, strobes2,
    q0, t0,
    max_internal_anchors=50
):
    """Original function - unchanged."""
    from ..algorithms.seed_matcher import build_strobe_index, generate_true_anchors

    if not strobes1 or not strobes2:
        return []

    filtered_strobes1 = []
    for st in strobes1:
        pos = None
        if hasattr(st, 'pos1'):
            pos = st.pos1
        elif hasattr(st, 'pos'):
            pos = st.pos
        elif hasattr(st, 'position'):
            pos = st.position
        elif hasattr(st, 'start'):
            pos = st.start
        elif hasattr(st, 'q_pos'):
            pos = st.q_pos
            
        if pos is not None and q0 <= pos < q0 + s1_len:
            filtered_strobes1.append(st)
    
    filtered_strobes2 = []
    for st in strobes2:
        pos = None
        if hasattr(st, 'pos1'):
            pos = st.pos1
        elif hasattr(st, 'pos'):
            pos = st.pos
        elif hasattr(st, 'position'):
            pos = st.position
        elif hasattr(st, 'start'):
            pos = st.start
        elif hasattr(st, 't_pos'):
            pos = st.t_pos
            
        if pos is not None and t0 <= pos < t0 + s2_len:
            filtered_strobes2.append(st)

    if not filtered_strobes1 or not filtered_strobes2:
        return []

    try:
        idx2 = build_strobe_index(filtered_strobes2)
        anchors = list(generate_true_anchors(filtered_strobes1, idx2))
    except Exception as e:
        print(f"WARNING: Could not generate anchors: {e}")
        return []

    for anchor in anchors:
        anchor.q_pos -= q0
        anchor.t_pos -= t0

    anchors.sort(key=lambda a: (a.q_pos, a.t_pos))
    return anchors[:max_internal_anchors]

detect_gap_explosion = _detect_gap_explosion_fast
nw_overlap_realign = _nw_overlap_realign

__all__ = [
    'nw_affine_chunk',
    'nw_affine_chunk_with_reanchor',
    'nw_overlap_realign',
    'detect_gap_explosion',
    'local_reanchor_chunk'
]