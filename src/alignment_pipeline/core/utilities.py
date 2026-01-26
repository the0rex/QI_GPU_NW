"""
Utility functions for alignment pipeline.
Author: Rowel Facunla
"""

import os
import gzip
from typing import Dict, Tuple, List

# IUPAC mapping & helpers
IUPAC_MAP = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'}, 'U': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'}, 'W': {'A', 'T'},
    'K': {'G', 'T'}, 'M': {'A', 'C'}, 'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'}, '-': {'-'}
}

def base_set(ch):
    if ch is None:
        return set()
    return IUPAC_MAP.get(ch.upper(), IUPAC_MAP["N"])

def iupac_is_match(a, b):
    """Exact-match only. Ambiguous bases never count as a match."""
    if a is None or b is None:
        return False
    if a == '-' or b == '-':
        return False
    return a.upper() == b.upper()

def iupac_partial_overlap(a, b):
    """True when bases overlap but are not identical."""
    s1, s2 = base_set(a), base_set(b)
    if not s1 or not s2:
        return False
    inter = s1 & s2
    return bool(inter) and s1 != s2

def compute_alignment_stats(a1: str, a2: str) -> Dict:
    """
    Compute comprehensive alignment statistics.
    
    Returns:
        Dictionary with alignment statistics including:
        - matches, mismatches, insertions, deletions
        - identity (percentage)
        - divergence (percentage)
        - total_aligned length
        - gap_openings, total_gaps, gap_percentage
    """
    if len(a1) != len(a2):
        raise ValueError(f"Alignment length mismatch: {len(a1)} != {len(a2)}")
    
    matches = mismatches = insertions = deletions = 0
    gap_openings = 0
    total_gaps = 0
    in_gap = False
    
    for i in range(len(a1)):
        x, y = a1[i], a2[i]
        
        if x != '-' and y != '-':
            if iupac_is_match(x, y):
                matches += 1
            else:
                mismatches += 1
            in_gap = False
        elif x == '-' and y != '-':
            insertions += 1
            total_gaps += 1
            if not in_gap:
                gap_openings += 1
                in_gap = True
        elif x != '-' and y == '-':
            deletions += 1
            total_gaps += 1
            if not in_gap:
                gap_openings += 1
                in_gap = True
        else:
            # Both are gaps - this shouldn't happen in proper alignments
            in_gap = True
    
    total = matches + mismatches + insertions + deletions
    identity = matches / total if total else 0
    divergence = (mismatches + insertions + deletions) / total if total else 0
    gap_percentage = (total_gaps / total * 100) if total else 0
    
    return {
        "matches": matches,
        "mismatches": mismatches,
        "insertions": insertions,
        "deletions": deletions,
        "identity": identity,
        "divergence": divergence,
        "total_aligned": total,
        "gap_openings": gap_openings,
        "total_gaps": total_gaps,
        "gap_percentage": gap_percentage
    }

def build_cigar(a1: str, a2: str) -> str:
    """
    Build CIGAR string from alignment.
    
    CIGAR operations:
    - M: match or mismatch (alignment match)
    - I: insertion in seq1 (gap in seq1)
    - D: deletion in seq1 (gap in seq2)
    - =: sequence match
    - X: sequence mismatch
    """
    cigar = []
    last_op = None
    count = 0
    
    for x, y in zip(a1, a2):
        if x != '-' and y != '-':
            # Both have bases - could be match or mismatch
            if x == y:
                op = '='  # Exact match
            else:
                op = 'X'  # Mismatch
        elif x == '-' and y != '-':
            op = 'I'  # Insertion in seq1
        elif x != '-' and y == '-':
            op = 'D'  # Deletion in seq1 (or insertion in seq2)
        else:
            # Both are gaps - shouldn't happen, treat as match
            op = 'M'
        
        if op == last_op:
            count += 1
        else:
            if last_op is not None:
                cigar.append(f"{count}{last_op}")
            last_op = op
            count = 1
    
    if last_op:
        cigar.append(f"{count}{last_op}")
    
    return ''.join(cigar)

def cigar_to_alignment(cigar: str, seq1: str, seq2: str) -> Tuple[str, str]:
    """
    Convert CIGAR string back to aligned sequences.
    
    Args:
        cigar: CIGAR string
        seq1: Original sequence 1 (without gaps)
        seq2: Original sequence 2 (without gaps)
    
    Returns:
        Tuple of (aligned_seq1, aligned_seq2)
    """
    import re
    
    # Parse CIGAR
    ops = re.findall(r'(\d+)([MIDNX=])', cigar)
    
    a1_chars = []
    a2_chars = []
    i = j = 0  # Positions in seq1 and seq2
    
    for count_str, op in ops:
        count = int(count_str)
        
        if op in 'M=X':  # Match or mismatch
            # Take 'count' bases from both sequences
            a1_chars.append(seq1[i:i+count])
            a2_chars.append(seq2[j:j+count])
            i += count
            j += count
        elif op == 'I':  # Insertion in seq1
            # seq1 has gaps, seq2 has bases
            a1_chars.append('-' * count)
            a2_chars.append(seq2[j:j+count])
            j += count
        elif op == 'D':  # Deletion in seq1 (insertion in seq2)
            # seq1 has bases, seq2 has gaps
            a1_chars.append(seq1[i:i+count])
            a2_chars.append('-' * count)
            i += count
        elif op == 'N':  # Skipped region
            # Skip 'count' bases in seq2
            j += count
    
    return ''.join(a1_chars), ''.join(a2_chars)

def realign_overlap_and_stitch(prev_a1, prev_a2, curr_a1, curr_a2, overlap_raw):
    """
    Realign overlap between consecutive chunks.
    
    Args:
        prev_a1: Previous chunk alignment for sequence 1
        prev_a2: Previous chunk alignment for sequence 2
        curr_a1: Current chunk alignment for sequence 1
        curr_a2: Current chunk alignment for sequence 2
        overlap_raw: Number of raw (ungapped) bases to overlap
    
    Returns:
        Tuple of (stitched_a1, stitched_a2, comparison_string)
    """
    if overlap_raw <= 0:
        # No overlap, just concatenate
        a1 = prev_a1 + curr_a1
        a2 = prev_a2 + curr_a2
        comp = ''.join('|' if iupac_is_match(x, y) else ' '
                       for x, y in zip(a1, a2))
        return a1, a2, comp
    
    def extract_tail(aln, n):
        """Extract columns from the end until we have n non-gap bases."""
        cols = []
        count = 0
        for i in range(len(aln) - 1, -1, -1):
            cols.append(i)
            if aln[i] != '-':
                count += 1
            if count == n:
                break
        cols.reverse()
        return cols
    
    def extract_head(aln, n):
        """Extract columns from the start until we have n non-gap bases."""
        cols = []
        count = 0
        for i in range(len(aln)):
            cols.append(i)
            if aln[i] != '-':
                count += 1
            if count == n:
                break
        return cols
    
    # Extract overlapping regions
    prev_cols = extract_tail(prev_a1, overlap_raw)
    curr_cols = extract_head(curr_a1, overlap_raw)
    
    # Get ungapped sequences from overlapping regions
    prev_seq1 = ''.join(prev_a1[i] for i in prev_cols if prev_a1[i] != '-')
    prev_seq2 = ''.join(prev_a2[i] for i in prev_cols if prev_a2[i] != '-')
    curr_seq1 = ''.join(curr_a1[i] for i in curr_cols if curr_a1[i] != '-')
    curr_seq2 = ''.join(curr_a2[i] for i in curr_cols if curr_a2[i] != '-')
    
    # Check if sequences already match perfectly
    if prev_seq1 == curr_seq1 and prev_seq2 == curr_seq2:
        # Perfect match, just stitch
        a1 = prev_a1[:prev_cols[0]] + curr_a1
        a2 = prev_a2[:prev_cols[0]] + curr_a2
        comp = ''.join('|' if iupac_is_match(x, y) else ' '
                       for x, y in zip(a1, a2))
        return a1, a2, comp
    
    # Need to realign the overlap
    try:
        from ..algorithms.nw_affine import nw_overlap_realign
        ov_a1, ov_a2 = nw_overlap_realign(
            prev_seq1,
            curr_seq1,
            match=1,
            mismatch=-1,
            gap=-1
        )
    except ImportError:
        # Fallback to simple alignment if nw_overlap_realign is not available
        # This is a simplified version - just concatenate with warning
        import warnings
        warnings.warn("nw_overlap_realign not available, using simple stitching")
        a1 = prev_a1[:prev_cols[0]] + curr_a1
        a2 = prev_a2[:prev_cols[0]] + curr_a2
        comp = ''.join('|' if iupac_is_match(x, y) else ' '
                       for x, y in zip(a1, a2))
        return a1, a2, comp
    
    # Cut points for stitching
    left_cut = prev_cols[0]
    right_cut = curr_cols[-1] + 1
    
    # Stitch with realigned overlap
    a1 = prev_a1[:left_cut] + ov_a1 + curr_a1[right_cut:]
    a2 = prev_a2[:left_cut] + ov_a2 + curr_a2[right_cut:]
    
    comp = ''.join('|' if iupac_is_match(x, y) else ' '
                   for x, y in zip(a1, a2))
    
    return a1, a2, comp

def write_cigar_gz(path: str, cigar: str):
    """Write CIGAR string to gzipped file."""
    with gzip.open(path, 'wt') as f:
        f.write(cigar + "\n")

def write_paf(path: str, a1: str, a2: str, cigar: str, stats: Dict, name1: str, name2: str):
    """
    Write PAF (Pairwise mApping Format) alignment.
    
    PAF format columns:
    1. Query sequence name
    2. Query sequence length
    3. Query start (0-based)
    4. Query end (0-based)
    5. Strand (+ or -)
    6. Target sequence name
    7. Target sequence length
    8. Target start (0-based)
    9. Target end (0-based)
    10. Number of matching bases
    11. Alignment block length
    12. Mapping quality (255 for unknown)
    13. Optional fields
    """
    # Remove gaps to get original sequence lengths
    seq1_len = len(a1.replace('-', ''))
    seq2_len = len(a2.replace('-', ''))
    
    # PAF uses matching bases count and total alignment length
    matches = stats["matches"]
    aln_len = stats["total_aligned"]
    
    with open(path, "w") as f:
        # Write PAF line
        f.write(
            f"{name1}\t{seq1_len}\t0\t{seq1_len}\t+\t"
            f"{name2}\t{seq2_len}\t0\t{seq2_len}\t"
            f"{matches}\t{aln_len}\t255\tcg:Z:{cigar}\n"
        )
        
        # Add optional tags for more information
        f.write(f"# Identity: {stats.get('identity', 0)*100:.2f}%\n")
        f.write(f"# Gap percentage: {stats.get('gap_percentage', 0):.2f}%\n")
        f.write(f"# Total gaps: {stats.get('total_gaps', 0)}\n")

def write_maf(path: str, a1: str, a2: str, name1: str, name2: str):
    """
    Write MAF (Multiple Alignment Format) alignment.
    
    MAF format:
    ##maf version=1
    a score=0
    s sequence_name start length strand total_length aligned_sequence
    """
    # Remove gaps to get original sequence lengths
    seq1_len = len(a1.replace('-', ''))
    seq2_len = len(a2.replace('-', ''))
    
    with open(path, "w") as f:
        f.write("##maf version=1\n")
        f.write("# Generated by Alignment Pipeline\n\n")
        
        # Write alignment block
        f.write("a\n")
        f.write(f"s {name1} 0 {seq1_len} + {seq1_len} {a1}\n")
        f.write(f"s {name2} 0 {seq2_len} + {seq2_len} {a2}\n\n")

def write_summary(path: str, stats: Dict, cigar: str, total_score: float):
    """Write comprehensive alignment summary."""
    with open(path, "w") as f:
        f.write("="*60 + "\n")
        f.write("ALIGNMENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Score: {total_score:.2f}\n\n")
        
        f.write("Alignment Statistics:\n")
        f.write("-"*40 + "\n")
        for key, value in stats.items():
            if key in ['identity', 'divergence', 'gap_percentage']:
                f.write(f"{key.replace('_', ' ').title()}: {value*100:.2f}%\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nCIGAR String:\n")
        f.write("-"*40 + "\n")
        f.write(f"{cigar}\n")
        
        f.write("\nCIGAR Operations:\n")
        f.write("-"*40 + "\n")
        import re
        ops = re.findall(r'(\d+)([MIDNX=])', cigar)
        for count, op in ops:
            op_name = {
                'M': 'Match/Mismatch',
                'I': 'Insertion',
                'D': 'Deletion',
                'N': 'Skipped',
                'X': 'Mismatch',
                '=': 'Exact Match'
            }.get(op, op)
            f.write(f"  {count} {op_name}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*60 + "\n")

def get_alignment_quality(a1: str, a2: str) -> Dict[str, float]:
    """
    Calculate alignment quality metrics.
    
    Returns:
        Dictionary with quality metrics:
        - identity_percentage
        - gap_percentage
        - conservation_score
        - complexity_score
    """
    stats = compute_alignment_stats(a1, a2)
    
    # Conservation score: weight matches more heavily than mismatches
    conservation = (stats["matches"] * 2 - stats["mismatches"]) / (stats["total_aligned"] * 2)
    
    # Complexity: measure of sequence complexity in alignment
    # Simple measure: ratio of transitions to transversions (for DNA)
    transitions = 0
    transversions = 0
    
    # Define transitions (A↔G, C↔T) and transversions (all others)
    transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    
    for x, y in zip(a1, a2):
        if x != '-' and y != '-' and x != y:
            if (x.upper(), y.upper()) in transition_pairs:
                transitions += 1
            else:
                transversions += 1
    
    tv_ratio = transitions / (transversions + 1)  # Add 1 to avoid division by zero
    
    return {
        "identity_percentage": stats["identity"] * 100,
        "gap_percentage": stats["gap_percentage"],
        "conservation_score": conservation * 100,
        "transition_transversion_ratio": tv_ratio,
        "matches_per_kb": (stats["matches"] / stats["total_aligned"]) * 1000 if stats["total_aligned"] > 0 else 0
    }

def trim_alignment_ends(a1: str, a2: str, min_conservation: float = 0.1) -> Tuple[str, str]:
    """
    Trim poorly conserved ends from alignment.
    
    Args:
        a1: Aligned sequence 1
        a2: Aligned sequence 2
        min_conservation: Minimum conservation percentage to keep (0.0 to 1.0)
    
    Returns:
        Tuple of (trimmed_a1, trimmed_a2)
    """
    if len(a1) != len(a2):
        raise ValueError("Alignment lengths must match")
    
    # Find start position to trim
    start = 0
    for i in range(len(a1)):
        matches = sum(1 for j in range(i, min(i + 10, len(a1))) 
                     if a1[j] != '-' and a2[j] != '-' and a1[j] == a2[j])
        total = sum(1 for j in range(i, min(i + 10, len(a1))) 
                   if a1[j] != '-' or a2[j] != '-')
        
        if total > 0 and matches / total >= min_conservation:
            start = i
            break
    
    # Find end position to trim
    end = len(a1)
    for i in range(len(a1) - 1, -1, -1):
        matches = sum(1 for j in range(max(i - 9, 0), i + 1) 
                     if a1[j] != '-' and a2[j] != '-' and a1[j] == a2[j])
        total = sum(1 for j in range(max(i - 9, 0), i + 1) 
                   if a1[j] != '-' or a2[j] != '-')
        
        if total > 0 and matches / total >= min_conservation:
            end = i + 1
            break
    
    # Ensure start < end
    if start >= end:
        # If no good region found, return original
        return a1, a2
    
    return a1[start:end], a2[start:end]

def calculate_coverage(a1: str, a2: str, window_size: int = 100) -> List[Dict]:
    """
    Calculate alignment coverage in sliding windows.
    
    Args:
        a1: Aligned sequence 1
        a2: Aligned sequence 2
        window_size: Size of sliding window
    
    Returns:
        List of dictionaries with window statistics
    """
    coverage = []
    
    for i in range(0, len(a1), window_size):
        end = min(i + window_size, len(a1))
        window_a1 = a1[i:end]
        window_a2 = a2[i:end]
        
        stats = compute_alignment_stats(window_a1, window_a2)
        
        coverage.append({
            "start": i,
            "end": end,
            "identity": stats["identity"] * 100,
            "matches": stats["matches"],
            "mismatches": stats["mismatches"],
            "gaps": stats["total_gaps"],
            "gap_percentage": stats["gap_percentage"]
        })
    
    return coverage

def validate_alignment(a1: str, a2: str) -> Tuple[bool, List[str]]:
    """
    Validate alignment for consistency.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check length match
    if len(a1) != len(a2):
        errors.append(f"Alignment length mismatch: {len(a1)} != {len(a2)}")
    
    # Check for invalid characters
    valid_chars = set('ACGTURYSWKMBDHVNacgturyswkmbdhvn-')
    
    for i, (c1, c2) in enumerate(zip(a1, a2)):
        if c1 not in valid_chars:
            errors.append(f"Invalid character '{c1}' at position {i} in sequence 1")
        if c2 not in valid_chars:
            errors.append(f"Invalid character '{c2}' at position {i} in sequence 2")
    
    # Check that sequences aren't all gaps
    if all(c == '-' for c in a1) or all(c == '-' for c in a2):
        errors.append("One or both sequences are all gaps")
    
    # Check for consecutive gaps that might indicate issues
    max_consecutive_gaps = 1000  # Arbitrary threshold
    consecutive_gaps = 0
    
    for c1, c2 in zip(a1, a2):
        if c1 == '-' or c2 == '-':
            consecutive_gaps += 1
            if consecutive_gaps > max_consecutive_gaps:
                errors.append(f"Excessive consecutive gaps (> {max_consecutive_gaps})")
                break
        else:
            consecutive_gaps = 0
    
    return len(errors) == 0, errors

__all__ = [
    'iupac_is_match',
    'iupac_partial_overlap',
    'compute_alignment_stats',
    'build_cigar',
    'cigar_to_alignment',
    'realign_overlap_and_stitch',
    'write_cigar_gz',
    'write_paf',
    'write_maf',
    'write_summary',
    'get_alignment_quality',
    'trim_alignment_ends',
    'calculate_coverage',
    'validate_alignment',
    'IUPAC_MAP',
    'base_set'
]