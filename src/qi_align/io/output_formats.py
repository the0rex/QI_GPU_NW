def cigar_to_paf(query_name, ref_name, cigar, qlen, rlen):
    from qi_align.stats.cigar import parse_cigar

    ops = parse_cigar(cigar)
    matches = sum(l for op, l in ops if op == "M")
    blocklen = sum(l for _, l in ops)

    strand = "+"
    qstart = 0
    rstart = 0
    qend = qlen
    rend = rlen

    paf = (
        f"{query_name}\t{qlen}\t{qstart}\t{qend}\t{strand}\t"
        f"{ref_name}\t{rlen}\t{rstart}\t{rend}\t"
        f"{matches}\t{blocklen}\t60"
    )
    return paf

def cigar_to_aligned_strings(ref_seq, qry_seq, cigar):
    """
    Reconstruct aligned sequences from CIGAR.
    Returns: (ref_aln_str, qry_aln_str)
    """
    ops = []
    last = None
    count = 0
    for ch in cigar:
        if last is None:
            last = ch
            count = 1
        elif ch == last:
            count += 1
        else:
            ops.append((last, count))
            last = ch
            count = 1
    if last:
        ops.append((last, count))

    r = []
    q = []
    ri = qi = 0

    for op, l in ops:
        if op == "M":
            r.extend(ref_seq[ri:ri+l].decode())
            q.extend(qry_seq[qi:qi+l].decode())
            ri += l
            qi += l
        elif op == "I":  # insertion in query → gap in ref
            r.extend("-" * l)
            q.extend(qry_seq[qi:qi+l].decode())
            qi += l
        elif op == "D":  # deletion in query → gap in query
            r.extend(ref_seq[ri:ri+l].decode())
            q.extend("-" * l)
            ri += l

    return "".join(r), "".join(q)

def cigar_to_maf(ref_name, qry_name, ref_seq, qry_seq, cigar):
    ref_aln, qry_aln = cigar_to_aligned_strings(ref_seq, qry_seq, cigar)

    size = len(ref_aln)
    maf = (
        "a\n"
        f"s {ref_name} 0 {size} + {len(ref_seq)} {ref_aln}\n"
        f"s {qry_name} 0 {size} + {len(qry_seq)} {qry_aln}\n"
    )
    return maf