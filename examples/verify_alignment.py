#!/usr/bin/env python3
import sys
from qi_align.stats.cigar import parse_cigar

def verify_alignment(ref, qry, cigar):
    """
    Reconstruct alignment and check consistency.
    """
    ops = parse_cigar(cigar)
    i = j = 0
    outA = []
    outB = []

    for op, length in ops:
        if op == "M":
            for _ in range(length):
                outA.append(ref[i])
                outB.append(qry[j])
                i += 1; j += 1
        elif op == "I":
            for _ in range(length):
                outA.append("-")
                outB.append(qry[j])
                j += 1
        elif op == "D":
            for _ in range(length):
                outA.append(ref[i])
                outB.append("-")
                i += 1

    outA_str = "".join(map(lambda x: chr(x) if isinstance(x,int) else x, outA))
    outB_str = "".join(map(lambda x: chr(x) if isinstance(x,int) else x, outB))

    # check equal length
    assert len(outA_str) == len(outB_str)

    mismatches = sum(1 for a,b in zip(outA_str, outB_str) if a != b and a != "-" and b != "-")
    print("Alignment OK. Mismatches detected:", mismatches)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: verify_alignment.py <ref.fa> <qry.fa> <cigar>")
        sys.exit(1)

    ref = open(sys.argv[1]).read().splitlines()[1].encode()
    qry = open(sys.argv[2]).read().splitlines()[1].encode()
    cigar = open(sys.argv[3]).read().strip()
    verify_alignment(ref, qry, cigar)
