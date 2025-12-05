# ================================================================
# generate_training_data.py
# Self-supervised training data generator for FNN
# ================================================================
import numpy as np
from qi_align.encoding.fourbit import encode_ascii_to_4bit
from qi_align.fnn.predictor import extract_features
from qi_align.stitch.overlap_dp import local_overlap_align
from qi_align.chain.chaining import chain_global
from qi_align.chunk.chunker import generate_chunks_from_chain
from qi_align.stats.compute_stats import compute_alignment_stats

import json

def generate_labels_for_region(ref4, qry4, x1, x2, y1, y2):
    """
    Generate labels for one region:
        chunk class
        chain break label
        scoring adjustment
    """
    windowA = ref4[x1:x2]
    feats = extract_features(windowA)

    # 1. chunk_class: heuristic from complexity
    entropy = feats[6]
    lc = feats[7]
    gc = feats[5]

    if entropy < 1.1 or lc > 0.6:
        chunk_class = 0  # 20k
    elif entropy < 1.3:
        chunk_class = 1  # 30k
    elif entropy < 1.4:
        chunk_class = 2
    elif entropy < 1.5:
        chunk_class = 3  # default
    else:
        chunk_class = 4  # 60k

    # 2. chain break label: good vs bad boundary
    # use stitching quality on overlap region
    overlap_region = 1000
    ovA = ref4[x2-overlap_region:x2]
    ovB = qry4[y2-overlap_region:y2]

    cig = local_overlap_align(ovA, ovB)

    # high mismatch/low score → bad boundary
    mism = cig.count("I") + cig.count("D")
    break_label = 1.0 if mism > overlap_region * 0.2 else 0.0

    # 3. scoring adjustment
    # target adjustments extracted from local mismatch patterns
    d_match = (gc - 0.5) * 2
    d_mismatch = (entropy - 1.5) * 0.5
    d_gamma = (entropy - 1.0) * 0.8

    return {
        "features": feats.tolist(),
        "chunk_class": chunk_class,
        "break_label": break_label,
        "score_adj": [float(d_match), float(d_mismatch), float(d_gamma)]
    }


def build_training_set(ref_path, qry_path, out_path="fnn_training.json"):
    ref = open(ref_path, "rb").read().splitlines()
    qry = open(qry_path, "rb").read().splitlines()

    ref = b"".join([l for l in ref if not l.startswith(b">")])
    qry = b"".join([l for l in qry if not l.startswith(b">")])

    ref4 = encode_ascii_to_4bit(ref)
    qry4 = encode_ascii_to_4bit(qry)

    # Build seeds → chain
    # (simple: we use a handful of random windows)
    region_size = 50000
    records = []

    for start in range(0, len(ref4)-region_size, 200000):
        x1 = start
        x2 = start + region_size
        y1 = start
        y2 = start + region_size

        rec = generate_labels_for_region(ref4, qry4, x1, x2, y1, y2)
        records.append(rec)

    json.dump(records, open(out_path, "w"))
    print(f"[OK] FNN training dataset written to {out_path}")
    return out_path
