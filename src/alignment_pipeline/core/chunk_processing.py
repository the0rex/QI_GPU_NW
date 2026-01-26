#!/usr/bin/env python3
"""
Chunk processing functions from main.py
"""

import os
import json
import numpy as np
import glob
from functools import lru_cache

def save_chunk_result(outdir, chunk_id, s1_start, s1_end, s2_start, s2_end, score, a1, comp, a2):
    """Save chunk result to file."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"chunk_{chunk_id:06d}.npz")
    np.savez_compressed(
        path,
        meta=json.dumps({
            "chunk_id": int(chunk_id),
            "s1_start": int(s1_start),
            "s1_end": int(s1_end),
            "s2_start": int(s2_start),
            "s2_end": int(s2_end),
            "score": float(score)
        }),
        a1=np.array(a1, dtype=object),
        comp=np.array(comp, dtype=object),
        a2=np.array(a2, dtype=object)
    )
    return path

def load_chunk_result(path):
    """Load chunk result from file."""
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data['meta'].tolist()))
    return meta, str(data['a1'].tolist()), str(data['comp'].tolist()), str(data['a2'].tolist())

def reload_and_merge_chunks(outdir, overlap_raw, cleanup=True, verbose=True):
    """Reload and merge chunks - from main.py."""
    files = sorted(glob.glob(os.path.join(outdir, "chunk_*.npz")))
    if not files:
        raise FileNotFoundError("No chunk files found in " + outdir)

    stitched_a1 = stitched_a2 = stitched_comp = ""
    total_score = 0.0

    for idx, path in enumerate(files):
        meta, a1, comp, a2 = load_chunk_result(path)
        total_score += float(meta.get("score", 0.0))

        if idx == 0:
            stitched_a1, stitched_a2, stitched_comp = a1, a2, comp
        else:
            # Import realign_overlap_and_stitch here
            from .stitching import realign_overlap_and_stitch
            stitched_a1, stitched_a2, stitched_comp = realign_overlap_and_stitch(
                stitched_a1, stitched_a2, a1, a2, overlap_raw
            )

        if verbose:
            print(f"Merged {os.path.basename(path)}")

    if cleanup:
        for p in files:
            try:
                os.remove(p)
            except Exception:
                pass

    return total_score, stitched_a1, stitched_comp, stitched_a2