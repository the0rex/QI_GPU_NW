# ================================================================
# pipeline.py
# Full QI-ALIGN pipeline orchestration
# ================================================================
import argparse
import yaml
import os
from pathlib import Path
from tqdm import tqdm

# Encoding
from qi_align.encoding.fourbit import encode_ascii_to_4bit
from qi_align.io.fasta import fasta_reader

# Sketching
from qi_align.sketch.minimizer import extract_minimizers
from qi_align.sketch.strobemer import extract_strobemers
from qi_align.sketch.external_sort import external_seed_sort
from qi_align.chain.seed_filter import filter_repetitive_seeds

# Chaining
from qi_align.chain.chaining import chain_global

# Chunking
from qi_align.chunk.chunker import generate_chunks_from_chain

# Alignment
from qi_align.align.qi_score import build_qi_matrix
from qi_align.align.hybrid_align import hybrid_global_align

# Stitching
from qi_align.stitch.stitcher import stitch_cigars_global

# Stats
from qi_align.stats.compute_stats import compute_alignment_stats

# Parallel
from qi_align.pipeline.parallel import parallel_map

# Logger
from qi_align.io.logger import log

#outputs
from qi_align.stats.cigar import compress_cigar
from qi_align.io.output_formats import cigar_to_paf, cigar_to_maf

# ----------------------------------------------------------------
# Worker function for one chunk
# ----------------------------------------------------------------
def _align_chunk(args):
    (
        ref_seq, qry_seq,
        chunk, score_matrix,
        gopen, gext
    ) = args

    rs = ref_seq[chunk.ref_start:chunk.ref_end]
    qs = qry_seq[chunk.qry_start:chunk.qry_end]

    cigar = hybrid_global_align(rs, qs, score_matrix, gopen, gext)
    return cigar


# ----------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------
def run_pipeline(ref_path, qry_path, config):

    # ========== (1) Load config ==========
    chunk_size  = config["chunk"]["size"]
    overlap     = config["chunk"]["overlap"]

    k           = config["sketch"]["k"]
    w           = config["sketch"]["w"]
    st_k        = config["sketch"]["strobe_k"]
    st_w        = config["sketch"]["strobe_w"]

    match       = config["score"]["match"]
    mismatch    = config["score"]["mismatch"]
    ambiguous   = config["score"]["ambiguous"]
    gamma       = config["score"]["gamma"]

    gopen       = config["score"].get("gap_open",-5)
    gext        = config["score"].get("gap_extend",-1)

    ram_limit   = config["ram"]["max_gb"]

    workers     = config.get("workers",4)

    # ========== (2) Load FASTA + 4-bit encode ==========

    log("Loading FASTA & 4-bit encoding…")
    ref_full = b"".join(fasta_reader(ref_path))
    qry_full = b"".join(fasta_reader(qry_path))

    ref4 = encode_ascii_to_4bit(ref_full)
    qry4 = encode_ascii_to_4bit(qry_full)

    # ============================================================
    # SPECIAL CASE: sequences too small for sketching/chaining
    # ============================================================
    if len(ref4) < 2000 or len(qry4) < 2000:
        log("Sequences are small — using direct single-chunk alignment.")

        score_matrix = build_qi_matrix(
            match,
            mismatch,
            ambiguous,
            gamma
        )

        cigar = hybrid_global_align(ref4, qry4, score_matrix, gopen, gext)

        stats = compute_alignment_stats(
            cigar,
            ref_length=len(ref4),
            qry_length=len(qry4)
        )

        return cigar, stats


    # ========== (3) Extract minimizers & strobemers ==========

    log("Extracting seeds (minimizers + strobemers)…")
    seed_iter = []

    # Minimizers
    for h,pos in extract_minimizers(ref_full, k, w):
        seed_iter.append((h, pos, None))
    for h,pos in extract_minimizers(qry_full, k, w):
        seed_iter.append((h, None, pos))

    # Strobemers (sensitive)
    for h,pos in extract_strobemers(ref_full, st_k, st_w):
        seed_iter.append((h, pos, None))
    for h,pos in extract_strobemers(qry_full, st_k, st_w):
        seed_iter.append((h, None, pos))

    # ========== (4) External sort ==========
    log("External sorting seeds (low RAM)…")
    sorted_seeds = list(external_seed_sort(seed_iter))

    # ========== (5) Filter repeats ==========
    log("Filtering repetitive seeds…")
    filtered = filter_repetitive_seeds(sorted_seeds)

    # ========== (6) Build positional seeds (must be pairs) ==========
    log("Pairing seeds by hash…")
    # seeds list (pos1,pos2) requires matching hashes
    hash_groups = {}
    for h,p1,p2 in filtered:
        if h not in hash_groups:
            hash_groups[h] = [None,None]
        if p1 is not None:
            hash_groups[h][0] = p1
        if p2 is not None:
            hash_groups[h][1] = p2

    paired = [(v[0],v[1]) for v in hash_groups.values() if v[0] is not None and v[1] is not None]

    # ========== (7) Global chaining ==========
    log("Computing global chain…")
    chain = chain_global(paired)

    # ========== (8) Generate chunks ==========
    log("Generating alignment chunks…")
    chunks = list(generate_chunks_from_chain(chain,
                                            chunk_size=chunk_size,
                                            overlap=overlap,
                                            ram_limit_gb=ram_limit))

    # ========== (9) GPU alignment per chunk ==========
    log("Aligning chunks…")
    score_matrix = build_qi_matrix(match, mismatch, ambiguous, gamma)

    # Prepare batch tasks
    tasks = []
    for ch in chunks:
        tasks.append((ref4, qry4, ch, score_matrix, gopen, gext))

    cigars = list(tqdm(
        parallel_map(_align_chunk, tasks, workers=workers),
        total=len(tasks),
        desc="GPU Align"
    ))

    # ========== (10) Stitch full genome CIGAR ==========
    log("Stitching genome-wide alignment…")
    global_cigar = stitch_cigars_global(cigars, overlap)

    # ========== (11) Compute stats ==========
    log("Computing statistics…")
    stats = compute_alignment_stats(global_cigar,
                                    ref_length=len(ref4),
                                    qry_length=len(qry4))

    return global_cigar, stats


# ----------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="QI-ALIGN: Quantum-Inspired Whole-Genome Aligner")

    parser.add_argument("--ref", required=True, help="Reference FASTA")
    parser.add_argument("--qry", required=True, help="Query FASTA")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # run pipeline
    cigar, stats = run_pipeline(args.ref, args.qry, config)

    # save outputs
    (outdir/"alignment.cigar").write_text(cigar)

    # Load sequences for auxiliary formats
    ref_full = b"".join(fasta_reader(args.ref))
    qry_full = b"".join(fasta_reader(args.qry))

    # Short compressed CIGAR
    short_cigar = compress_cigar(cigar)
    (outdir/"alignment.cigar").write_text(short_cigar)

    # PAF output
    paf = cigar_to_paf(
        query_name=Path(args.qry).stem,
        ref_name=Path(args.ref).stem,
        cigar=cigar,
        qlen=len(qry_full),
        rlen=len(ref_full)
    )
    (outdir/"alignment.paf").write_text(paf)

    # MAF output
    maf = cigar_to_maf(
        ref_name=Path(args.ref).stem,
        qry_name=Path(args.qry).stem,
        ref_seq=ref_full,
        qry_seq=qry_full,
        cigar=cigar,
    )
    (outdir/"alignment.maf").write_text(maf)

    # Stats JSON
    import json
    (outdir/"stats.json").write_text(json.dumps(stats, indent=2))

    log(f"Results written to: {outdir}")
    return 0


if __name__ == "__main__":
    main_cli()
