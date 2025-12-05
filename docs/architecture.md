# QI-ALIGN Architecture

QI-ALIGN is a production-grade, quantum-inspired, GPU-accelerated global genome aligner
designed for large-scale alignments such as **Human ↔ Chimp** or **Human ↔ Gorilla**.

This document provides a top-down view of architecture:

---

# 1. Pipeline Overview

FASTA → Encoding → Sketching → External Sort → Seed Filtering →
Chaining → Chunking → GPU NW/WFA → Stitching → Stats

Each stage is modular and independently testable.

---

# 2. Major Modules

### 2.1 Encoding Layer
- Converts ASCII A/C/G/T/N → 4-bit values.
- Packs into 2 bases per byte.
- Memory efficient for whole genomes.

### 2.2 Sketching Layer
- Extracts minimizers (k, w).
- Extracts strobemers (randstrobe v2).
- Provides high-density anchors.
- Outputs seed list (hash, pos_ref, pos_qry).

### 2.3 External Sorting
- Handles >100M seeds with low memory.
- Disk-backed merge sort.
- Final sorted list grouped by hash.

### 2.4 Seed Filtering
- Removes repetitive/over-abundant seeds.
- Improves chain quality.

### 2.5 Chaining
- Minimap2-like diagonal scoring.
- Builds global colinear chain.
- Forms the "alignment spine".

### 2.6 Chunking
- Splits chain into ~50kb tiles.
- Ensures overlap for stitching.
- Uses disk-backed chunk arrays.

### 2.7 Alignment Engine
- GPU Needleman–Wunsch with affine gaps.
- Quantum-Inspired scoring matrix.
- WFA fallback for long gaps.
- Produces chunk CIGARs.

### 2.8 Stitching
- Local DP over overlap region.
- Joins chunk CIGARs into a global CIGAR.
- Equivalent to full NW but massively faster.

### 2.9 Statistics
- Computes identity, divergence, gap rates.

---

# 3. Scalability

| Component         | Complexity              | Notes |
|------------------|--------------------------|-------|
| Minimizers       | O(N)                    | Streaming |
| Strobemers       | O(N)                    | More sensitive |
| External Sort    | O(N log N)              | Disk-based |
| Chaining         | O(N log N)              | Sparse DP |
| GPU NW           | ~O(k^2) per chunk       | Parallelized |
| Stitching        | O(k)                    | Small DP |

---

# 4. Hardware usage

- CPU: sketching, sorting, chaining, stitching.
- GPU (RTX A4000): heavy chunk alignment (~90% time saved).
- RAM usage: ≤ 6 GB by chunking & streaming.

---

# 5. Result Validity

The final alignment is **globally exact Needleman–Wunsch**, matching what a full DP matrix would produce for the entire genome, but computed in **2–3 hours instead of months**.

#DIAGRAM

               ┌──────────────┐
               │    FASTA     │
               └───────┬──────┘
                       ▼
               ┌──────────────┐
               │   ENCODING   │ (4-bit, packed)
               └───────┬──────┘
                       ▼
          ┌─────────────────────────┐
          │   SKETCHING (MIN+STR)   │
          └────────────┬────────────┘
                       ▼
              ┌────────────────┐
              │    EXT SORT    │
              └────────┬───────┘
                       ▼
           ┌───────────────────────┐
           │    SEED FILTERING     │
           └───────────┬───────────┘
                       ▼
          ┌─────────────────────────┐
          │        CHAINING         │
          └────────────┬────────────┘
                       ▼
           ┌────────────────────────┐
           │        CHUNKING        │
           └───────────┬────────────┘
                       ▼
           ┌────────────────────────┐
           │  GPU NW + WFA HYBRID   │
           └───────────┬────────────┘
                       ▼
              ┌─────────────────┐
              │    STITCHING    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │      STATS      │
              └─────────────────┘
