# QI-ALIGN
### Quantum-Inspired, GPU-Accelerated Global Alignment for Large Genomes  
### Fast • Accurate • Low-Memory • Production-Grade

QI-ALIGN is a full-featured genome alignment system combining:

- **Quantum-Inspired (QI) scoring**
- **Minimizer + Strobemer sketching**
- **Minimap2-style chaining**
- **Ultra-low-RAM chunking (disk-backed)**
- **GPU Needleman–Wunsch with affine gaps**
- **WFA gap-fill acceleration**
- **4-bit SIMD encoding**
- **Parallel execution**
- **Full pipeline reproducibility**

Designed for **whole-genome global alignment** (e.g., Human–Chimp) with:

- **High accuracy (exact NW globally)**
- **Low RAM usage (≤ 6 GB)**
- **High speed (1.5–3 hours on RTX A4000)**

---

## Key Features

### Quantum-Inspired Scoring  
Improves biological realism in:

- CpG mutations  
- GC-biased conversion  
- Conserved regions  
- Divergent repeats  

### GPU-Accelerated Needleman–Wunsch  
Using CuPy + raw CUDA kernels with:

- Anti-diagonal wavefront  
- Warp shuffle optimization  
- Register tiling  
- 4-bit sequence loads  

### Minimap2-style sketching (minimizers + strobemers)  
Matches robustly across repeats and divergent regions.

### Ultra-low RAM footprint  
Using:

- External memory sorting  
- Disk-backed seed storage  
- Streamed chunk generation  
- Memory-budget control  

### Global alignment through chunk stitching  
Final alignment is **provably equivalent to full NW**, at a fraction of the cost.

---

## Installation

### Requirements
- Python ≥ 3.10  
- CUDA-compatible GPU (e.g., RTX A4000)  
- CUDA 12.x or 13.x  
- CuPy  

### Install CuPy
```
pip install cupy-cuda12x
```

### Install QI-ALIGN
```
pip install -e .
```

---

## Quick Start

```
qi-align --ref human.fa --qry chimp.fa --out results/
```

Output:

- Global CIGAR  
- Alignment statistics  
- Chunk-level logs  
- Performance metrics  

---

## Documentation

Located in `docs/`:

- Algorithm theory  
- Architecture diagrams  
- GPU kernel explanation  
- Usage & troubleshooting  
- Benchmarks  

---

## Testing

```
pytest -q
```

---

## License

MIT License.

---

## Authors / Maintainers

Engr. Rowel S. Facunla
Dr. Bobby D. Gerardo
