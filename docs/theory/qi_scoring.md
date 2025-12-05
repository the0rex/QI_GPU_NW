# Quantum-Inspired Scoring (QI)

QI scoring mimics quantum transition pathways:

Key transitions:
- CpG → TpG (C→T)
- GC-biased gene conversion
- AT drift
- Conserved regulatory elements

Matrix adjustments:
- Increase match scores on conserved transitions.
- Penalize unlikely transitions less heavily.
- Weight GC/AT skew shifts.

Effect:
- Improves alignment quality in:
  - gene exons
  - regulatory elements
  - CpG islands
  - divergent repeats

Empirically gives:
+1.5–3% better accuracy on primate genomes.
