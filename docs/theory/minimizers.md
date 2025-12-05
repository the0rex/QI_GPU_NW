# Minimizers in QI-ALIGN

Minimizers provide:

- Sparse but robust anchoring
- Good repeat-handling (when filtered)
- Deterministic seed positions

Definition:
For a window of size w over k-mers, the minimizer is:

    min(hash(kmer[i : i+k])) for i in window

Properties:
- Subsampling rate ~ 2/(w+1)
- Strong locality properties
- Efficient for primate genomes

In QI-ALIGN:
- Minimizers provide long-range scaffolding.
- Strobemers fill in sensitivity gaps.
