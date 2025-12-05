# Strobemers (Randstrobes v2)

Strobemers connect k1 + k2 + k3 k-mers chosen in a stochastic sliding window.

Advantages:
- Extremely resistant to point substitutions.
- High stability across divergent regions (5–20% divergence).
- Used by modern aligners (e.g., StrobeMap, Winnowmap2 variants).

Parameter choice:
- k=15, w=50 for human–chimp provides sensitive but stable anchors.
