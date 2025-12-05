# Global Chaining (Minimap2-Style)

Given seeds (x, y), chaining finds:

A maximal colinear set satisfying:
  x1 < x2 AND y1 < y2 AND |diag1 - diag2| ≤ Δ

Objective:
  maximize chain length or score.

QI-ALIGN uses:
- O(N log N) DP
- Diagonal pruning
- Repeat filtering
- Monotonic reconstruction

Result:
A single global chain representing the backbone of alignment.
