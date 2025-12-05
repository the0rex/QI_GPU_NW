# Troubleshooting

## GPU OOM (out of memory)
Reduce chunk size:
  chunk.size = 20000

## Pipeline slow at sketching
Increase minimizer window:
  sketch.w = 20

## Candidate chain missing segments
Increase diagonal bandwidth:
  chain.diag_thresh = 100000

## Inaccurate stitching
Increase overlap:
  chunk.overlap = 5000
