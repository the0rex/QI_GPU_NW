#!/bin/bash
set -e

echo "Running QI-ALIGN Example..."

qi-align \
  --ref examples/fasta/seqA.fa \
  --qry examples/fasta/seqB.fa \
  --out examples/output \
  --config config/default.yaml

echo "Done. Output in examples/output/"
