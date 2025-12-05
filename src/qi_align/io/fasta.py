# ================================================================
# fasta.py
# Streaming FASTA reader for large genomes
# ================================================================
def fasta_reader(path):
    """
    Yield raw sequence lines (bytes) from FASTA file.
    Ignores header lines.
    """
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b">"):
                continue
            yield line.strip()
