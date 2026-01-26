"""
4-bit compression and decompression functions.
Author: Rowel Facunla
"""

from functools import lru_cache
from Bio import SeqIO
from typing import Tuple
from ..cpp_extensions._compress_cpp import compress_4bit

# IUPAC mapping with ALL codes
IUPAC_4BIT = {
    # Standard bases
    'A': 0x1,  'C': 0x2,  'G': 0x4,  'T': 0x8,
    'a': 0x1,  'c': 0x2,  'g': 0x4,  't': 0x8,
    
    # Uracil (RNA)
    #'U': 0x8, 'u': 0x8,
    
    # IUPAC ambiguous bases
    'R': 0x5,  'Y': 0xA,  'S': 0x6,  'W': 0x9,
    'K': 0xC,  'M': 0x3,  'B': 0xE,  'D': 0xD,
    'H': 0xB,  'V': 0x7,  'N': 0xF,
    
    # Lowercase ambiguous bases
    'r': 0x5,  'y': 0xA,  's': 0x6,  'w': 0x9,
    'k': 0xC,  'm': 0x3,  'b': 0xE,  'd': 0xD,
    'h': 0xB,  'v': 0x7,  'n': 0xF,
    
    # Gap
    '-': 0x0
}

REV_IUPAC = {v: k for k, v in IUPAC_4BIT.items()}

def compress_4bit_sequence(seq: str) -> Tuple[bytes, int]:
    """
    Compress a DNA sequence into 4-bit representation.
    """
    buf = compress_4bit(seq)
    return buf, len(seq)

def decompress_base(buf: bytes, idx: int) -> str:
    """
    Return a single base at index idx from compressed buffer.
    """
    byte_val = buf[idx // 2]
    if idx % 2 == 0:
        code = (byte_val >> 4) & 0xF
    else:
        code = byte_val & 0xF
    return REV_IUPAC.get(code, 'N')

def decompress_slice(buf: bytes, L: int, start: int, end: int) -> str:
    """
    Extract raw substring start:end from compressed buffer.
    """
    out = []
    for i in range(start, min(end, L)):
        out.append(decompress_base(buf, i))
    return ''.join(out)

@lru_cache(maxsize=2)
def load_two_fasta_sequences_cached(f1: str, f2: str, start: int = 0, end: int = None) -> Tuple[Tuple[bytes, int], Tuple[bytes, int]]:
    """
    Cached version of FASTA loading.
    """
    seq1 = str(next(SeqIO.parse(f1, "fasta")).seq)
    seq2 = str(next(SeqIO.parse(f2, "fasta")).seq)

    if end is not None:
        seq1 = seq1[start:end]
        seq2 = seq2[start:end]

    comp1, L1 = compress_4bit_sequence(seq1)
    comp2, L2 = compress_4bit_sequence(seq2)
    
    # Convert to bytes if needed
    if isinstance(comp1, bytearray):
        comp1 = bytes(comp1)
    if isinstance(comp2, bytearray):
        comp2 = bytes(comp2)
    
    return (comp1, L1), (comp2, L2)

def load_two_fasta_sequences(f1: str, f2: str, start: int = 0, end: int = None) -> Tuple[Tuple[bytes, int], Tuple[bytes, int]]:
    """
    Load two FASTA sequences and compress them.
    """
    return load_two_fasta_sequences_cached(f1, f2, start, end)

__all__ = [
    'compress_4bit_sequence',
    'decompress_base',
    'decompress_slice',
    'load_two_fasta_sequences',
    'IUPAC_4BIT',
    'REV_IUPAC'
]