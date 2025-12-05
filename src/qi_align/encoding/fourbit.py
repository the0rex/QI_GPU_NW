# ================================================================
# fourbit.py
# 4-bit packed DNA encoding (A,C,G,T,N → 0..4)
# Includes streaming encoder for large genomes.
# ================================================================
import numpy as np

# ----------------------------------------------------------------
# Mapping A/C/G/T/N → {0..4}
# ----------------------------------------------------------------
_ASCII_TO_4BIT = np.full(256, 4, dtype=np.uint8)  # default N=4

_ASCII_TO_4BIT[ord("A")] = 0
_ASCII_TO_4BIT[ord("C")] = 1
_ASCII_TO_4BIT[ord("G")] = 2
_ASCII_TO_4BIT[ord("T")] = 3
_ASCII_TO_4BIT[ord("a")] = 0
_ASCII_TO_4BIT[ord("c")] = 1
_ASCII_TO_4BIT[ord("g")] = 2
_ASCII_TO_4BIT[ord("t")] = 3
_ASCII_TO_4BIT[ord("N")] = 4
_ASCII_TO_4BIT[ord("n")] = 4


# ================================================================
# 1. Convert ASCII → 4-bit array
# ================================================================
def encode_ascii_to_4bit(seq: bytes) -> np.ndarray:
    """
    Convert ASCII DNA bases (A,C,G,T,N) into 0-4 encoded integers.
    Output:
        uint8 array, one value per base.
    """
    arr = np.frombuffer(seq, dtype=np.uint8)
    return _ASCII_TO_4BIT[arr]


# ================================================================
# 2. Pack 4-bit values (two bases per byte)
# ================================================================
def pack_4bit(values: np.ndarray) -> np.ndarray:
    """
    Pack an array of 4-bit values into an array of uint8:
        byte = high_nibble << 4 | low_nibble
    Padding:
        If odd length, append N (4).
    """
    if values.ndim != 1:
        raise ValueError("4-bit values must be 1D array")

    n = values.size
    if n % 2 == 1:
        values = np.append(values, 4)

    high = values[0::2]
    low  = values[1::2]
    return (high << 4) | low


# ================================================================
# 3. Streaming encoder (for very large FASTA)
# ================================================================
def encode_stream_4bit(iterator, chunk=10_000_000):
    """
    Stream large sequences in chunks.
    `iterator` yields bytes segments (FASTA streaming).

    Yields tuples:
        (packed_bytes, original_length)

    Packed_bytes: uint8 array containing 4-bit encoding
    """
    buf = bytearray()

    for block in iterator:
        buf.extend(block)
        if len(buf) >= chunk:
            arr = encode_ascii_to_4bit(bytes(buf))
            packed = pack_4bit(arr)
            yield packed, len(arr)
            buf.clear()

    if buf:
        arr = encode_ascii_to_4bit(bytes(buf))
        packed = pack_4bit(arr)
        yield packed, len(arr)
