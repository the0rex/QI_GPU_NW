# ================================================================
# restore.py
# Decode 4-bit encoded DNA back to ASCII sequence
# ================================================================
import numpy as np

_4BIT_TO_ASCII = np.array(list(b"ACGTN"), dtype=np.uint8)

def unpack_4bit(packed: np.ndarray) -> np.ndarray:
    """
    Unpack uint8 array into 4-bit values (high & low nibbles).
    Returns a uint8 array of values 0..4.
    """
    if packed.dtype != np.uint8:
        raise TypeError("Packed array must be uint8")

    high = (packed >> 4) & 0xF
    low  = packed & 0xF
    return np.concatenate([high, low])


def decode_4bit_to_ascii(values: np.ndarray) -> bytes:
    """
    Convert 0..4 encoded bases back to ASCII bytes.
    """
    return _4BIT_TO_ASCII[values].tobytes()
