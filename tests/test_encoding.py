# ================================================================
# Unit tests for the encoding module
# ================================================================
import numpy as np
from qi_align.encoding.fourbit import encode_ascii_to_4bit, pack_4bit
from qi_align.encoding.restore import unpack_4bit, decode_4bit_to_ascii
from qi_align.encoding.simd_helpers import (
    simd_equal, simd_not_equal, simd_mask_count_same
)

def test_encode_decode_roundtrip():
    seq = b"ACGTNACGTNACGTA"
    enc = encode_ascii_to_4bit(seq)
    packed = pack_4bit(enc)
    unpack = unpack_4bit(packed)
    dec = decode_4bit_to_ascii(unpack[:len(seq)])
    assert dec == seq

def test_simd_equal():
    a = np.array([0,1,2,3,4], dtype=np.uint8)
    b = np.array([0,1,9,3,4], dtype=np.uint8)
    eq = simd_equal(a, b)
    assert eq.tolist() == [True, True, False, True, True]
    assert simd_mask_count_same(eq) == 4

def test_simd_not_equal():
    a = np.array([0,1,2], dtype=np.uint8)
    b = np.array([0,9,2], dtype=np.uint8)
    neq = simd_not_equal(a, b)
    assert neq.tolist() == [False, True, False]
