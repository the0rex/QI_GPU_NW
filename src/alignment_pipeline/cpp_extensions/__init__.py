"""
C++ extensions for the alignment pipeline.

This module provides Python bindings for performance-critical C++ code,
including 4-bit compression and syncmer/strobemer generation.

Author: Rowel Facunla
"""

import sys
import warnings
from typing import Optional

# Try to import C++ extensions
try:
    from ._compress_cpp import compress_4bit
    COMPRESSION_EXTENSION_AVAILABLE = True
    _compress_cpp_error = None
except ImportError as e:
    COMPRESSION_EXTENSION_AVAILABLE = False
    _compress_cpp_error = str(e)
    
    # Provide a fallback Python implementation
    def compress_4bit_fallback(seq: str) -> bytes:
        """Fallback Python implementation of 4-bit compression."""
        warnings.warn(f"Using Python fallback for compression: {_compress_cpp_error}")
        
        # IUPAC to 4-bit mapping
        IUPAC_4BIT = {
            'A': 0x1, 'C': 0x2, 'G': 0x4, 'T': 0x8,
            'R': 0x5, 'Y': 0xA, 'S': 0x6, 'W': 0x9,
            'K': 0xC, 'M': 0x3, 'B': 0xE, 'D': 0xD,
            'H': 0xB, 'V': 0x7, 'N': 0xF, '-': 0x0,
            'a': 0x1, 'c': 0x2, 'g': 0x4, 't': 0x8,
            'r': 0x5, 'y': 0xA, 's': 0x6, 'w': 0x9,
            'k': 0xC, 'm': 0x3, 'b': 0xE, 'd': 0xD,
            'h': 0xB, 'v': 0x7, 'n': 0xF,
        }
        
        L = len(seq)
        out = bytearray((L + 1) // 2)
        
        for i in range(0, L, 2):
            hi = IUPAC_4BIT.get(seq[i], 0xF)
            lo = IUPAC_4BIT.get(seq[i + 1], 0xF) if i + 1 < L else 0xF
            out[i >> 1] = (hi << 4) | lo
        
        return bytes(out)
    
    # Use fallback
    compress_4bit = compress_4bit_fallback

try:
    from ._syncmer_cpp import strobes_from_4bit_buffer, Strobemer
    SYNC_EXTENSION_AVAILABLE = True
    _syncmer_cpp_error = None
except ImportError as e:
    SYNC_EXTENSION_AVAILABLE = False
    _syncmer_cpp_error = str(e)
    
    # Provide a fallback Python Strobemer class
    class Strobemer:
        """Fallback Python Strobemer class."""
        def __init__(self, pos1, pos2, hash_val, span, length):
            self.pos1 = pos1
            self.pos2 = pos2
            self.hash = hash_val
            self.span = span
            self.length = length
    
    # Provide a dummy function
    def strobes_from_4bit_buffer_fallback(
        buf: bytes,
        L: int,
        k: int = 21,
        s: int = 5,
        sync_pos: int = 2,
        w_min: int = 20,
        w_max: int = 70
    ):
        """Fallback Python implementation of strobe generation."""
        warnings.warn(f"Using Python fallback for strobe generation: {_syncmer_cpp_error}")
        return []
    
    # Use fallbacks
    strobes_from_4bit_buffer = strobes_from_4bit_buffer_fallback
    Strobemer = Strobemer

# Module metadata
__version__ = "1.0.0"
__author__ = "Rowel Facunla"
__description__ = "C++ extensions for high-performance sequence processing"

# Export what's available
__all__ = [
    'compress_4bit',
    'strobes_from_4bit_buffer',
    'Strobemer',
    'COMPRESSION_EXTENSION_AVAILABLE',
    'SYNC_EXTENSION_AVAILABLE',
]

def check_extensions() -> dict:
    """
    Check which C++ extensions are available.
    
    Returns:
        Dictionary with extension availability information
    """
    return {
        'compression': {
            'available': COMPRESSION_EXTENSION_AVAILABLE,
            'error': _compress_cpp_error if not COMPRESSION_EXTENSION_AVAILABLE else None,
        },
        'syncmer_generation': {
            'available': SYNC_EXTENSION_AVAILABLE,
            'error': _syncmer_cpp_error if not SYNC_EXTENSION_AVAILABLE else None,
        },
        'platform': sys.platform,
        'python_version': sys.version,
    }

def print_extension_info():
    """Print information about C++ extensions."""
    info = check_extensions()
    
    print("=" * 60)
    print("C++ EXTENSIONS STATUS")
    print("=" * 60)
    
    for name, ext_info in info.items():
        if name in ['platform', 'python_version']:
            continue
        
        status = "✅ AVAILABLE" if ext_info['available'] else "❌ NOT AVAILABLE"
        print(f"{name.replace('_', ' ').title():20} {status}")
        
        if not ext_info['available'] and ext_info['error']:
            print(f"  Error: {ext_info['error']}")
    
    print()
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print("=" * 60)
    
    # Provide instructions if extensions are missing
    if not COMPRESSION_EXTENSION_AVAILABLE or not SYNC_EXTENSION_AVAILABLE:
        print("\n⚠  Some C++ extensions are not available.")
        print("   Performance will be degraded.")
        print("\nTo build extensions:")
        print("  1. Make sure you have a C++ compiler installed")
        print("  2. Run: python setup.py build_ext --inplace")
        print("  3. Or install with: pip install -e .")
        print("=" * 60)

# Check extensions on import
if __name__ != "__main__":
    # Only check if not being run directly
    if not COMPRESSION_EXTENSION_AVAILABLE or not SYNC_EXTENSION_AVAILABLE:
        warnings.warn(
            "Some C++ extensions are not available. "
            "Performance will be degraded. Run print_extension_info() for details.",
            ImportWarning
        )

if __name__ == "__main__":
    # Run diagnostics if module is executed directly
    print_extension_info()