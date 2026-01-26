import os
import sys
from pathlib import Path

# Add the src directory to the Python path
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Package metadata - keep these lightweight
__version__ = "1.0.0"
__author__ = "Rowel Facunla"
__email__ = "rowel.facunla@tip.edu.ph"
__description__ = "High-performance sequence alignment pipeline with chunked processing"
__license__ = "MIT"

# Don't import heavy modules at top level - just set flag
HAS_DEPENDENCIES = True  # Assume true, will be checked when needed

def lazy_import_all():
    """Lazy import all modules when needed."""
    # Import the lazy import function from alignment_pipeline
    try:
        from alignment_pipeline import lazy_import as _lazy_import
        _lazy_import()
        return True
    except ImportError as e:
        print(f"Warning: Could not lazy import modules: {e}")
        return False

def get_version():
    """Get the package version."""
    return __version__

def get_authors():
    """Get package authors."""
    return [__author__]

def get_config_path():
    """Get the path to the default configuration file."""
    config_path = Path(__file__).parent / "alignment_pipeline" / "config" / "pipeline_config.yaml"
    if config_path.exists():
        return str(config_path)
    return None

def get_example_path():
    """Get the path to the example directory."""
    example_path = Path(__file__).parent.parent / "examples"
    if example_path.exists():
        return str(example_path)
    return None

def print_info():
    """Print package information."""
    print("=" * 60)
    print("Alignment Pipeline")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print(f"License: {__license__}")
    print()
    print("Key Features:")
    print("  • 4-bit sequence compression for memory efficiency")
    print("  • Syncmer/strobemer seeding for fast anchor finding")
    print("  • Minimap2-style anchor chaining")
    print("  • Affine gap Needleman-Wunsch with beam search")
    print("  • Parallel chunk processing")
    print("  • Comprehensive visualization and diagnostics")
    print()
    print("Available modules:")
    print("  • alignment_pipeline.core - Core alignment functions")
    print("  • alignment_pipeline.algorithms - Algorithm implementations")
    print("  • alignment_pipeline.io - Input/output utilities")
    print("  • alignment_pipeline.visualization - Visualization tools")
    print("  • alignment_pipeline.diagnostics - Diagnostics and validation")
    print("  • alignment_pipeline.pipeline - Main pipeline implementation")
    print("=" * 60)

def check_installation():
    """Check if the package is properly installed (lightweight version)."""
    errors = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8 or higher is required")
    
    # Check for required directories
    required_dirs = [
        Path(__file__).parent / "alignment_pipeline",
        Path(__file__).parent / "alignment_pipeline" / "core",
        Path(__file__).parent / "alignment_pipeline" / "algorithms",
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"Required directory not found: {dir_path.relative_to(Path(__file__).parent.parent)}")
    
    # Check for required files
    required_files = [
        Path(__file__).parent / "alignment_pipeline" / "core" / "__init__.py",
        Path(__file__).parent / "alignment_pipeline" / "algorithms" / "__init__.py",
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            errors.append(f"Required file not found: {file_path.relative_to(Path(__file__).parent.parent)}")
    
    return errors, warnings

# Export key components for direct import from src
__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    '__license__',
    
    # Functions
    'get_version',
    'get_authors',
    'get_config_path',
    'get_example_path',
    'print_info',
    'check_installation',
    'lazy_import_all',
    
    # Module flags
    'HAS_DEPENDENCIES',
]

# Remove all the heavy import logic from the bottom - just keep metadata