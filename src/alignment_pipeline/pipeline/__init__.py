from .main_pipeline import main

# Alias for convenience
run_pipeline = main

__all__ = [
    'main',
    'run_pipeline',
]

__version__ = "1.0.0"
__author__ = "Rowel Facunla"

def get_pipeline_info():
    """Get information about the pipeline."""
    return {
        'version': __version__,
        'author': __author__,
        'description': 'Main pipeline for sequence alignment',
        'entry_point': 'alignment_pipeline.pipeline.main_pipeline:main',
    }