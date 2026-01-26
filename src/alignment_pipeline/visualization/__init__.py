"""
Visualization modules for the alignment pipeline.
"""

import sys

# First, check if we can import the visualization dependencies
try:
    import matplotlib
    import seaborn
    import plotly
    import pandas
    VISUALIZATION_DEPS_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_DEPS_AVAILABLE = False
    missing_deps = str(e)

# Now try to import the visualizer module
if VISUALIZATION_DEPS_AVAILABLE:
    # All dependencies are available, import everything
    try:
        from .visualizer import *
        
        __all__ = [
            'visualize_pipeline_results',
            'create_comprehensive_visualization',
            'visualize_alignment_snippet',
            'visualize_alignment_statistics',
            'visualize_alignment_heatmap',
            'visualize_chunk_alignment',
            'create_interactive_alignment_report',
            'iupac_is_match',
            'VISUALIZATION_DEPS_AVAILABLE',
        ]
        
    except ImportError as e:
        # visualizer.py has other issues
        print(f"WARNING: Could not import visualizer module: {e}")
        VISUALIZATION_DEPS_AVAILABLE = False
else:
    # Dependencies not available, provide minimal functionality
    print(f"NOTE: Visualization dependencies not available: {missing_deps}")
    print("Some visualization features will be limited.")
    
    # Import basic functions from core utilities
    from ..core.utilities import iupac_is_match, compute_alignment_stats
    
    # Create dummy functions that warn when called
    def visualization_not_available(*args, **kwargs):
        print("ERROR: Visualization not available.")
        print(f"Missing dependencies: {missing_deps}")
        print("Install with: pip install matplotlib seaborn plotly pandas")
        return {} if kwargs.get('return_stats', False) else None
    
    # Create minimal implementations
    visualize_pipeline_results = visualization_not_available
    create_comprehensive_visualization = visualization_not_available
    visualize_alignment_snippet = visualization_not_available
    visualize_alignment_statistics = compute_alignment_stats  # This one works without deps
    visualize_alignment_heatmap = visualization_not_available
    visualize_chunk_alignment = visualization_not_available
    create_interactive_alignment_report = visualization_not_available
    
    __all__ = [
        'visualize_pipeline_results',
        'create_comprehensive_visualization',
        'visualize_alignment_snippet',
        'visualize_alignment_statistics',
        'visualize_alignment_heatmap',
        'visualize_chunk_alignment',
        'create_interactive_alignment_report',
        'iupac_is_match',
        'VISUALIZATION_DEPS_AVAILABLE',
    ]