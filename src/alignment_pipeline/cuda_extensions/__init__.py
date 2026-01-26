"""
CUDA extensions for GPU acceleration.
CUDA is initialized in the main package __init__.py
"""

# CUDA should already be initialized by the main package
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except:
    CUDA_AVAILABLE = False

PYCUDA_AVAILABLE = False  # Never use pycuda
USE_CUDA = CUDA_AVAILABLE

# Import extensions
try:
    from .strobe_processor import CUDAStrobeProcessor, batch_generate_anchors_gpu
    from .beam_pruner import CUDABeamPruner
    from .batch_aligner import CUDABatchAligner, compute_batch_scores_gpu
    from .window_predictor import TorchWindowPredictor, predict_window_gpu, load_torch_predictor
    
    CUDA_EXTENSIONS_AVAILABLE = True

except ImportError as e:
    CUDA_EXTENSIONS_AVAILABLE = False
    print(f"[CUDA_EXTENSIONS] Error loading extensions: {e}")
    # Create dummy classes...

__all__ = [
    'CUDAStrobeProcessor',
    'CUDABeamPruner',
    'CUDABatchAligner',
    'TorchWindowPredictor',
    'batch_generate_anchors_gpu',
    'compute_batch_scores_gpu',
    'predict_window_gpu',
    'load_torch_predictor',
    'CUDA_AVAILABLE',
    'PYCUDA_AVAILABLE',
    'USE_CUDA',
    'CUDA_EXTENSIONS_AVAILABLE'
]