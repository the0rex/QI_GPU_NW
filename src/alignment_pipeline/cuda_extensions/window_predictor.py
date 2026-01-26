"""
GPU-accelerated window size prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

class TorchWindowPredictor(nn.Module):
    """PyTorch model for window prediction compatible with TensorFlow model."""
    
    def __init__(self, input_size: int = 10, hidden_sizes: list = [64, 32]):
        """Initialize the predictor model."""
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    @classmethod
    def from_tensorflow(cls, tf_model_path: str, x_scaler_path: str,
                       y_scaler_path: str) -> 'TorchWindowPredictor':
        """
        Convert TensorFlow model to PyTorch.
        
        Note: This requires the original TensorFlow model architecture.
        For now, we create a compatible PyTorch model.
        """
        # Create new model with default architecture
        model = cls()
        
        # Note: In production, you would implement actual TF->PyTorch conversion
        # For now, we save the TF model info for later conversion
        model.tf_model_path = tf_model_path
        model.x_scaler_path = x_scaler_path
        model.y_scaler_path = y_scaler_path
        
        return model

def predict_window_gpu(seq1: str, seq2: str, 
                      model: Optional[TorchWindowPredictor] = None,
                      device: str = 'cuda') -> int:
    """
    GPU-accelerated window prediction.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        model: Optional PyTorch model
        device: Device to use
    
    Returns:
        Predicted window size
    """
    # Import feature computation functions
    from ...algorithms.predict_window import compute_features_df
    
    # Compute features
    X_df = compute_features_df(seq1, seq2)
    
    if model is None or not torch.cuda.is_available():
        # Fallback to CPU prediction
        from ...algorithms.predict_window import simple_predict_window
        return simple_predict_window(seq1, seq2)
    
    # Convert to tensor
    features = torch.tensor(X_df.values.astype(np.float32), 
                          device=device, 
                          dtype=torch.float32)
    
    # Predict
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        window_pred = model(features).cpu().numpy()[0, 0]
    
    # Ensure positive integer
    window_size = max(1, int(round(float(window_pred))))
    
    # Apply bounds
    min_len = min(len(seq1), len(seq2))
    window_size = max(5, min(window_size, min_len // 2))
    
    return window_size

def load_torch_predictor(model_path: str = "model_torch.pt",
                        device: str = 'cuda') -> TorchWindowPredictor:
    """
    Load PyTorch predictor model.
    
    Args:
        model_path: Path to PyTorch model
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    model = TorchWindowPredictor()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except FileNotFoundError:
        warnings.warn(f"PyTorch model not found at {model_path}. Using untrained model.")
    
    return model