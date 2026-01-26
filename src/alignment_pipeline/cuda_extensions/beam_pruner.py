"""
GPU-accelerated beam pruning for alignment.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any

class CUDABeamPruner:
    """GPU-accelerated beam pruning for alignment chunks."""
    
    def __init__(self, device: str = 'cuda:0'):
        """Initialize pruner with specified device."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def prune_beams_gpu(self, beam_items: List[Tuple[Any, Tuple]], 
                       beam_width: int = 100) -> List[Tuple[Any, Tuple]]:
        """
        Prune beams using GPU-accelerated top-k selection.
        
        Args:
            beam_items: List of (key, (log_score, energy, ...)) tuples
            beam_width: Maximum number of beams to keep
        
        Returns:
            Pruned list of beam items
        """
        if not beam_items or len(beam_items) <= beam_width:
            return beam_items
        
        if self.device.type == 'cpu':
            return self._prune_beams_cpu(beam_items, beam_width)
        
        # Extract scores
        scores = torch.tensor(
            [item[1][0] for item in beam_items],  # log_score is first element
            device=self.device,
            dtype=torch.float32
        )
        
        # Get top-k indices
        k = min(beam_width, len(scores))
        top_scores, top_indices = torch.topk(scores, k=k)
        
        # Return pruned beams
        return [beam_items[i] for i in top_indices.cpu().numpy()]
    
    def _prune_beams_cpu(self, beam_items: List[Tuple[Any, Tuple]],
                        beam_width: int = 100) -> List[Tuple[Any, Tuple]]:
        """CPU fallback implementation."""
        if len(beam_items) <= beam_width:
            return beam_items
        
        # Sort by log_score (first element of tuple)
        sorted_items = sorted(beam_items, key=lambda x: x[1][0], reverse=True)
        return sorted_items[:beam_width]
    
    def batch_prune_chunks(self, chunk_scores: torch.Tensor,
                          max_chunks: int = 100) -> torch.Tensor:
        """
        Prune chunks based on scores using GPU.
        
        Args:
            chunk_scores: Tensor of chunk scores
            max_chunks: Maximum number of chunks to keep
        
        Returns:
            Indices of top chunks
        """
        if self.device.type == 'cpu':
            # CPU fallback
            scores_np = chunk_scores.numpy()
            if len(scores_np) <= max_chunks:
                return torch.arange(len(scores_np))
            
            top_indices = np.argsort(scores_np)[-max_chunks:][::-1]
            return torch.tensor(top_indices)
        
        # GPU implementation
        chunk_scores_gpu = chunk_scores.to(self.device)
        k = min(max_chunks, len(chunk_scores_gpu))
        
        if k == len(chunk_scores_gpu):
            return torch.arange(len(chunk_scores_gpu))
        
        top_scores, top_indices = torch.topk(chunk_scores_gpu, k=k)
        return top_indices.cpu()