"""
GPU-accelerated strobe indexing and anchor generation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import numpy as np

class CUDAStrobeProcessor:
    """GPU-accelerated strobe indexing and anchor generation."""
    
    def __init__(self, device: str = 'cuda:0'):
        """Initialize processor with specified device."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"StrobeProcessor using device: {self.device}")
    
    def build_index_gpu(self, strobe_hashes: torch.Tensor, 
                       strobe_positions: torch.Tensor,
                       max_occ: int = 200) -> Dict[int, np.ndarray]:
        """
        Build index with occurrence filtering on GPU.
        
        Args:
            strobe_hashes: Tensor of strobe hashes
            strobe_positions: Corresponding positions
            max_occ: Maximum occurrences per hash
        
        Returns:
            Dictionary mapping hashes to positions
        """
        if self.device.type == 'cpu':
            # Fallback to CPU implementation
            return self._build_index_cpu(strobe_hashes.cpu().numpy(), 
                                       strobe_positions.cpu().numpy(), 
                                       max_occ)
        
        # Move tensors to GPU
        hashes_gpu = strobe_hashes.to(self.device)
        positions_gpu = strobe_positions.to(self.device)
        
        # Get unique hashes and counts
        unique_hashes, inverse_indices, counts = torch.unique(
            hashes_gpu, return_inverse=True, return_counts=True
        )
        
        # Filter by max_occ
        valid_mask = counts <= max_occ
        filtered_hashes = unique_hashes[valid_mask]
        
        # Create position lists for valid hashes
        index = {}
        
        # Process each valid hash
        for i, hash_val in enumerate(filtered_hashes.cpu().numpy()):
            # Find positions for this hash
            mask = (hashes_gpu == hash_val)
            positions = positions_gpu[mask].cpu().numpy()
            index[int(hash_val)] = positions
        
        return index
    
    def _build_index_cpu(self, strobe_hashes: np.ndarray,
                        strobe_positions: np.ndarray,
                        max_occ: int = 200) -> Dict[int, np.ndarray]:
        """CPU fallback implementation."""
        from collections import defaultdict
        
        # Group positions by hash
        hash_to_positions = defaultdict(list)
        for h, pos in zip(strobe_hashes, strobe_positions):
            hash_to_positions[h].append(pos)
        
        # Filter by max_occ
        filtered_index = {}
        for h, positions in hash_to_positions.items():
            if len(positions) <= max_occ:
                filtered_index[int(h)] = np.array(positions)
        
        return filtered_index
    
    def convert_strobes_to_tensors(self, strobes: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert strobe objects to hash and position tensors.
        
        Args:
            strobes: List of strobe objects with .hash and .pos1 attributes
        
        Returns:
            Tuple of (hashes_tensor, positions_tensor)
        """
        hashes = []
        positions = []
        
        for st in strobes:
            hashes.append(st.hash)
            positions.append(st.pos1)
        
        hashes_tensor = torch.tensor(hashes, dtype=torch.int64, device=self.device)
        positions_tensor = torch.tensor(positions, dtype=torch.int64, device=self.device)
        
        return hashes_tensor, positions_tensor

def batch_generate_anchors_gpu(
    seq1_strobe_hashes: torch.Tensor,
    seq1_positions: torch.Tensor,
    seq2_index: Dict[int, np.ndarray],
    seq2_strobe_positions: torch.Tensor = None
) -> List[Tuple[int, int, int]]:
    """
    Batch generate anchors using GPU tensor operations.
    
    Args:
        seq1_strobe_hashes: Tensor of hashes from seq1
        seq1_positions: Corresponding positions in seq1
        seq2_index: Dictionary mapping hashes to positions in seq2
        seq2_strobe_positions: Optional tensor of all seq2 positions
    
    Returns:
        List of (qpos, tpos, hash) tuples
    """
    anchors = []
    device = seq1_strobe_hashes.device
    
    if device.type == 'cpu':
        # CPU fallback
        for i in range(len(seq1_strobe_hashes)):
            hash_val = int(seq1_strobe_hashes[i].item())
            qpos = int(seq1_positions[i].item())
            
            if hash_val in seq2_index:
                for tpos in seq2_index[hash_val]:
                    anchors.append((qpos, tpos, hash_val))
        return anchors
    
    # GPU implementation
    if seq2_strobe_positions is not None:
        # If we have all seq2 positions, use tensor operations
        seq2_hashes_tensor = torch.tensor(
            list(seq2_index.keys()), 
            dtype=torch.int64,
            device=device
        )
        
        # Find matching hashes
        hash_set = set(seq2_index.keys())
        for i in range(len(seq1_strobe_hashes)):
            hash_val = int(seq1_strobe_hashes[i].item())
            if hash_val in hash_set:
                qpos = int(seq1_positions[i].item())
                for tpos in seq2_index[hash_val]:
                    anchors.append((qpos, tpos, hash_val))
    else:
        # Slower but memory-efficient method
        for i in range(len(seq1_strobe_hashes)):
            hash_val = int(seq1_strobe_hashes[i].item())
            qpos = int(seq1_positions[i].item())
            
            if hash_val in seq2_index:
                for tpos in seq2_index[hash_val]:
                    anchors.append((qpos, tpos, hash_val))
    
    return anchors