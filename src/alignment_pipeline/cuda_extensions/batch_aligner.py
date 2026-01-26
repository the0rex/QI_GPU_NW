"""
GPU-accelerated batch alignment scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any

class CUDABatchAligner:
    """GPU-accelerated batch alignment operations."""
    
    def __init__(self, device: str = 'cuda:0'):
        """Initialize batch aligner."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Define scoring parameters (can be configured)
        self.match_score = 1.0
        self.mismatch_score = -2.0
        self.gap_open = -4.0
        self.gap_extend = -0.5
        
    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode DNA sequences to one-hot representation.
        
        Args:
            sequences: List of DNA sequence strings
        
        Returns:
            One-hot encoded tensor of shape (batch, length, 4)
        """
        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)
        
        # Create tensor
        encoded = torch.zeros((batch_size, max_len, 4), 
                            device=self.device, 
                            dtype=torch.float32)
        
        # Mapping from base to index
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        
        for i, seq in enumerate(sequences):
            seq_upper = seq.upper()
            for j, base in enumerate(seq_upper):
                idx = base_to_idx.get(base, 0)  # Default to A/N
                encoded[i, j, idx] = 1.0
        
        return encoded
    
    def compute_quick_scores(self, seqs1: List[str], 
                           seqs2: List[str]) -> torch.Tensor:
        """
        Compute quick alignment score estimates for batch.
        
        Args:
            seqs1: List of first sequences
            seqs2: List of second sequences
        
        Returns:
            Tensor of estimated scores
        """
        if len(seqs1) != len(seqs2):
            raise ValueError("Batch sizes must match")
        
        batch_size = len(seqs1)
        scores = torch.zeros(batch_size, device=self.device)
        
        # Simple match counting (can be replaced with more sophisticated GPU kernel)
        for i in range(batch_size):
            # Pad sequences to same length for comparison
            max_len = max(len(seqs1[i]), len(seqs2[i]))
            s1 = seqs1[i].ljust(max_len, 'N')
            s2 = seqs2[i].ljust(max_len, 'N')
            
            # Count matches
            matches = sum(1 for a, b in zip(s1, s2) if a == b and a != 'N' and b != 'N')
            scores[i] = matches * self.match_score
        
        return scores
    
    def filter_chunks_by_score(self, chunks: List[Any], 
                             seq1_tuples: List[Tuple[bytes, int]],
                             seq2_tuples: List[Tuple[bytes, int]],
                             threshold: float = 0.0) -> List[Any]:
        """
        Filter chunks based on quick score estimates.
        
        Args:
            chunks: List of chunk objects
            seq1_tuples: List of (buffer, length) for seq1 chunks
            seq2_tuples: List of (buffer, length) for seq2 chunks
            threshold: Score threshold for filtering
        
        Returns:
            Filtered list of chunks
        """
        if self.device.type == 'cpu' or len(chunks) <= 1:
            return chunks
        
        # Extract sequences
        from ...core.compression import decompress_slice
        
        seqs1 = []
        seqs2 = []
        
        for i, chunk in enumerate(chunks):
            buf1, L1 = seq1_tuples[i]
            buf2, L2 = seq2_tuples[i]
            
            s1 = decompress_slice(buf1, L1, chunk.q_start, chunk.q_end)
            s2 = decompress_slice(buf2, L2, chunk.t_start, chunk.t_end)
            
            seqs1.append(s1)
            seqs2.append(s2)
        
        # Compute scores
        scores = self.compute_quick_scores(seqs1, seqs2)
        
        # Filter by threshold
        keep_indices = (scores >= threshold).nonzero().flatten().cpu().numpy()
        
        return [chunks[i] for i in keep_indices]

def compute_batch_scores_gpu(
    seq1_chunks: List[str],
    seq2_chunks: List[str],
    gap_open: float = -4.0,
    gap_extend: float = -0.5,
    match_score: float = 1.0,
    mismatch_score: float = -2.0
) -> torch.Tensor:
    """
    Compute alignment scores for batch of chunks on GPU.
    
    Args:
        seq1_chunks: List of sequence 1 chunks
        seq2_chunks: List of sequence 2 chunks
        gap_open: Gap opening penalty
        gap_extend: Gap extension penalty
        match_score: Match score
        mismatch_score: Mismatch score
    
    Returns:
        Tensor of alignment scores
    """
    if not torch.cuda.is_available():
        # CPU fallback
        scores = torch.zeros(len(seq1_chunks))
        for i in range(len(seq1_chunks)):
            # Simple scoring
            matches = sum(1 for a, b in zip(seq1_chunks[i], seq2_chunks[i]) 
                         if a == b)
            scores[i] = matches * match_score
        return scores
    
    # GPU implementation
    device = torch.device('cuda')
    aligner = CUDABatchAligner(device=device)
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.gap_open = gap_open
    aligner.gap_extend = gap_extend
    
    return aligner.compute_quick_scores(seq1_chunks, seq2_chunks)