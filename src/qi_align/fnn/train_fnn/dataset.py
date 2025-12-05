# ================================================================
# dataset.py
# PyTorch dataset for FNN multitask training
# ================================================================
import torch
from torch.utils.data import Dataset
import numpy as np

class FNNDataset(Dataset):
    """
    Each sample contains:
        features: 32-dim vector from extract_features()
        chunk_class: integer 0..4
        break_label: float (0 or 1)
        scoring_adjustment: 3 floats
    """
    def __init__(self, records):
        """
        records: list of dicts
        {
            "features": [...32 floats...],
            "chunk_class": int,
            "break_label": 0/1,
            "score_adj": [d_match, d_mismatch, d_gamma]
        }
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        features = torch.tensor(r["features"], dtype=torch.float32)
        chunk_class = torch.tensor(r["chunk_class"], dtype=torch.long)
        break_label = torch.tensor(r["break_label"], dtype=torch.float32)
        score_adj = torch.tensor(r["score_adj"], dtype=torch.float32)
        return features, chunk_class, break_label, score_adj
