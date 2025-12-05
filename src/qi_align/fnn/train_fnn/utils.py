# ================================================================
# utils.py
# Utility helpers for training
# ================================================================
import torch
import random
import numpy as np

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
