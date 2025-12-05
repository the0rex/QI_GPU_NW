# ================================================================
# predictor.py
# Unified FNN for:
#   (1) Chunk size prediction
#   (2) Chain break prediction
#   (3) Local scoring adjustment
# ================================================================
import numpy as np
import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Feature extractor for genomic windows
# ------------------------------------------------------------
def extract_features(seq4_window):
    """
    seq4_window: 4-bit array of a window around a chain region.
    Returns 32-dim normalized feature vector.
    Features include:
        - base frequencies
        - GC%
        - entropy estimate
        - minimizer density (approx via 4bit uniqueness)
        - low complexity score
    """
    v = np.zeros(32, dtype=np.float32)

    # Base counts
    for b in range(5):
        v[b] = np.mean(seq4_window == b)

    # GC%
    v[5] = v[1] + v[2]  # C + G

    # entropy estimate
    p = v[:4]
    entropy = -np.sum([pi*np.log2(pi+1e-9) for pi in p])
    v[6] = entropy

    # low-complexity estimate (fraction of runs)
    diffs = np.diff(seq4_window)
    v[7] = np.mean(diffs == 0)

    # approximate minimizer density proxy:
    # count of unique values in sliding windows of 10
    uniq = []
    sw = 10
    for i in range(0, len(seq4_window)-sw, sw):
        uniq.append(len(set(seq4_window[i:i+sw])))
    v[8] = np.mean(uniq) / 10.0

    # fill rest with zeros for future extension
    return v


# ------------------------------------------------------------
# Unified Neural Network
# ------------------------------------------------------------
class FNNPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared feature encoder
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)

        # Head 1: Chunk size classification (5 classes)
        self.chunk_head = nn.Linear(64, 5)

        # Head 2: Chain break probability (0–1)
        self.break_head = nn.Linear(64, 1)

        # Head 3: Scoring adjustments (Δmatch, Δmismatch, Δgamma)
        self.score_head = nn.Linear(64, 3)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        chunk_logits = self.chunk_head(h)        # softmax later
        break_prob  = torch.sigmoid(self.break_head(h))
        score_adj   = self.score_head(h)         # raw values

        return chunk_logits, break_prob, score_adj


# ------------------------------------------------------------
# Load model or fallback to defaults
# ------------------------------------------------------------
class PredictorWrapper:
    def __init__(self, model_path="model/fnn.pt"):
        self.model = FNNPredictor()

        if Path(model_path).exists():
            print("[FNN] Loading trained model:", model_path)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.trained = True
        else:
            print("[FNN] No trained model found → using default heuristic mode.")
            self.trained = False

        self.model.eval()

    # -------------------------
    # Predict all outputs
    # -------------------------
    def predict(self, seq4_window):
        feats = extract_features(seq4_window)
        x = torch.from_numpy(feats).float().unsqueeze(0)

        if not self.trained:
            return self._default_predict(feats)

        with torch.no_grad():
            chunk_logits, break_prob, score_adj = self.model(x)

            chunk_index = int(torch.argmax(chunk_logits).item())
            break_prob = float(break_prob.item())
            score_adj = score_adj.squeeze(0).numpy().tolist()

        return {
            "chunk_index": chunk_index,
            "break_prob": break_prob,
            "score_adj": score_adj
        }

    # -------------------------
    # Default fallback model
    # -------------------------
    def _default_predict(self, feats):
        gc = feats[5]
        entropy = feats[6]
        lc = feats[7]

        # Heuristic chunk size
        if gc > 0.6 or entropy < 1.2:
            chunk_idx = 0   # small chunk (20k)
        elif lc > 0.50:
            chunk_idx = 1   # 30k
        else:
            chunk_idx = 3   # default 50k

        # chain break predictor: avoid break in low complexity
        break_prob = float(min(1.0, lc * 2.0))

        # scoring adjustment
        # ↓match slightly in repetitive regions
        # ↑gamma in regulatory-like regions (entropy high)
        score_adj = [
            float((gc - 0.5) * 2),      # Δmatch
            float((entropy - 1.5) * 0.5),  # Δmismatch
            float((entropy - 1.0) * 0.8),  # Δgamma
        ]

        return {
            "chunk_index": chunk_idx,
            "break_prob": break_prob,
            "score_adj": score_adj
        }
