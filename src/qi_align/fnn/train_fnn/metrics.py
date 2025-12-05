# ================================================================
# metrics.py
# Metrics for FNN multitask learning
# ================================================================
import torch

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def break_auc(pred, target):
    # simple proxy: threshold = 0.5
    correct = ((pred > 0.5).int() == target.int()).float().mean().item()
    return correct

def score_loss(pred, target):
    return ((pred - target)**2).mean().item()
