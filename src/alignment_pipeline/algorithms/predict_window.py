# predict_window_size.py (updated)
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from itertools import product
from collections import Counter
from math import log2

MODEL_PATH = "src/alignment_pipeline/algorithms/predictor_models/model_Adam.h5"
X_SCALER_PATH = "src/alignment_pipeline/algorithms/predictor_models/x_scaler.pkl"
Y_SCALER_PATH = "src/alignment_pipeline/algorithms/predictor_models/y_scaler.pkl"

# Feature column order used in trainingb
FEATURE_COLS = [
    'Seq1_GC', 'Seq2_GC', 'GC_Ratio_Diff',
    'Seq1_Entropy', 'Seq2_Entropy',
    'Seq1_Length', 'Seq2_Length', 'Length_Diff',
    'Seq_Identity', 'Kmer_Similarity'
]

# ---------------------------
# Utility functions (match training)
# ---------------------------
def compute_gc_content(seq):
    seq = str(seq).upper().replace("-", "")
    L = len(seq)
    if L == 0:
        return 0.0
    gc = seq.count('G') + seq.count('C')
    return (gc / L) * 100.0   # percent (matches train_fnn.py)

def compute_shannon_entropy(seq):
    seq = str(seq).upper().replace("-", "")
    L = len(seq)
    if L == 0:
        return 0.0
    bases = ['A', 'T', 'C', 'G']
    entropy = 0.0
    for b in bases:
        p = seq.count(b) / L
        if p > 0:
            entropy -= p * log2(p)
    return entropy

def seq_length_no_gaps(seq):
    return len(str(seq).replace("-", ""))

def gc_ratio_difference(seq1, seq2):
    return abs(compute_gc_content(seq1) - compute_gc_content(seq2))

def sequence_identity(seq1, seq2):
    s1 = str(seq1).upper()
    s2 = str(seq2).upper()
    # compare aligned positions where both exist (allow different lengths)
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    matches = 0
    count_positions = 0
    for i in range(min_len):
        a = s1[i]
        b = s2[i]
        if a == "-" or b == "-":
            continue
        count_positions += 1
        if a == b:
            matches += 1
    if count_positions == 0:
        return 0.0
    return (matches / count_positions) * 100.0  # percent

def kmer_frequencies(sequence, k=3):
    seq = str(sequence).upper().replace("-", "")
    if len(seq) < k:
        return {}
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    total = sum(counts.values())
    return {kmer: v/total for kmer, v in counts.items()}

def kmer_similarity(seq1, seq2, k=3):
    kmer_list = [''.join(p) for p in product('ATCG', repeat=k)]
    f1 = kmer_frequencies(seq1, k)
    f2 = kmer_frequencies(seq2, k)
    v1 = np.array([f1.get(kmer, 0.0) for kmer in kmer_list], dtype=float)
    v2 = np.array([f2.get(kmer, 0.0) for kmer in kmer_list], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return (float(np.dot(v1, v2) / denom) if denom != 0 else 0.0)

# ---------------------------
# 1. Create feature vector DataFrame (10 features)
# ---------------------------
def compute_features_df(seq1, seq2):
    seq1_s = str(seq1)
    seq2_s = str(seq2)

    s1_gc = compute_gc_content(seq1_s)
    s2_gc = compute_gc_content(seq2_s)

    s1_entropy = compute_shannon_entropy(seq1_s)
    s2_entropy = compute_shannon_entropy(seq2_s)

    s1_len = seq_length_no_gaps(seq1_s)
    s2_len = seq_length_no_gaps(seq2_s)
    length_diff = abs(s1_len - s2_len)

    gc_diff = gc_ratio_difference(seq1_s, seq2_s)
    identity = sequence_identity(seq1_s, seq2_s)
    ksim = kmer_similarity(seq1_s, seq2_s, k=3)

    data = {
        'Seq1_GC': [s1_gc],
        'Seq2_GC': [s2_gc],
        'GC_Ratio_Diff': [gc_diff],
        'Seq1_Entropy': [s1_entropy],
        'Seq2_Entropy': [s2_entropy],
        'Seq1_Length': [s1_len],
        'Seq2_Length': [s2_len],
        'Length_Diff': [length_diff],
        'Seq_Identity': [identity],
        'Kmer_Similarity': [ksim]
    }
    return pd.DataFrame(data, columns=FEATURE_COLS)

# ---------------------------
# 2. Load model + scalers only once
# ---------------------------
print("[Predict Window] Loading model and scalers...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# Optional sanity: print what scaler expects (uncomment if debugging)
# print("x_scaler.n_features_in_:", getattr(x_scaler, "n_features_in_", None))
# print("x_scaler.feature_names_in_:", getattr(x_scaler, "feature_names_in_", None))

# ---------------------------
# 3. Predict window size
# ---------------------------
def predict_window(seq1, seq2):
    # Build DataFrame with 10 features in correct order
    X_df = compute_features_df(seq1, seq2)

    # Scale (pass DataFrame so feature names align)
    X_scaled = x_scaler.transform(X_df)

    # Predict
    y_scaled = model.predict(X_scaled, verbose=0)
    window_pred = y_scaler.inverse_transform(y_scaled)[0][0]

    # Return rounded integer (minimum 1)
    return max(1, int(round(float(window_pred)*1.1)))
