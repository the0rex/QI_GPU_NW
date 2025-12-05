# ================================================================
# train_fnn.py
# Multitask training pipeline for QI-ALIGN FNN
# ================================================================
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FNNDataset
from metrics import accuracy, break_auc, score_loss
from utils import set_seed
from qi_align.fnn.predictor import FNNPredictor

def train_fnn(json_path, model_out="fnn.pt", epochs=10):
    set_seed(1234)

    records = json.load(open(json_path))
    dataset = FNNDataset(records)

    # 90/10 train/val split
    n_train = int(len(dataset)*0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = FNNPredictor()
    model.train()

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for feats, chunk_class, break_label, score_adj in train_loader:
            opt.zero_grad()

            chunk_logits, break_prob, score_pred = model(feats)

            loss1 = criterion_cls(chunk_logits, chunk_class)
            loss2 = criterion_bce(break_prob.squeeze(), break_label)
            loss3 = criterion_mse(score_pred, score_adj)

            loss = loss1 + loss2 + 0.5 * loss3
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # validation metrics
        model.eval()
        with torch.no_grad():
            acc = 0
            auc = 0
            sl = 0
            count = 0
            for feats, chunk_class, break_label, score_adj in val_loader:
                ch, br, sc = model(feats)
                acc += accuracy(ch, chunk_class)
                auc += break_auc(br.squeeze(), break_label)
                sl += score_loss(sc, score_adj)
                count += 1

        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss:.2f}, "
              f"acc={acc/count:.3f}, break={auc/count:.3f}, score_loss={sl/count:.3f}")

    torch.save(model.state_dict(), model_out)
    print("[OK] Training complete. Saved model →", model_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="fnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_fnn(args.data, args.out, args.epochs)
