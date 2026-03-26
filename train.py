"""
OpenVaccine mRNA Degradation Predictor — Simple GRU Baseline

Agents: modify this file to improve the MCRMSE score.
Do NOT modify eval/ or prepare.sh.

This baseline uses a bidirectional GRU with sequence, structure, and loop type embeddings.
Predictions are saved to predictions.csv for scoring by eval/score.py.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# ======================= CONFIG =======================
SEED = 42          # keep in sync with eval/score.py
VAL_SPLIT = 0.2    # keep in sync with eval/score.py
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SEQ_SCORED = 68
# ======================================================

SEQ_VOCAB  = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_VOCAB = {".": 0, "(": 1, ")": 2}
LOOP_VOCAB = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq_t    = torch.tensor([SEQ_VOCAB.get(c, 0)    for c in row["sequence"]],            dtype=torch.long)
        struct_t = torch.tensor([STRUCT_VOCAB.get(c, 0) for c in row["structure"]],           dtype=torch.long)
        loop_t   = torch.tensor([LOOP_VOCAB.get(c, 0)   for c in row["predicted_loop_type"]], dtype=torch.long)

        labels = np.zeros((SEQ_SCORED, len(TARGETS)), dtype=np.float32)
        for i, t in enumerate(TARGETS):
            if t in row and row[t]:
                vals = row[t][:SEQ_SCORED]
                labels[:len(vals), i] = vals

        return seq_t, struct_t, loop_t, torch.tensor(labels), row["id"]


def collate_fn(batch):
    seqs, structs, loops, labels, ids = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    B = len(seqs)
    seq_p    = torch.zeros(B, max_len, dtype=torch.long)
    struct_p = torch.zeros(B, max_len, dtype=torch.long)
    loop_p   = torch.zeros(B, max_len, dtype=torch.long)
    for i, (s, st, l) in enumerate(zip(seqs, structs, loops)):
        seq_p[i, :len(s)]    = s
        struct_p[i, :len(st)] = st
        loop_p[i, :len(l)]   = l
    return seq_p, struct_p, loop_p, torch.stack(labels), list(ids)


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb    = nn.Embedding(len(SEQ_VOCAB) + 1,    32)
        self.struct_emb = nn.Embedding(len(STRUCT_VOCAB) + 1, 16)
        self.loop_emb   = nn.Embedding(len(LOOP_VOCAB) + 1,   16)
        self.gru = nn.GRU(
            64, HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_SIZE * 2, len(TARGETS))

    def forward(self, seq, struct, loop):
        x = torch.cat([self.seq_emb(seq), self.struct_emb(struct), self.loop_emb(loop)], dim=-1)
        out, _ = self.gru(x)
        return self.head(self.drop(out))  # (B, L, n_targets)


def mcrmse(preds, labels):
    scored_idx = [TARGETS.index(t) for t in SCORED_TARGETS]
    return torch.sqrt(((preds[:, :, scored_idx] - labels[:, :, scored_idx]) ** 2).mean(dim=(0, 1))).mean().item()


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    all_data = load_json("data/train.json")

    # Fixed train/val split — must match eval/score.py
    np.random.seed(SEED)
    idx = np.random.permutation(len(all_data))
    val_size = int(len(all_data) * VAL_SPLIT)
    val_idx = set(idx[:val_size].tolist())
    train_split = [d for i, d in enumerate(all_data) if i not in val_idx]
    val_split   = [d for i, d in enumerate(all_data) if i in val_idx]

    # Filter training by signal quality
    train_filtered = [d for d in train_split if d.get("signal_to_noise", 0) >= 1.0]
    print(f"Train: {len(train_filtered)} (filtered from {len(train_split)}) | Val: {len(val_split)}")

    train_loader = DataLoader(RNADataset(train_filtered), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(RNADataset(val_split),      batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = GRUModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_score = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for seq, struct, loop, labels, _ in train_loader:
            seq, struct, loop, labels = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE), labels.to(DEVICE)
            preds = model(seq, struct, loop)[:, :SEQ_SCORED, :]
            loss  = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, struct, loop, labels, _ in val_loader:
                seq, struct, loop = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE)
                all_preds.append(model(seq, struct, loop)[:, :SEQ_SCORED, :].cpu())
                all_labels.append(labels)

        val_score = mcrmse(torch.cat(all_preds), torch.cat(all_labels))
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | val MCRMSE: {val_score:.4f} | best: {best_score:.4f}")

    # Save predictions with best model
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, struct, loop, _, ids in val_loader:
            seq, struct, loop = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE)
            preds = model(seq, struct, loop)[:, :SEQ_SCORED, :].cpu().numpy()
            for b, sid in enumerate(ids):
                for pos in range(SEQ_SCORED):
                    row = {"id_seqpos": f"{sid}_{pos}"}
                    row.update({t: float(preds[b, pos, k]) for k, t in enumerate(TARGETS)})
                    rows.append(row)

    pd.DataFrame(rows).to_csv("predictions.csv", index=False)
    print(f"\nBest val MCRMSE: {best_score:.4f}")
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
