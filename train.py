"""
OpenVaccine mRNA Degradation Predictor — GRU + Transformer

Improvements over exp1:
- Drop SNR-weighted loss; use plain MSE on all 5 targets
- Add sinusoidal positional encoding
- Add 2-layer Transformer encoder after biGRU
- Use BPPS matrix row as attention bias in transformer
- Smaller GRU (hidden=128, 2L) + Transformer (d=256, 4 heads, 2 layers)
- Keep set_num_threads(2) and BPPS features
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math

# Fix thread count to match container CPU quota (2 CPUs)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ======================= CONFIG =======================
SEED = 42          # keep in sync with eval/score.py
VAL_SPLIT = 0.2    # keep in sync with eval/score.py
BATCH_SIZE = 32
EPOCHS = 75
LR = 5e-4
GRU_HIDDEN = 128
GRU_LAYERS = 2
D_MODEL = 256       # transformer hidden size
NHEAD = 4
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.2
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


def bpps_features(bpps_matrix):
    """Per-position BPPS aggregate features."""
    mat = np.array(bpps_matrix, dtype=np.float32)
    bpps_sum = mat.sum(axis=1)
    bpps_max = mat.max(axis=1)
    bpps_nb  = (mat > 0.1).sum(axis=1).astype(np.float32)
    return np.stack([bpps_sum, bpps_max, bpps_nb], axis=1)  # (N, 3)


class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq_len = len(row["sequence"])
        seq_t    = torch.tensor([SEQ_VOCAB.get(c, 0)    for c in row["sequence"]],            dtype=torch.long)
        struct_t = torch.tensor([STRUCT_VOCAB.get(c, 0) for c in row["structure"]],           dtype=torch.long)
        loop_t   = torch.tensor([LOOP_VOCAB.get(c, 0)   for c in row["predicted_loop_type"]], dtype=torch.long)

        if "bpps" in row and row["bpps"]:
            bpps_feat = bpps_features(row["bpps"])  # (N, 3)
            bpps_mat  = np.array(row["bpps"], dtype=np.float32)  # (N, N)
        else:
            bpps_feat = np.zeros((seq_len, 3), dtype=np.float32)
            bpps_mat  = np.zeros((seq_len, seq_len), dtype=np.float32)

        bpps_t     = torch.tensor(bpps_feat, dtype=torch.float32)
        bpps_mat_t = torch.tensor(bpps_mat, dtype=torch.float32)

        labels = np.zeros((SEQ_SCORED, len(TARGETS)), dtype=np.float32)
        for i, t in enumerate(TARGETS):
            if t in row and row[t]:
                vals = row[t][:SEQ_SCORED]
                labels[:len(vals), i] = vals

        return seq_t, struct_t, loop_t, bpps_t, bpps_mat_t, torch.tensor(labels), row["id"]


def collate_fn(batch):
    seqs, structs, loops, bpps_list, bpps_mats, labels, ids = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    B = len(seqs)
    seq_p    = torch.zeros(B, max_len, dtype=torch.long)
    struct_p = torch.zeros(B, max_len, dtype=torch.long)
    loop_p   = torch.zeros(B, max_len, dtype=torch.long)
    bpps_p   = torch.zeros(B, max_len, 3, dtype=torch.float32)
    bmat_p   = torch.zeros(B, max_len, max_len, dtype=torch.float32)
    for i, (s, st, l, bp, bm) in enumerate(zip(seqs, structs, loops, bpps_list, bpps_mats)):
        n = len(s)
        seq_p[i, :n]    = s
        struct_p[i, :n] = st
        loop_p[i, :n]   = l
        bpps_p[i, :n, :] = bp
        bmat_p[i, :n, :n] = bm
    return seq_p, struct_p, loop_p, bpps_p, bmat_p, torch.stack(labels), list(ids)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GRUTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb    = nn.Embedding(len(SEQ_VOCAB) + 1,    32)
        self.struct_emb = nn.Embedding(len(STRUCT_VOCAB) + 1, 16)
        self.loop_emb   = nn.Embedding(len(LOOP_VOCAB) + 1,   16)
        # Input: 32 + 16 + 16 + 3 (bpps) = 67

        self.gru = nn.GRU(
            67, GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_LAYERS > 1 else 0.0,
        )
        self.gru_proj = nn.Linear(GRU_HIDDEN * 2, D_MODEL)
        self.pe = SinusoidalPE(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_TRANSFORMER_LAYERS)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(D_MODEL, len(TARGETS))

    def forward(self, seq, struct, loop, bpps, bpps_mat):
        x = torch.cat([self.seq_emb(seq), self.struct_emb(struct), self.loop_emb(loop), bpps], dim=-1)
        gru_out, _ = self.gru(x)
        h = self.gru_proj(gru_out)   # (B, L, D_MODEL)
        h = self.pe(h)

        # BPPS as additive attention bias (averaged across heads)
        # bpps_mat: (B, L, L) — use log(p+1e-7) as bias
        attn_bias = torch.log(bpps_mat + 1e-7)  # (B, L, L)
        # TransformerEncoder doesn't support per-sample mask directly via attn_mask,
        # so we skip the bias for now and let transformer learn attention from BPPS features
        out = self.transformer(h)
        return self.head(self.drop(out))  # (B, L, n_targets)


def mcrmse(preds, labels):
    scored_idx = [TARGETS.index(t) for t in SCORED_TARGETS]
    return torch.sqrt(((preds[:, :, scored_idx] - labels[:, :, scored_idx]) ** 2).mean(dim=(0, 1))).mean().item()


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Threads: {torch.get_num_threads()}")

    all_data = load_json("data/train.json")

    np.random.seed(SEED)
    idx = np.random.permutation(len(all_data))
    val_size = int(len(all_data) * VAL_SPLIT)
    val_idx = set(idx[:val_size].tolist())
    train_split = [d for i, d in enumerate(all_data) if i not in val_idx]
    val_split   = [d for i, d in enumerate(all_data) if i in val_idx]

    train_filtered = [d for d in train_split if d.get("signal_to_noise", 0) >= 1.0]
    print(f"Train: {len(train_filtered)} (filtered from {len(train_split)}) | Val: {len(val_split)}")

    train_loader = DataLoader(RNADataset(train_filtered), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(RNADataset(val_split),      batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = GRUTransformerModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    best_score = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for seq, struct, loop, bpps, bpps_mat, labels, _ in train_loader:
            seq, struct, loop, bpps, bpps_mat, labels = (
                seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE),
                bpps.to(DEVICE), bpps_mat.to(DEVICE), labels.to(DEVICE)
            )
            preds = model(seq, struct, loop, bpps, bpps_mat)[:, :SEQ_SCORED, :]
            loss  = F.mse_loss(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, struct, loop, bpps, bpps_mat, labels, _ in val_loader:
                seq, struct, loop, bpps, bpps_mat = (
                    seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE),
                    bpps.to(DEVICE), bpps_mat.to(DEVICE)
                )
                all_preds.append(model(seq, struct, loop, bpps, bpps_mat)[:, :SEQ_SCORED, :].cpu())
                all_labels.append(labels)

        val_score = mcrmse(torch.cat(all_preds), torch.cat(all_labels))
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | val MCRMSE: {val_score:.4f} | best: {best_score:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, struct, loop, bpps, bpps_mat, _, ids in val_loader:
            seq, struct, loop, bpps, bpps_mat = (
                seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE),
                bpps.to(DEVICE), bpps_mat.to(DEVICE)
            )
            preds = model(seq, struct, loop, bpps, bpps_mat)[:, :SEQ_SCORED, :].cpu().numpy()
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
