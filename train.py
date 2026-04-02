"""
OpenVaccine mRNA Degradation Predictor — exp3

Building on exp1 (SNR weighting + wider GRU + cosine LR):
- Per-position error weighting (weight loss by 1/(1+error) per position)
- Structural features from dot-bracket: is_paired, partner_dist, stem_len
- Both sample SNR and per-position error in the loss
- Keep set_num_threads(2), hidden=256 3L biGRU, BPPS aggregates (zeros), cosine LR, 75 epochs
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import math

# Fix thread count to match container CPU quota (2 CPUs)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ======================= CONFIG =======================
SEED = 42          # keep in sync with eval/score.py
VAL_SPLIT = 0.2    # keep in sync with eval/score.py
BATCH_SIZE = 64
EPOCHS = 75
LR = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
ERROR_KEYS = ["reactivity_error", "deg_error_Mg_pH10", "deg_error_pH10", "deg_error_Mg_50C", "deg_error_50C"]
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


def parse_structure(structure):
    """Extract base-pair partners from dot-bracket notation."""
    n = len(structure)
    partner = [-1] * n  # -1 = unpaired
    stack = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                partner[i] = j
                partner[j] = i
    return partner


def structural_features(structure):
    """
    Per-position structural features:
    - is_paired: 1 if in a stem, 0 otherwise
    - partner_norm: normalized partner distance (0 if unpaired, else dist/L)
    - stem_local: fraction of local 3-window that is paired
    """
    n = len(structure)
    partner = parse_structure(structure)
    is_paired = np.array([1.0 if partner[i] >= 0 else 0.0 for i in range(n)], dtype=np.float32)
    partner_norm = np.array([
        abs(partner[i] - i) / n if partner[i] >= 0 else 0.0
        for i in range(n)
    ], dtype=np.float32)
    # local paired fraction (window 3)
    stem_local = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo, hi = max(0, i-1), min(n, i+2)
        stem_local[i] = is_paired[lo:hi].mean()
    return np.stack([is_paired, partner_norm, stem_local], axis=1)  # (N, 3)


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

        # Structural features from dot-bracket
        struct_feat = torch.tensor(structural_features(row["structure"]), dtype=torch.float32)  # (N, 3)

        labels = np.zeros((SEQ_SCORED, len(TARGETS)), dtype=np.float32)
        errors = np.ones((SEQ_SCORED, len(TARGETS)), dtype=np.float32)  # default: no weighting
        for i, (t, ek) in enumerate(zip(TARGETS, ERROR_KEYS)):
            if t in row and row[t]:
                vals = row[t][:SEQ_SCORED]
                labels[:len(vals), i] = vals
            if ek in row and row[ek]:
                errs = row[ek][:SEQ_SCORED]
                errors[:len(errs), i] = errs

        snr = float(row.get("signal_to_noise", 1.0))
        return seq_t, struct_t, loop_t, struct_feat, torch.tensor(labels), torch.tensor(errors), snr, row["id"]


def collate_fn(batch):
    seqs, structs, loops, sfeat_list, labels, errors, snrs, ids = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    B = len(seqs)
    seq_p    = torch.zeros(B, max_len, dtype=torch.long)
    struct_p = torch.zeros(B, max_len, dtype=torch.long)
    loop_p   = torch.zeros(B, max_len, dtype=torch.long)
    sfeat_p  = torch.zeros(B, max_len, 3, dtype=torch.float32)
    for i, (s, st, l, sf) in enumerate(zip(seqs, structs, loops, sfeat_list)):
        n = len(s)
        seq_p[i, :n]    = s
        struct_p[i, :n] = st
        loop_p[i, :n]   = l
        sfeat_p[i, :n, :] = sf
    return (seq_p, struct_p, loop_p, sfeat_p,
            torch.stack(labels), torch.stack(errors),
            torch.tensor(snrs, dtype=torch.float32), list(ids))


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_emb    = nn.Embedding(len(SEQ_VOCAB) + 1,    32)
        self.struct_emb = nn.Embedding(len(STRUCT_VOCAB) + 1, 16)
        self.loop_emb   = nn.Embedding(len(LOOP_VOCAB) + 1,   16)
        # Input: 32 + 16 + 16 + 3 (struct_feat) = 67
        self.gru = nn.GRU(
            67, HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_SIZE * 2, len(TARGETS))

    def forward(self, seq, struct, loop, sfeat):
        x = torch.cat([self.seq_emb(seq), self.struct_emb(struct), self.loop_emb(loop), sfeat], dim=-1)
        out, _ = self.gru(x)
        return self.head(self.drop(out))  # (B, L, n_targets)


def error_snr_weighted_loss(preds, labels, errors, snr_weights):
    """
    Combined per-position error weighting and per-sample SNR weighting.
    Per-position: w = 1/(1 + error_normalized)
    Per-sample: w = snr / sum(snr)
    """
    # Error-based per-position weights (B, L, T)
    # Normalize errors by global median to avoid scale issues
    err_median = errors.median().clamp(min=1e-6)
    err_norm = errors / err_median
    pos_weights = 1.0 / (1.0 + err_norm)  # (B, L, T)

    # Weighted MSE per sample
    mse = ((preds - labels) ** 2 * pos_weights).sum(dim=(1, 2)) / pos_weights.sum(dim=(1, 2)).clamp(min=1e-6)  # (B,)

    # SNR sample weighting
    w = snr_weights / snr_weights.sum()
    return (mse * w).sum() * len(snr_weights)


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

    model     = GRUModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_score = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for seq, struct, loop, sfeat, labels, errors, snrs, _ in train_loader:
            seq, struct, loop, sfeat, labels, errors, snrs = (
                seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE),
                sfeat.to(DEVICE), labels.to(DEVICE), errors.to(DEVICE), snrs.to(DEVICE)
            )
            preds = model(seq, struct, loop, sfeat)[:, :SEQ_SCORED, :]
            loss  = error_snr_weighted_loss(preds, labels, errors, snrs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, struct, loop, sfeat, labels, _, _, _ in val_loader:
                seq, struct, loop, sfeat = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE), sfeat.to(DEVICE)
                all_preds.append(model(seq, struct, loop, sfeat)[:, :SEQ_SCORED, :].cpu())
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
        for seq, struct, loop, sfeat, _, _, _, ids in val_loader:
            seq, struct, loop, sfeat = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE), sfeat.to(DEVICE)
            preds = model(seq, struct, loop, sfeat)[:, :SEQ_SCORED, :].cpu().numpy()
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
