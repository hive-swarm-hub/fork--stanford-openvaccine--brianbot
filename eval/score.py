"""
Scoring script for stanford-openvaccine task.
Reads predictions.csv and data/train.json, computes MCRMSE on the validation split.
DO NOT MODIFY.
"""

import json
import sys
import numpy as np
import pandas as pd

SEED = 42
VAL_SPLIT = 0.2
SEQ_SCORED = 68
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def main():
    try:
        all_data = load_json("data/train.json")
    except FileNotFoundError:
        print("ERROR: data/train.json not found. Run prepare.sh first.", file=sys.stderr)
        sys.exit(1)

    # Reconstruct val split — must match train.py
    np.random.seed(SEED)
    idx = np.random.permutation(len(all_data))
    val_size = int(len(all_data) * VAL_SPLIT)
    val_idx = set(idx[:val_size].tolist())
    val_split = [d for i, d in enumerate(all_data) if i in val_idx]

    # Ground truth lookup
    gt = {}
    for sample in val_split:
        sid = sample["id"]
        for pos in range(SEQ_SCORED):
            key = f"{sid}_{pos}"
            gt[key] = {t: sample[t][pos] for t in SCORED_TARGETS if t in sample and sample[t]}

    try:
        preds_df = pd.read_csv("predictions.csv")
    except FileNotFoundError:
        print("ERROR: predictions.csv not found. Make sure train.py saves it.", file=sys.stderr)
        sys.exit(1)

    preds = preds_df.set_index("id_seqpos")

    errors = {t: [] for t in SCORED_TARGETS}
    missing = 0
    for key, truth in gt.items():
        if key not in preds.index:
            missing += 1
            continue
        for t in SCORED_TARGETS:
            if t in truth and t in preds.columns:
                errors[t].append((float(preds.loc[key, t]) - truth[t]) ** 2)

    if missing > 0:
        print(f"WARNING: {missing} predictions missing from predictions.csv", file=sys.stderr)

    rmse_per_target = []
    for t in SCORED_TARGETS:
        if errors[t]:
            rmse = np.sqrt(np.mean(errors[t]))
            rmse_per_target.append(rmse)
            print(f"  RMSE {t}: {rmse:.4f}")

    mcrmse = float(np.mean(rmse_per_target))
    n_preds = len(errors[SCORED_TARGETS[0]])
    n_total = len(val_split) * SEQ_SCORED

    print("")
    print("---")
    print(f"mcrmse:           {mcrmse:.4f}")
    print(f"correct:          {n_preds}")
    print(f"total:            {n_total}")


if __name__ == "__main__":
    main()
