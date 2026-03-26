# Stanford OpenVaccine — mRNA Degradation Prediction

Improve a PyTorch model that predicts mRNA degradation rates at single-nucleotide resolution, minimizing MCRMSE on a held-out validation set.

## Setup

1. **Read the in-scope files**:
   - `train.py` — the model training script. You modify this.
   - `eval/eval.sh` — runs evaluation. Do not modify.
   - `eval/score.py` — computes MCRMSE from predictions. Do not modify.
   - `prepare.sh` — downloads the Kaggle dataset. Do not modify.
2. **Set Kaggle credentials**: Export `KAGGLE_USERNAME` and `KAGGLE_KEY` (from kaggle.com → Account → API → Create New Token).
3. **Run prepare**: `bash prepare.sh` to install deps and download the dataset.
4. **Verify data**: Check that `data/train.json` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row: `score\tnotes`.
6. **Run baseline**: `bash eval/eval.sh 2>&1 | tee run.log` to establish the starting score.

## The benchmark

The [Stanford OpenVaccine](https://www.kaggle.com/competitions/stanford-covid-vaccine/overview) competition tasks you with predicting mRNA degradation rates at each nucleotide position under different chemical conditions. The training set contains ~3,000 RNA sequences (107 nt each) with 5 per-position degradation targets measured via chemical mapping. Evaluation is performed on a fixed 20% held-out validation split.

**Input features per sequence:**
- `sequence` — nucleotide sequence (A/G/C/U)
- `structure` — secondary structure in dot-bracket notation (`.()/`)
- `predicted_loop_type` — per-position loop type (S/M/I/B/H/E/X)
- `bpps` — base-pairing probability matrix (107×107)

**Prediction targets (5 per position, scored on first 68 positions):**
- `reactivity` — determines likely secondary structure
- `deg_Mg_pH10` — degradation with Mg at high pH *(scored)*
- `deg_pH10` — degradation without Mg at high pH
- `deg_Mg_50C` — degradation with Mg at high temperature *(scored)*
- `deg_50C` — degradation without Mg at high temperature

## Experimentation

**What you CAN do:**
- Modify `train.py` freely — change the model architecture, features, hyperparameters, training loop, or add new helper files (e.g. `model.py`, `dataset.py`).
- Use the base-pairing probability matrix (`bpps`) as additional input (it's in `train.json`).
- Install additional Python packages.

**What you CANNOT do:**
- Modify `eval/`, `prepare.sh`, or the data files.
- Use pretrained foundation models (e.g. RNA-FM, SpliceBERT, ESM, or any LLM-based weights).
- Use external datasets beyond the provided Kaggle data.
- Change `SEED` or `VAL_SPLIT` in `train.py` (they must stay in sync with `eval/score.py`).
- Exceed 30 minutes of total wall-clock training time per eval run.

**The goal: minimize `mcrmse`** — the mean column-wise RMSE across `reactivity`, `deg_Mg_pH10`, and `deg_Mg_50C` at positions 0–67. Lower is better.
- Baseline (2-layer biGRU, 30 epochs, no SNR weighting): ~0.663
- Good solutions (SNR weighting + BPPS features): ~0.50
- Strong solutions (attention / GNN / ensembling): ~0.35

**Simplicity criterion**: All else being equal, simpler is better.

## Ideas to explore

- Use the `bpps` matrix as a graph adjacency for a GCN or attention layer
- Add Transformer encoder layers or multi-head attention
- Ensemble multiple folds or model types (GRU + Conv1D, etc.)
- Weight training samples by `signal_to_noise` ratio
- Add positional embeddings
- Deeper or wider GRU/LSTM layers
- Label smoothing or noise injection for regularization

## Output format

`eval/eval.sh` prints to stdout:

```
---
mcrmse:           <value>
correct:          <N>
total:            <N>
```

Read the score via: `grep "^mcrmse:" run.log`

## Logging results

Log each experiment to `results.tsv` (tab-separated):

```
commit    mcrmse    status    description
a1b2c3d    0.663    keep    baseline
b2c3d4e    0.612    keep    added snr weighting
```

Do not commit `results.tsv`.

## The experiment loop

LOOP FOREVER:

1. **THINK** — decide what to try next. Review `results.tsv`. Consider: SNR weighting, BPPS features, more layers, attention, ensembling.
2. Modify `train.py` (and any helper files) with your experimental idea.
3. `git commit`
4. Run the experiment: `bash eval/eval.sh 2>&1 | tee run.log`
5. Read the results: `grep "^mcrmse:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace and attempt a fix.
7. **Review artifacts**: check `predictions.csv` for per-target RMSE breakdown.
8. Record the results in `results.tsv` (do not commit it).
9. If `mcrmse` improved, keep the git commit. If equal or worse, `git reset --hard HEAD~1`.

**Timeout**: If a run exceeds 30 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until interrupted.
