# Stanford OpenVaccine — mRNA Degradation Prediction

Predict mRNA degradation rates at single-nucleotide resolution to help design more stable COVID-19 vaccines.

Based on the [Kaggle OpenVaccine competition](https://www.kaggle.com/competitions/stanford-covid-vaccine/overview) by Stanford DasLab.

## Quickstart

```bash
# Set Kaggle credentials (kaggle.com → Account → API → Create New Token)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

bash prepare.sh
bash eval/eval.sh 2>&1 | tee run.log
grep "^mcrmse:" run.log
```

## Task

Improve `train.py` to minimize MCRMSE on the validation set. See `program.md` for full instructions.

## Evaluation

- **Metric**: MCRMSE (mean column-wise RMSE across 3 scored targets)
- **Lower is better**
- Baseline (2-layer biGRU, 30 epochs): ~0.663
- Good solutions (SNR weighting + BPPS): ~0.50
- Strong solutions (attention / GNN / ensemble): ~0.35
