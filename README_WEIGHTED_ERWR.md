# Weighted ERWR with continuous synergy scores

This branch adds a weighted ERWR pipeline that preserves a continuous synergy score instead of converting every positive pair to edge weight `1`.

## Supported score columns

Use the name of a numeric column in `extract.xlsx`, for example:

```text
S
HSA
Bliss
ZIP
Loewe
```

The input must also contain the existing binary `result` column. Only rows with `result == 1` are used to construct the training graph, while the selected continuous score controls each edge weight.

## Weight modes

- `rank`: percentile/rank-based weights; recommended when score scales differ between cell lines.
- `minmax`: min-max weights within each training graph.
- `raw`: nonnegative score divided by the maximum training score.

Threshold selection and all evaluation metrics are leakage-safe: the threshold is selected on an inner validation split, while graph weights are built from the outer training split only.

The report includes AUROC, AUPRC, F1, Precision, Recall, Specificity, BalancedAccuracy, Accuracy, and MCC.

## Colab commands

```bash
pip install -q numpy pandas scikit-learn networkx openpyxl
```

Run HSA on drug-pair split:

```bash
python run_weighted_erwr.py \
  --input extract.xlsx \
  --score-column HSA \
  --split drug_pair \
  --folds 5 \
  --seed 42 \
  --weight-mode rank \
  --output output/weighted_erwr/drug_pair_HSA
```

Run all score columns:

```bash
for score in S HSA Bliss ZIP Loewe; do
  python run_weighted_erwr.py \
    --input extract.xlsx \
    --score-column "$score" \
    --split drug_pair \
    --folds 5 \
    --seed 42 \
    --weight-mode rank \
    --output "output/weighted_erwr/drug_pair_${score}"
done
```

For the sample split, replace:

```text
--split drug_pair
```

with:

```text
--split sample
```

For the cell-line split, replace it with:

```text
--split cell_line
```

## Important interpretation

This is an experiment against the binary ERWR baseline. It does not use drug or cell-line embeddings. The selected continuous score is used only to weight positive training edges, preventing the loss of synergy magnitude during graph construction.
