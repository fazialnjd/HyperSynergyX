# Threshold-objective extension

The weighted ERWR runner now supports four leakage-safe threshold objectives. The objective is selected only on the inner validation split:

```text
f1                 maximize F1
balanced_accuracy  maximize balanced accuracy
mcc                maximize Matthews correlation coefficient
youden             maximize sensitivity + specificity - 1
```

The ranking metrics (`AUROC`, `AUPRC`) do not depend on the selected threshold. The threshold-sensitive metrics (`F1`, `Specificity`, `BalancedAccuracy`, `MCC`, and predicted positive rate) do.

## Recommended runs

Use `S` with `minmax`, because it was the best configuration in the previous experiment for AUROC/AUPRC:

```bash
for objective in f1 balanced_accuracy mcc youden; do
  python run_weighted_erwr.py \
    --input extract.xlsx \
    --score-column S \
    --split drug_pair \
    --folds 5 \
    --seed 42 \
    --weight-mode minmax \
    --threshold-objective "$objective" \
    --output "output/weighted_erwr/drug_pair_S_minmax_${objective}"
done
```

The most important comparison columns are:

```text
AUROC
AUPRC
Specificity
BalancedAccuracy
MCC
predicted_positive_rate
```

`f1` reproduces the previous behavior. `balanced_accuracy`, `mcc`, and `youden` are intended to reduce the strong positive prediction bias observed in the earlier runs.
