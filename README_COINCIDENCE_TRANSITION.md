# Coincidence-transition ERWR branch

This branch replaces direct drug-embedding similarity edges with structural coincidence and transition propagation.

For each training fold it builds:

```text
H_train              positive-pair incidence matrix
C = H De^-1 H.T     coincidence matrix
T = row_normalize(C) transition matrix
C_prop = (1-lambda)C + lambda*T*C*T.T
```

The positive ERWR adjacency is then reweighted without adding new edges:

```text
A_final = A_positive * (1 + beta * normalize(C_prop))
```

This keeps observed positive synergy edges as the graph structure and uses coincidence only to modify their weights. Optional `KG_drug_feature` data is converted into a nonnegative drug-feature incidence matrix and combined with the structural coincidence matrix.

## Run without KG features

```bash
python run_coincidence_transition_erwr.py \
  --input extract.xlsx \
  --split drug_pair \
  --folds 5 \
  --seed 42 \
  --beta 0.05 \
  --propagation-lambda 0.10 \
  --output output/coincidence_transition/drug_pair_structural
```

## Run with KG drug features

```bash
python run_coincidence_transition_erwr.py \
  --input extract.xlsx \
  --kg-drug-features KG_drug_feature.xlsx \
  --split drug_pair \
  --folds 5 \
  --seed 42 \
  --beta 0.05 \
  --propagation-lambda 0.10 \
  --kg-alpha 0.50 \
  --output output/coincidence_transition/drug_pair_structural_kg
```

The KG file must contain one drug identifier column and numeric feature columns. The identifier column is inferred as the first non-numeric column unless the file format requires manual preprocessing.

## Output files

```text
metrics_<split>_coincidence_summary.csv
graph_stats_<split>_coincidence.csv
predictions_<split>_coincidence.csv
metrics_<split>_coincidence_folds.json
kg_drug_incidence_aligned.csv  # only when KG features are supplied
```

Use AUROC, AUPRC, and MCC for the main comparison. The graph statistics also report hyperedge count, coincidence edges, density, isolated nodes, and connected components.
