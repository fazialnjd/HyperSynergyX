# Feature-enriched ERWR, sparse top-k version

This version avoids turning the 48-drug graph into a complete graph. The original embedding cosine similarities are high (`~0.80-0.97`), so an absolute similarity threshold would connect almost every pair. Instead, the enriched graph adds only the symmetrized union of each drug's top-k similarity neighbors.

The cosine similarity is clipped to `[0, 1]`; it is not shifted by `(cosine + 1) / 2`, because that shift creates artificial similarity around `0.5` for unrelated vectors.

## Modes

- `baseline`: positive training pairs only.
- `enriched`: positive training pairs plus a weak top-k drug-embedding prior.
- `similarity_only`: top-k drug-embedding graph without positive training edges.

Positive training edges have weight `1`. Similarity edges have weight `beta * cosine_similarity`, with default `beta=0.10`, so the prior remains weaker than observed positive edges.

## Run

```bash
pip install numpy pandas scikit-learn networkx openpyxl
```

Recommended first comparison on the unseen-drug-pair split:

```bash
python run_feature_enriched_erwr.py \
  --input extract.xlsx \
  --drug-features drug_feature.csv \
  --cell-features KG_cell_line_feature.xlsx \
  --split drug_pair \
  --mode baseline \
  --folds 5 \
  --seed 42 \
  --beta 0.10 \
  --similarity-top-k 5 \
  --output output/feature_enrichment_v2/drug_pair_baseline

python run_feature_enriched_erwr.py \
  --input extract.xlsx \
  --drug-features drug_feature.csv \
  --cell-features KG_cell_line_feature.xlsx \
  --split drug_pair \
  --mode enriched \
  --folds 5 \
  --seed 42 \
  --beta 0.10 \
  --similarity-top-k 5 \
  --output output/feature_enrichment_v2/drug_pair_topk5_beta010
```

Use the same commands with `--similarity-top-k 3` and `10` for ablation. The threshold for binary classification is selected on an inner validation split; AUROC/AUPRC use continuous scores.
