# Isolate-aware feature-enriched ERWR

This version uses drug embeddings only to repair isolated drug nodes in each positive training graph.

For each graph:

1. Build the positive training graph exactly as in the baseline.
2. Find drugs with zero degree in that graph.
3. For each isolated drug, add only its top-k embedding-similarity neighbors.
4. Give observed positive edges weight `1` and similarity fallback edges weight `beta * similarity`.

The recommended first configuration is:

```text
mode = isolate_enriched
similarity_top_k = 2
beta = 0.05
```

## Colab commands

```bash
pip install numpy pandas scikit-learn networkx openpyxl
```

Baseline:

```bash
python run_feature_enriched_erwr.py \
  --input extract.xlsx \
  --drug-features drug_feature.csv \
  --cell-features KG_cell_line_feature.xlsx \
  --split drug_pair \
  --mode baseline \
  --folds 5 \
  --seed 42 \
  --beta 0.05 \
  --similarity-top-k 2 \
  --output output/feature_enrichment_isolate/drug_pair_baseline
```

Isolate-aware enrichment:

```bash
python run_feature_enriched_erwr.py \
  --input extract.xlsx \
  --drug-features drug_feature.csv \
  --cell-features KG_cell_line_feature.xlsx \
  --split drug_pair \
  --mode isolate_enriched \
  --folds 5 \
  --seed 42 \
  --beta 0.05 \
  --similarity-top-k 2 \
  --output output/feature_enrichment_isolate/drug_pair_isolate_topk2_beta005
```

The graph report includes:

- `n_isolated_before_similarity`
- `n_similarity_edges`
- final `isolated_nodes`
- final `connected_components`
- `similarity_scope`

The main comparison remains AUROC, AUPRC, and MCC. A reduction in isolated nodes is useful only if ranking metrics do not deteriorate.
