# Isolate-repaired weighted ERWR

This version combines continuous synergy edge weights with sparse, isolate-only drug similarity repair.

## Design

1. Build the positive training graph using the selected continuous score (`S`, `synergy_hsa`, `synergy_bliss`, `synergy_zip`, or `synergy_loewe`).
2. Detect drugs with zero degree.
3. Add only the top-k similarity neighbors for those isolated drugs.
4. Give repair edges weight `similarity_beta * cosine_similarity`.
5. Never add similarity edges for drugs that already have a positive training neighbor.

This avoids turning dense cell-line graphs such as HT29 into complete similarity graphs while still giving sparse graphs such as LNCAP and LOVO a fallback neighborhood.

## Run with S and min-max weighting

```bash
python run_weighted_erwr.py \
  --input extract.xlsx \
  --score-column S \
  --drug-features drug_feature.csv \
  --split drug_pair \
  --folds 5 \
  --seed 42 \
  --weight-mode minmax \
  --similarity-top-k 2 \
  --similarity-beta 0.05 \
  --threshold-objective balanced_accuracy \
  --output output/weighted_erwr/drug_pair_S_minmax_isolate_repair
```

## Compare repair strengths

```bash
for k in 1 2 3; do
  python run_weighted_erwr.py \
    --input extract.xlsx \
    --score-column S \
    --drug-features drug_feature.csv \
    --split drug_pair \
    --folds 5 \
    --seed 42 \
    --weight-mode minmax \
    --similarity-top-k "$k" \
    --similarity-beta 0.05 \
    --threshold-objective balanced_accuracy \
    --output "output/weighted_erwr/drug_pair_S_minmax_isolate_k${k}"
done
```

The graph report now includes:

```text
isolated_nodes_before_similarity
n_similarity_edges
isolated_nodes
connected_components
```

Use the previous no-repair run as the baseline by omitting `--drug-features` or setting `--similarity-top-k 0`.
