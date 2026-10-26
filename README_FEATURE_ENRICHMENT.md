# Feature-enriched ERWR

This branch adds an embedding-enriched binary ERWR pipeline using:

- `drug_feature.csv`: 384-dimensional precomputed drug embeddings.
- `KG_cell_line_feature.xlsx`: 128-dimensional precomputed cell-line embeddings.
- `extract.xlsx`: binary drug-pair/cell-line observations.

The drug embedding is converted to a cosine-similarity prior. In `enriched` mode, this prior is added as a weaker weighted edge to the positive training graph. In `cell_line` split, unseen test cell lines are scored by a similarity-weighted average of graphs built for training cell lines using the cell-line embeddings.

The precomputed embeddings are treated as external features. Their provenance should be documented before making a strict leakage-free claim, especially if drug-cell effectiveness edges used to create the cell-line embedding came from the same synergy dataset.

## Run

Install dependencies:

```bash
pip install numpy pandas scikit-learn networkx openpyxl
```

Run the enriched model:

```bash
python run_feature_enriched_erwr.py \
  --input extract.xlsx \
  --drug-features drug_feature.csv \
  --cell-features KG_cell_line_feature.xlsx \
  --split drug_pair \
  --mode enriched \
  --folds 5 \
  --beta 0.25 \
  --similarity-threshold 0.20 \
  --output output/feature_enrichment/drug_pair_enriched
```

Available modes:

- `baseline`: positive training graph only.
- `enriched`: positive graph plus drug embedding similarity.
- `similarity_only`: drug embedding similarity without positive training edges.

The threshold is selected on an inner validation split for every outer fold. AUROC and AUPRC use continuous ERWR scores; threshold-based metrics use the selected validation threshold.
