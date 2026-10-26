# Binary ERWR for two-drug synergy

This adds a Python binary-classification pipeline to the repository. It uses `result` (0/1) from `extract.xlsx`; continuous synergy columns are retained but are not used as labels.

## Input

Required columns:

```text
drug_row, drug_col, cell_line_name, result
```

The optional columns `synergy_zip`, `synergy_bliss`, `synergy_loewe`, `synergy_hsa`, and `S` are copied to the prediction output when present.

## Method

For each outer fold, only positive training records (`result == 1`) become undirected graph edges. Negative records are evaluation samples and are not inserted as negative-weight edges. Pairs absent from the file are unknown, not assumed negative.

For `sample` and `drug_pair` splits, a separate drug graph is built for each test cell line. For `cell_line` split, the test cell line is unseen during training, so the leakage-safe fallback is one global graph built only from positive training pairs.

The outer test fold is never used to choose the classification threshold. An inner validation split is created from the outer training fold, ERWR scores are computed from the inner-training graph, and the threshold maximizing validation F1 is then applied once to the outer test fold. AUROC and AUPRC always use raw ERWR scores and do not depend on the threshold.

`rwrdc` is the pairwise Random Walk with Restart implementation derived from `wangqi27/ERWR/rwrdc.py`. Its output is a normalized network score, not the continuous value in `S`/ZIP/Bliss/Loewe/HSA.

The three splits are:

- `sample`: random record-level stratified folds.
- `drug_pair`: all records for an unordered drug pair stay in one fold.
- `cell_line`: all records for a cell line stay in one fold.

The script reports AUROC, AUPRC, F1, precision, recall, accuracy, MCC, plus per-graph density, sparsity, isolated nodes, connected components, selected threshold, and validation F1.

## Run

```bash
pip install numpy pandas scikit-learn networkx openpyxl
python run_binary_erwr.py --input extract.xlsx --split sample --folds 5
python run_binary_erwr.py --input extract.xlsx --split drug_pair --folds 5
python run_binary_erwr.py --input extract.xlsx --split cell_line --folds 5
```

Outputs are written to `output/binary_erwr/` by default. Existing output files with the same names will be overwritten.
