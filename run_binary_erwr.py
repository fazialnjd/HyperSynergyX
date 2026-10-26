"""Run binary cell-line-specific ERWR on extract.xlsx.

Examples:
  python run_binary_erwr.py --input extract.xlsx --split sample --folds 5
  python run_binary_erwr.py --input extract.xlsx --split drug_pair --folds 5
  python run_binary_erwr.py --input extract.xlsx --split cell_line --folds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from binary_erwr import build_graph, load_interactions, make_splits, rwrdc, score_pairs


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    result = {"n": int(len(y_true)), "positive": int(y_true.sum()), "negative": int((y_true == 0).sum())}
    if len(np.unique(y_true)) < 2:
        return result
    pred = (scores >= threshold).astype(int)
    result.update({
        "AUROC": float(roc_auc_score(y_true, scores)),
        "AUPRC": float(average_precision_score(y_true, scores)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "threshold": float(threshold),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", choices=["sample", "drug_pair", "cell_line"], default="sample")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=0.05, help="ERWR walk coefficient from the original code")
    parser.add_argument("--output", default="output/binary_erwr")
    parser.add_argument("--min-density-warning", type=float, default=0.001)
    parser.add_argument("--max-density-warning", type=float, default=0.25)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_interactions(args.input)
    all_drugs = sorted(set(df["drug_a"]) | set(df["drug_b"]))
    splits = make_splits(df, args.split, args.folds, args.seed)
    all_predictions, all_metrics, all_stats = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        train_graphs = {}
        test_lines = sorted(test["cell_line_name"].unique())
        for cell in test_lines:
            # For normal/pair splits, use a cell-line-specific graph. For an unseen
            # cell line in cell_line split, use only a global TRAIN graph.
            if args.split == "cell_line":
                A, index, stats = build_graph(train, all_drugs, cell_line=None)
                stats.cell_line = f"GLOBAL_FOR_{cell}"
            else:
                A, index, stats = build_graph(train, all_drugs, cell_line=cell)
            if stats.density < args.min_density_warning or stats.density > args.max_density_warning:
                print(f"Warning fold={fold} cell_line={stats.cell_line}: density={stats.density:.6f}")
            train_graphs[cell] = (rwrdc(A, c=args.c), index, stats)
            all_stats.append(stats.as_dict() | {"fold": fold, "split": args.split})

        fold_scores = []
        for cell, group in test.groupby("cell_line_name", sort=False):
            score_matrix, index, _ = train_graphs[cell]
            fold_scores.append(pd.DataFrame({
                "row_index": group.index,
                "erwr_score": score_pairs(score_matrix, index, group),
            }))
        scored = pd.concat(fold_scores, ignore_index=True).set_index("row_index")
        test["erwr_score"] = scored.loc[test.index, "erwr_score"].to_numpy()
        test["fold"], test["split"] = fold, args.split
        all_predictions.append(test)
        all_metrics.append(metrics(test["result"].to_numpy(), test["erwr_score"].to_numpy()))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out / f"predictions_{args.split}.csv", index=False)
    pd.DataFrame(all_stats).to_csv(out / f"graph_stats_{args.split}.csv", index=False)
    with open(out / f"metrics_{args.split}_folds.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    numeric = pd.DataFrame(all_metrics).select_dtypes(include="number")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    summary.to_csv(out / f"metrics_{args.split}_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
