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
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from binary_erwr import build_graph, load_interactions, make_splits, rwrdc, score_pairs


def score_with_training_graph(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    all_drugs: list[str],
    split: str,
    c: float,
):
    """Score query rows using graphs built only from train_df."""
    if query_df.empty:
        return np.array([], dtype=float), []

    graph_cache = {}
    scores = np.zeros(len(query_df), dtype=float)
    stats_rows = []

    for cell, group in query_df.groupby("cell_line_name", sort=False):
        graph_cell = None if split == "cell_line" else cell
        key = graph_cell or "GLOBAL"
        if key not in graph_cache:
            A, index, stats = build_graph(train_df, all_drugs, cell_line=graph_cell)
            if split == "cell_line":
                stats.cell_line = f"GLOBAL_FOR_{cell}"
            graph_cache[key] = (rwrdc(A, c=c), index, stats)
            stats_rows.append(stats.as_dict())
        matrix, index, _ = graph_cache[key]
        scores[group.index.to_numpy()] = score_pairs(matrix, index, group)

    return scores, stats_rows


def make_inner_validation_split(train_df: pd.DataFrame, split: str, seed: int):
    """Make an inner validation split for threshold selection without test leakage."""
    if len(train_df) < 4 or train_df["result"].nunique() < 2:
        return train_df.copy(), train_df.iloc[0:0].copy()

    if split == "sample":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner_idx, val_idx = next(splitter.split(train_df, train_df["result"].values))
    else:
        groups = (
            train_df["drug_a"] + "||" + train_df["drug_b"]
            if split == "drug_pair"
            else train_df["cell_line_name"]
        )
        if groups.nunique() < 2:
            return train_df.copy(), train_df.iloc[0:0].copy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner_idx, val_idx = next(splitter.split(train_df, train_df["result"].values, groups))

    return train_df.iloc[inner_idx].copy(), train_df.iloc[val_idx].copy()


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Select the threshold maximizing validation F1."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        current = f1_score(y_true, pred, zero_division=0)
        if current > best_f1:
            best_f1 = float(current)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) < 2:
        return result
    pred = (scores >= threshold).astype(int)
    result.update(
        {
            "AUROC": float(roc_auc_score(y_true, scores)),
            "AUPRC": float(average_precision_score(y_true, scores)),
            "F1": float(f1_score(y_true, pred, zero_division=0)),
            "Precision": float(precision_score(y_true, pred, zero_division=0)),
            "Recall": float(recall_score(y_true, pred, zero_division=0)),
            "Accuracy": float(accuracy_score(y_true, pred)),
            "MCC": float(matthews_corrcoef(y_true, pred)),
        }
    )
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
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        # Select a threshold using an inner validation set. The outer test fold
        # is not used for threshold selection.
        inner_train, validation = make_inner_validation_split(train, args.split, args.seed + fold)
        validation_scores, _ = score_with_training_graph(
            inner_train, validation, all_drugs, args.split, args.c
        )
        threshold, validation_f1 = select_threshold(
            validation["result"].to_numpy(), validation_scores
        )

        test_scores, fold_stats = score_with_training_graph(
            train, test, all_drugs, args.split, args.c
        )
        test["erwr_score"] = test_scores
        test["predicted_result"] = (test_scores >= threshold).astype(int)
        test["threshold"] = threshold
        test["fold"], test["split"] = fold, args.split

        for stats in fold_stats:
            stats.update(
                {
                    "fold": fold,
                    "split": args.split,
                    "threshold": threshold,
                    "validation_f1": validation_f1,
                }
            )
            if stats["density"] < args.min_density_warning or stats["density"] > args.max_density_warning:
                print(
                    f"Warning fold={fold} cell_line={stats['cell_line']}: "
                    f"density={stats['density']:.6f}"
                )
            all_stats.append(stats)

        all_predictions.append(test)
        all_metrics.append(metrics(test["result"].to_numpy(), test_scores, threshold))

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
