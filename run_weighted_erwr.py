"""Run leakage-safe weighted ERWR using continuous synergy scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from binary_erwr import load_interactions, make_splits
from weighted_erwr import score_weighted_graph



def make_inner_validation_split(train_df, split, seed):
    if len(train_df) < 4 or train_df["result"].nunique() < 2:
        return train_df.copy(), train_df.iloc[0:0].copy()
    if split == "sample":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner, validation = next(splitter.split(train_df, train_df["result"].values))
    else:
        groups = (
            train_df["drug_a"] + "||" + train_df["drug_b"]
            if split == "drug_pair"
            else train_df["cell_line_name"]
        )
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner, validation = next(splitter.split(train_df, train_df["result"].values, groups))
    return train_df.iloc[inner].copy(), train_df.iloc[validation].copy()


def select_threshold(y_true, scores):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        current = float(f1_score(y_true, scores >= threshold, zero_division=0))
        if current > best_f1 or (np.isclose(current, best_f1) and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), current
    return best_threshold, best_f1


def calculate_metrics(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    result = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "predicted_positive_rate": float(predicted.mean()) if len(predicted) else 0.0,
    }
    if len(np.unique(y_true)) < 2:
        return result
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    fp = int(((y_true == 0) & (predicted == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    result.update({
        "AUROC": float(roc_auc_score(y_true, scores)),
        "AUPRC": float(average_precision_score(y_true, scores)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "Specificity": float(specificity),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, predicted)),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "MCC": float(matthews_corrcoef(y_true, predicted)),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--score-column", required=True, help="Continuous score column, e.g. HSA, S, Bliss, ZIP, or Loewe")
    parser.add_argument("--split", choices=["sample", "drug_pair", "cell_line"], default="drug_pair")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-mode", choices=["rank", "minmax", "raw"], default="rank")
    parser.add_argument("--output", default="output/weighted_erwr")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(args.input) if args.input.lower().endswith((".xlsx", ".xls")) else pd.read_csv(args.input)
    if args.score_column not in raw.columns:
        raise ValueError(f"Score column {args.score_column!r} not found. Available columns: {list(raw.columns)}")
    if "result" not in raw.columns:
        raise ValueError("The input must contain the binary result column")

    df = load_interactions(args.input)
    # load_interactions preserves extra continuous columns while adding drug_a/drug_b.
    df[args.score_column] = pd.to_numeric(raw[args.score_column], errors="coerce").to_numpy()
    if df[args.score_column].isna().all():
        raise ValueError(f"Score column {args.score_column!r} contains no numeric values")

    all_drugs = sorted(set(df["drug_a"]) | set(df["drug_b"]))
    drug_index = {drug: i for i, drug in enumerate(all_drugs)}
    splits = make_splits(df, args.split, args.folds, args.seed)
    all_predictions, all_metrics, all_stats = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        inner_train, validation = make_inner_validation_split(train, args.split, args.seed + fold)
        validation_scores, _ = score_weighted_graph(
            inner_train, validation, all_drugs, drug_index, args.split,
            args.score_column, args.weight_mode,
        )
        test_scores, fold_stats = score_weighted_graph(
            train, test, all_drugs, drug_index, args.split,
            args.score_column, args.weight_mode,
        )
        threshold, validation_f1 = select_threshold(validation["result"].to_numpy(), validation_scores)
        test["weighted_erwr_score"] = test_scores
        test["predicted_result"] = (test_scores >= threshold).astype(int)
        test["threshold"] = threshold
        test["validation_f1"] = validation_f1
        test["fold"] = fold
        test["split"] = args.split
        test["score_column"] = args.score_column
        test["weight_mode"] = args.weight_mode
        all_predictions.append(test)
        for stat in fold_stats:
            stat.update({
                "fold": fold,
                "split": args.split,
                "score_column": args.score_column,
                "weight_mode": args.weight_mode,
                "threshold": threshold,
                "validation_f1": validation_f1,
            })
            all_stats.append(stat)
        all_metrics.append(calculate_metrics(test["result"].to_numpy(), test_scores, threshold))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / f"predictions_{args.split}_{args.score_column}_weighted_erwr.csv", index=False)
    pd.DataFrame(all_stats).to_csv(output / f"graph_stats_{args.split}_{args.score_column}_weighted_erwr.csv", index=False)
    with open(output / f"metrics_{args.split}_{args.score_column}_weighted_erwr_folds.json", "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    numeric = pd.DataFrame(all_metrics).select_dtypes(include="number")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    summary.to_csv(output / f"metrics_{args.split}_{args.score_column}_weighted_erwr_summary.csv")
    print(summary)
    print(f"Score column: {args.score_column}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Split: {args.split}")


if __name__ == "__main__":
    main()
