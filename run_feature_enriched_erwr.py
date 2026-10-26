"""Run top-k feature-enriched binary ERWR with drug and cell-line embeddings."""

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

from binary_erwr import load_interactions, make_splits
from enriched_erwr import score_cell_line_transfer, score_with_graph
from feature_similarity import align_similarity, cosine_similarity_matrix, load_feature_table


def make_inner_validation_split(train_df: pd.DataFrame, split: str, seed: int):
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
    """Select F1 threshold; prefer the higher threshold when F1 ties."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        current = float(f1_score(y_true, scores >= threshold, zero_division=0))
        if current > best_f1 or (np.isclose(current, best_f1) and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), current
    return best_threshold, best_f1


def calculate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "predicted_positive_rate": None,
    }
    if len(np.unique(y_true)) < 2:
        return result
    predicted = (scores >= threshold).astype(int)
    result.update(
        {
            "AUROC": float(roc_auc_score(y_true, scores)),
            "AUPRC": float(average_precision_score(y_true, scores)),
            "F1": float(f1_score(y_true, predicted, zero_division=0)),
            "Precision": float(precision_score(y_true, predicted, zero_division=0)),
            "Recall": float(recall_score(y_true, predicted, zero_division=0)),
            "Accuracy": float(accuracy_score(y_true, predicted)),
            "MCC": float(matthews_corrcoef(y_true, predicted)),
            "predicted_positive_rate": float(predicted.mean()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--drug-features", required=True)
    parser.add_argument("--cell-features", required=True)
    parser.add_argument("--split", choices=["sample", "drug_pair", "cell_line"], default="sample")
    parser.add_argument("--mode", choices=["baseline", "enriched", "similarity_only"], default="enriched")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument("--output", default="output/feature_enrichment")
    args = parser.parse_args()

    if args.similarity_top_k < 1:
        raise ValueError("--similarity-top-k must be at least 1")
    if args.beta < 0:
        raise ValueError("--beta must be non-negative")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = load_interactions(args.input)
    all_drugs = sorted(set(df["drug_a"]) | set(df["drug_b"]))
    drug_index = {drug: i for i, drug in enumerate(all_drugs)}

    drug_names, drug_features = load_feature_table(args.drug_features, id_col="drug_name")
    drug_similarity_raw = cosine_similarity_matrix(drug_features)
    drug_similarity, matched_drugs = align_similarity(all_drugs, drug_names, drug_similarity_raw)
    if matched_drugs != len(all_drugs):
        missing = sorted(set(all_drugs) - set(drug_names))
        raise ValueError(f"Missing drug embeddings for: {missing}")

    cell_names, cell_features = load_feature_table(args.cell_features)
    cell_similarity_raw = cosine_similarity_matrix(cell_features)
    cell_index = {name: i for i, name in enumerate(cell_names)}
    missing_cells = sorted(set(df["cell_line_name"]) - set(cell_names))
    if missing_cells:
        raise ValueError(f"Missing cell-line embeddings for: {missing_cells}")

    pd.DataFrame(drug_similarity, index=all_drugs, columns=all_drugs).to_csv(out / "drug_similarity.csv")
    pd.DataFrame(cell_similarity_raw, index=cell_names, columns=cell_names).to_csv(out / "cell_similarity.csv")

    splits = make_splits(df, args.split, args.folds, args.seed)
    all_predictions, all_metrics, all_stats = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        inner_train, validation = make_inner_validation_split(train, args.split, args.seed + fold)

        if args.split == "cell_line":
            validation_scores, _ = score_cell_line_transfer(
                inner_train, validation, all_drugs, drug_similarity, drug_index,
                cell_names, cell_similarity_raw, cell_index, args.mode, args.beta,
                args.similarity_top_k,
            )
            test_scores, fold_stats = score_cell_line_transfer(
                train, test, all_drugs, drug_similarity, drug_index,
                cell_names, cell_similarity_raw, cell_index, args.mode, args.beta,
                args.similarity_top_k,
            )
        else:
            validation_scores, _ = score_with_graph(
                inner_train, validation, all_drugs, drug_similarity, drug_index,
                args.split, args.mode, args.beta, args.similarity_top_k,
            )
            test_scores, fold_stats = score_with_graph(
                train, test, all_drugs, drug_similarity, drug_index,
                args.split, args.mode, args.beta, args.similarity_top_k,
            )

        threshold, validation_f1 = select_threshold(validation["result"].to_numpy(), validation_scores)
        test["erwr_score"] = test_scores
        test["predicted_result"] = (test_scores >= threshold).astype(int)
        test["threshold"] = threshold
        test["validation_f1"] = validation_f1
        test["fold"] = fold
        test["split"] = args.split
        test["mode"] = args.mode
        all_predictions.append(test)

        for stat in fold_stats:
            stat.update(
                {
                    "fold": fold,
                    "split": args.split,
                    "mode": args.mode,
                    "beta": args.beta,
                    "similarity_top_k": args.similarity_top_k,
                    "threshold": threshold,
                    "validation_f1": validation_f1,
                }
            )
            all_stats.append(stat)

        all_metrics.append(calculate_metrics(test["result"].to_numpy(), test_scores, threshold))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out / f"predictions_{args.split}_{args.mode}.csv", index=False)
    pd.DataFrame(all_stats).to_csv(out / f"graph_stats_{args.split}_{args.mode}.csv", index=False)
    with open(out / f"metrics_{args.split}_{args.mode}_folds.json", "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

    numeric = pd.DataFrame(all_metrics).select_dtypes(include="number")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    summary.to_csv(out / f"metrics_{args.split}_{args.mode}_summary.csv")
    print(summary)
    print(f"Matched drug embeddings: {matched_drugs}/{len(all_drugs)}")
    print(f"Cell-line embeddings: {len(cell_names)}")
    print(f"Drug similarity top-k: {args.similarity_top_k}")
    print(f"Similarity beta: {args.beta}")


if __name__ == "__main__":
    main()
