"""Run coincidence/transition-enriched ERWR without direct similarity edges."""

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
from coincidence_transition import (
    align_feature_incidence,
    score_with_coincidence_graph,
)
from feature_similarity import load_feature_table


def make_inner_validation_split(train_df, split, seed):
    if len(train_df) < 4 or train_df["result"].nunique() < 2:
        return train_df.copy(), train_df.iloc[0:0].copy()
    if split == "sample":
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed
        )
        inner, validation = next(
            splitter.split(train_df, train_df["result"].values)
        )
    else:
        groups = (
            train_df["drug_a"] + "||" + train_df["drug_b"]
            if split == "drug_pair"
            else train_df["cell_line_name"]
        )
        if groups.nunique() < 2:
            return train_df.copy(), train_df.iloc[0:0].copy()
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed
        )
        inner, validation = next(
            splitter.split(train_df, train_df["result"].values, groups)
        )
    return train_df.iloc[inner].copy(), train_df.iloc[validation].copy()


def select_threshold(y_true, scores):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        value = float(f1_score(y_true, scores >= threshold, zero_division=0))
        if value > best_f1 or (
            np.isclose(value, best_f1) and threshold > best_threshold
        ):
            best_threshold, best_f1 = float(threshold), value
    return best_threshold, best_f1


def calculate_metrics(y_true, scores, threshold):
    result = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) < 2:
        return result
    predicted = (scores >= threshold).astype(int)
    result.update(
        {
            "predicted_positive_rate": float(predicted.mean()),
            "AUROC": float(roc_auc_score(y_true, scores)),
            "AUPRC": float(average_precision_score(y_true, scores)),
            "F1": float(f1_score(y_true, predicted, zero_division=0)),
            "Precision": float(precision_score(y_true, predicted, zero_division=0)),
            "Recall": float(recall_score(y_true, predicted, zero_division=0)),
            "Accuracy": float(accuracy_score(y_true, predicted)),
            "MCC": float(matthews_corrcoef(y_true, predicted)),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--kg-drug-features", default=None)
    parser.add_argument("--split", choices=["sample", "drug_pair", "cell_line"], default="drug_pair")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--propagation-lambda", type=float, default=0.10)
    parser.add_argument("--kg-alpha", type=float, default=0.50)
    parser.add_argument("--output", default="output/coincidence_transition")
    args = parser.parse_args()

    if args.beta < 0 or not 0 <= args.propagation_lambda <= 1:
        raise ValueError("beta must be non-negative and lambda must be in [0, 1]")
    if not 0 <= args.kg_alpha <= 1:
        raise ValueError("kg-alpha must be in [0, 1]")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df = load_interactions(args.input)
    all_drugs = sorted(set(df["drug_a"]) | set(df["drug_b"]))
    drug_index = {drug: i for i, drug in enumerate(all_drugs)}

    kg_incidence = None
    if args.kg_drug_features:
        names, features = load_feature_table(args.kg_drug_features)
        kg_incidence = align_feature_incidence(all_drugs, names, features)
        pd.DataFrame(
            kg_incidence,
            index=all_drugs,
        ).to_csv(output / "kg_drug_incidence_aligned.csv")

    splits = make_splits(df, args.split, args.folds, args.seed)
    all_predictions, all_metrics, all_stats = [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        inner_train, validation = make_inner_validation_split(
            train, args.split, args.seed + fold
        )

        validation_scores, _ = score_with_coincidence_graph(
            inner_train, validation, all_drugs, drug_index, args.split,
            args.beta, args.propagation_lambda, kg_incidence, args.kg_alpha
        )
        test_scores, fold_stats = score_with_coincidence_graph(
            train, test, all_drugs, drug_index, args.split,
            args.beta, args.propagation_lambda, kg_incidence, args.kg_alpha
        )

        threshold, validation_f1 = select_threshold(
            validation["result"].to_numpy(), validation_scores
        )
        test["erwr_score"] = test_scores
        test["predicted_result"] = (test_scores >= threshold).astype(int)
        test["threshold"] = threshold
        test["validation_f1"] = validation_f1
        test["fold"] = fold
        test["split"] = args.split
        test["mode"] = "coincidence_reweighted"
        all_predictions.append(test)

        for stat in fold_stats:
            stat.update({
                "fold": fold,
                "split": args.split,
                "beta": args.beta,
                "propagation_lambda": args.propagation_lambda,
                "kg_alpha": args.kg_alpha,
                "has_kg_features": bool(kg_incidence is not None),
                "threshold": threshold,
                "validation_f1": validation_f1,
            })
            all_stats.append(stat)

        all_metrics.append(calculate_metrics(
            test["result"].to_numpy(), test_scores, threshold
        ))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / f"predictions_{args.split}_coincidence.csv", index=False)
    pd.DataFrame(all_stats).to_csv(output / f"graph_stats_{args.split}_coincidence.csv", index=False)
    with open(output / f"metrics_{args.split}_coincidence_folds.json", "w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)
    numeric = pd.DataFrame(all_metrics).select_dtypes(include="number")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    summary.to_csv(output / f"metrics_{args.split}_coincidence_summary.csv")

    print(summary)
    print(f"Drugs: {len(all_drugs)}")
    print(f"KG incidence: {kg_incidence is not None}")
    print(f"beta: {args.beta}")
    print(f"propagation_lambda: {args.propagation_lambda}")
    print(f"kg_alpha: {args.kg_alpha}")


if __name__ == "__main__":
    main()
