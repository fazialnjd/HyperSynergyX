# create_split_separate.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from pathlib import Path


def create_breast_split(output_path="split_breast.json", n_splits=5):
    """
    Create stratified split for breast cancer triplets
    """
    print("=" * 70)
    print(f"CREATING {n_splits}-FOLD SPLIT FOR BREAST CANCER")
    print("=" * 70)

    items = []

    # Load breast cancer incidence matrix
    incidence_path = Path("hypergraph_output/breast_incidence_matrix.tsv")
    if not incidence_path.exists():
        print(f"❌ File not found: {incidence_path}")
        return

    incidence = pd.read_csv(incidence_path, sep="\t", index_col=0)
    drug_names = incidence.index.tolist()

    # Extract positive triplets
    positive_triplets = []
    for col in incidence.columns:
        drugs_in_combo = incidence.index[incidence[col] == 1].tolist()
        if len(drugs_in_combo) == 3:
            positive_triplets.append(tuple(sorted(drugs_in_combo)))

    positive_triplets = list(set(positive_triplets))
    print(f"📁 Found {len(positive_triplets)} positive triplets")

    # Create positive samples
    for d1, d2, d3 in positive_triplets:
        items.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 1, "dataset": "breast"}
        )

    # Create negative samples (random triplets not in positive set)
    n_drugs = len(drug_names)
    all_possible = set()
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            for k in range(j + 1, n_drugs):
                all_possible.add(
                    tuple(sorted([drug_names[i], drug_names[j], drug_names[k]]))
                )

    negative_set = all_possible - set(positive_triplets)
    negative_list = list(negative_set)

    n_positive = len(positive_triplets)
    np.random.seed(42)
    n_negative = min(n_positive, len(negative_list))
    selected_negatives = np.random.choice(
        len(negative_list), size=n_negative, replace=False
    )

    for idx in selected_negatives:
        d1, d2, d3 = negative_list[idx]
        items.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 0, "dataset": "breast"}
        )

    print(
        f"📊 Total: {len(items)} items ({n_positive} positive, {n_negative} negative)"
    )

    # Stratified split
    labels = [item["label"] for item in items]

    # Use StratifiedKFold only if n_splits <= min class count
    min_class_count = min(
        sum(1 for l in labels if l == 0), sum(1 for l in labels if l == 1)
    )

    if n_splits > min_class_count:
        print(f"⚠️ Warning: n_splits={n_splits} > min_class_count={min_class_count}")
        print(f"   Using n_splits={min_class_count} instead")
        n_splits = min_class_count

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(items, labels)):
        for idx in test_idx:
            items[idx]["fold"] = fold

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\n✅ Split file saved to: {output_path}")
    print(f"\n📊 Fold distribution ({n_splits} folds):")
    for fold in range(n_splits):
        fold_items = [i for i in items if i.get("fold") == fold]
        pos = sum(1 for i in fold_items if i["label"] == 1)
        neg = sum(1 for i in fold_items if i["label"] == 0)
        print(
            f"   Fold {fold}: {len(fold_items)} items ({pos} positive, {neg} negative)"
        )


def create_lung_split(output_path="split_lung.json"):
    """
    Create stratified split for lung cancer triplets
    Since there are only 4 positive samples, use 2-fold or 4-fold
    """
    print("=" * 70)
    print("CREATING STRATIFIED SPLIT FOR LUNG CANCER")
    print("=" * 70)

    items = []

    # Load lung cancer incidence matrix
    incidence_path = Path("hypergraph_output/lung_incidence_matrix.tsv")
    if not incidence_path.exists():
        print(f"❌ File not found: {incidence_path}")
        return

    incidence = pd.read_csv(incidence_path, sep="\t", index_col=0)
    drug_names = incidence.index.tolist()

    # Extract positive triplets
    positive_triplets = []
    for col in incidence.columns:
        drugs_in_combo = incidence.index[incidence[col] == 1].tolist()
        if len(drugs_in_combo) == 3:
            positive_triplets.append(tuple(sorted(drugs_in_combo)))

    positive_triplets = list(set(positive_triplets))
    n_positive = len(positive_triplets)
    print(f"📁 Found {n_positive} positive triplets")

    # Create positive samples
    for d1, d2, d3 in positive_triplets:
        items.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 1, "dataset": "lung"}
        )

    # Create negative samples
    n_drugs = len(drug_names)
    all_possible = set()
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            for k in range(j + 1, n_drugs):
                all_possible.add(
                    tuple(sorted([drug_names[i], drug_names[j], drug_names[k]]))
                )

    negative_set = all_possible - set(positive_triplets)
    negative_list = list(negative_set)

    # Take exactly n_positive negative samples (balanced dataset)
    np.random.seed(42)
    n_negative = min(n_positive, len(negative_list))
    selected_negatives = np.random.choice(
        len(negative_list), size=n_negative, replace=False
    )

    for idx in selected_negatives:
        d1, d2, d3 = negative_list[idx]
        items.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 0, "dataset": "lung"}
        )

    print(
        f"📊 Total: {len(items)} items ({n_positive} positive, {n_negative} negative)"
    )

    # Determine appropriate number of folds
    # For small datasets, use Leave-One-Out or 2-fold
    labels = [item["label"] for item in items]
    min_class_count = min(
        sum(1 for l in labels if l == 0), sum(1 for l in labels if l == 1)
    )

    # Use 2-fold for lung cancer (since only 4 positive samples)
    n_splits = 2 if min_class_count >= 2 else 1

    print(f"   Using n_splits={n_splits} (min_class_count={min_class_count})")

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(items, labels)):
            for idx in test_idx:
                items[idx]["fold"] = fold
    else:
        # If only 1 sample per class, use single fold
        for idx in range(len(items)):
            items[idx]["fold"] = 0

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"\n✅ Split file saved to: {output_path}")
    print(f"\n📊 Fold distribution ({n_splits} folds):")
    for fold in range(n_splits):
        fold_items = [i for i in items if i.get("fold") == fold]
        pos = sum(1 for i in fold_items if i["label"] == 1)
        neg = sum(1 for i in fold_items if i["label"] == 0)
        print(
            f"   Fold {fold}: {len(fold_items)} items ({pos} positive, {neg} negative)"
        )


if __name__ == "__main__":
    # Create separate split files
    create_breast_split("split_breast.json", n_splits=5)
    print("\n" + "=" * 70 + "\n")
    create_lung_split("split_lung.json")
