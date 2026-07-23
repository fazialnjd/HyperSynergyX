import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def create_split_json(output_path="split_breast_lung.json"):
    """
    Create 5-fold stratified split for breast and lung cancer triplets
    """

    # ================================================================
    # 1. Load breast cancer triplets from synergy_top_20.csv or from incidence matrix
    # ================================================================

    # روش اول: از فایل synergy_top_20.csv که از قبل دارید
    breast_triplets = []

    # بارگذاری ترکیبات برتر سرطان پستان
    breast_scores = pd.read_csv("output/breast_synergy/synergy_triplets_ranked.csv")

    # ترکیبات مثبت (موجود در دیتاست) - تمام ترکیباتی که در دیتاست اصلی بودند
    # برای این کار می‌توانید از incidence matrix استفاده کنید
    incidence_breast = pd.read_csv(
        "hypergraph_output/breast_incidence_matrix.tsv", sep="\t", index_col=0
    )
    combo_cols = incidence_breast.columns

    # استخراج ترکیبات از نام ستون‌ها (فرض کنیم ستون‌ها فرمت drug1|drug2|drug3 دارند)
    positive_triplets = set()
    for combo in combo_cols:
        parts = combo.split("|") if "|" in combo else combo.split("_")
        if len(parts) >= 3:
            positive_triplets.add(tuple(sorted(parts[:3])))

    # ایجاد لیست نمونه‌های مثبت
    for d1, d2, d3 in positive_triplets:
        breast_triplets.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 1, "dataset": "breast"}
        )

    # ================================================================
    # 2. Create negative samples (random triplets not in positive set)
    # ================================================================

    # لیست تمام داروهای سرطان پستان
    breast_drugs = incidence_breast.index.tolist()

    # ایجاد تمام ترکیبات ممکن (حدود 2925 ترکیب)
    all_possible_triplets = set()
    n_drugs = len(breast_drugs)
    for i in range(n_drugs):
        for j in range(i + 1, n_drugs):
            for k in range(j + 1, n_drugs):
                all_possible_triplets.add(
                    tuple(sorted([breast_drugs[i], breast_drugs[j], breast_drugs[k]]))
                )

    # نمونه‌های منفی = ترکیبات ممکن - ترکیبات مثبت
    negative_triplets = all_possible_triplets - positive_triplets

    # نمونه‌برداری به تعداد مساوی نمونه‌های مثبت
    n_positive = len(positive_triplets)
    negative_samples = list(negative_triplets)
    np.random.seed(42)
    selected_negatives = np.random.choice(
        len(negative_samples),
        size=min(n_positive, len(negative_samples)),
        replace=False,
    )

    for idx in selected_negatives:
        d1, d2, d3 = negative_samples[idx]
        breast_triplets.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 0, "dataset": "breast"}
        )

    # ================================================================
    # 3. Same for lung cancer
    # ================================================================

    lung_triplets = []

    incidence_lung = pd.read_csv(
        "hypergraph_output/lung_incidence_matrix.tsv", sep="\t", index_col=0
    )
    combo_cols_lung = incidence_lung.columns
    lung_drugs = incidence_lung.index.tolist()

    # ترکیبات مثبت ریه
    positive_lung = set()
    for combo in combo_cols_lung:
        parts = combo.split("|") if "|" in combo else combo.split("_")
        if len(parts) >= 3:
            positive_lung.add(tuple(sorted(parts[:3])))

    for d1, d2, d3 in positive_lung:
        lung_triplets.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 1, "dataset": "lung"}
        )

    # ترکیبات منفی ریه
    all_possible_lung = set()
    n_lung = len(lung_drugs)
    for i in range(n_lung):
        for j in range(i + 1, n_lung):
            for k in range(j + 1, n_lung):
                all_possible_lung.add(
                    tuple(sorted([lung_drugs[i], lung_drugs[j], lung_drugs[k]]))
                )

    negative_lung = all_possible_lung - positive_lung
    n_positive_lung = len(positive_lung)
    negative_samples_lung = list(negative_lung)
    np.random.seed(42)
    selected_negatives_lung = np.random.choice(
        len(negative_samples_lung),
        size=min(n_positive_lung, len(negative_samples_lung)),
        replace=False,
    )

    for idx in selected_negatives_lung:
        d1, d2, d3 = negative_samples_lung[idx]
        lung_triplets.append(
            {"drugA": d1, "drugB": d2, "drugC": d3, "label": 0, "dataset": "lung"}
        )

    # ================================================================
    # 4. Combine all triplets
    # ================================================================

    all_items = breast_triplets + lung_triplets

    # ================================================================
    # 5. 5-Fold stratified split
    # ================================================================

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [item["label"] for item in all_items]

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_items, labels)):
        for idx in test_idx:
            all_items[idx]["fold"] = fold

    # ================================================================
    # 6. Save to JSON
    # ================================================================

    with open(output_path, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"✅ Created split file: {output_path}")
    print(f"   Total items: {len(all_items)}")
    print(f"   Positive: {sum(1 for i in all_items if i['label']==1)}")
    print(f"   Negative: {sum(1 for i in all_items if i['label']==0)}")
    print(f"   Breast: {sum(1 for i in all_items if i['dataset']=='breast')}")
    print(f"   Lung: {sum(1 for i in all_items if i['dataset']=='lung')}")


if __name__ == "__main__":
    # create_split_json("split_breast_lung.json")
    print("This script is not meant to be run directly.")
    print("Use evaluate_simple.py instead.")
