import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


def create_split_json(output_path="split_breast_lung.json"):
    """
    Create 5-fold stratified split for breast and lung cancer triplets
    """

    print("=" * 70)
    print("CREATING 5-FOLD SPLIT FOR EVALUATION")
    print("=" * 70)

    # ================================================================
    # 1. Load incidence matrices and extract triplets
    # ================================================================

    all_items = []

    # ==================== BREAST CANCER ====================
    print("\n📁 Processing Breast Cancer data...")

    incidence_breast_path = Path("hypergraph_output/breast_incidence_matrix.tsv")
    if incidence_breast_path.exists():
        incidence_breast = pd.read_csv(incidence_breast_path, sep="\t", index_col=0)
        drug_names = incidence_breast.index.tolist()

        # استخراج ترکیبات از نام ستون‌ها
        # ستون‌ها معمولاً به صورت "drug1|drug2|drug3" یا "drug1_drug2_drug3" هستند
        positive_triplets = []

        for col in incidence_breast.columns:
            # پیدا کردن داروهایی که در این ستون مقدار 1 دارند
            drugs_in_combo = incidence_breast.index[incidence_breast[col] == 1].tolist()
            if len(drugs_in_combo) == 3:
                positive_triplets.append(tuple(sorted(drugs_in_combo)))
            else:
                # اگر ستون به صورت نام ترکیب است
                if "|" in col:
                    parts = col.split("|")
                elif "_" in col:
                    parts = col.split("_")
                else:
                    parts = col.split()

                if len(parts) >= 3:
                    positive_triplets.append(tuple(sorted(parts[:3])))

        positive_triplets = list(set(positive_triplets))  # حذف تکراری‌ها
        print(f"   Found {len(positive_triplets)} positive triplets")

        # ایجاد نمونه‌های مثبت
        for d1, d2, d3 in positive_triplets:
            all_items.append(
                {"drugA": d1, "drugB": d2, "drugC": d3, "label": 1, "dataset": "breast"}
            )

        # ==================== ایجاد نمونه‌های منفی ====================
        # اگر تعداد داروها کم است، همه ترکیبات ممکن را بسازید
        n_drugs = len(drug_names)
        all_possible = set()
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                for k in range(j + 1, n_drugs):
                    all_possible.add(
                        tuple(sorted([drug_names[i], drug_names[j], drug_names[k]]))
                    )

        negative_set = all_possible - set(positive_triplets)

        # نمونه‌برداری به تعداد مساوی نمونه‌های مثبت
        n_positive = len(positive_triplets)
        negative_list = list(negative_set)

        if len(negative_list) > 0:
            np.random.seed(42)
            n_negative = min(n_positive, len(negative_list))
            selected_negatives = np.random.choice(
                len(negative_list), size=n_negative, replace=False
            )

            for idx in selected_negatives:
                d1, d2, d3 = negative_list[idx]
                all_items.append(
                    {
                        "drugA": d1,
                        "drugB": d2,
                        "drugC": d3,
                        "label": 0,
                        "dataset": "breast",
                    }
                )

        print(f"   Breast: {len(positive_triplets)} positive, {n_negative} negative")

    else:
        print(f"   ❌ File not found: {incidence_breast_path}")

    # ==================== LUNG CANCER ====================
    print("\n📁 Processing Lung Cancer data...")

    incidence_lung_path = Path("hypergraph_output/lung_incidence_matrix.tsv")
    if incidence_lung_path.exists():
        incidence_lung = pd.read_csv(incidence_lung_path, sep="\t", index_col=0)
        drug_names_lung = incidence_lung.index.tolist()

        positive_lung = []
        for col in incidence_lung.columns:
            drugs_in_combo = incidence_lung.index[incidence_lung[col] == 1].tolist()
            if len(drugs_in_combo) == 3:
                positive_lung.append(tuple(sorted(drugs_in_combo)))
            else:
                if "|" in col:
                    parts = col.split("|")
                elif "_" in col:
                    parts = col.split("_")
                else:
                    parts = col.split()
                if len(parts) >= 3:
                    positive_lung.append(tuple(sorted(parts[:3])))

        positive_lung = list(set(positive_lung))
        print(f"   Found {len(positive_lung)} positive triplets")

        for d1, d2, d3 in positive_lung:
            all_items.append(
                {"drugA": d1, "drugB": d2, "drugC": d3, "label": 1, "dataset": "lung"}
            )

        # نمونه‌های منفی برای ریه
        n_lung = len(drug_names_lung)
        all_possible_lung = set()
        for i in range(n_lung):
            for j in range(i + 1, n_lung):
                for k in range(j + 1, n_lung):
                    all_possible_lung.add(
                        tuple(
                            sorted(
                                [
                                    drug_names_lung[i],
                                    drug_names_lung[j],
                                    drug_names_lung[k],
                                ]
                            )
                        )
                    )

        negative_lung_set = all_possible_lung - set(positive_lung)
        negative_lung_list = list(negative_lung_set)

        n_positive_lung = len(positive_lung)
        if len(negative_lung_list) > 0:
            np.random.seed(42)
            n_negative_lung = min(n_positive_lung, len(negative_lung_list))
            selected_negatives_lung = np.random.choice(
                len(negative_lung_list), size=n_negative_lung, replace=False
            )

            for idx in selected_negatives_lung:
                d1, d2, d3 = negative_lung_list[idx]
                all_items.append(
                    {
                        "drugA": d1,
                        "drugB": d2,
                        "drugC": d3,
                        "label": 0,
                        "dataset": "lung",
                    }
                )

        print(f"   Lung: {len(positive_lung)} positive, {n_negative_lung} negative")

    else:
        print(f"   ❌ File not found: {incidence_lung_path}")

    # ================================================================
    # 2. Check if we have data
    # ================================================================

    if len(all_items) == 0:
        print("\n❌ ERROR: No items found! Please check the incidence matrix files.")
        return

    print(f"\n📊 Total items: {len(all_items)}")
    print(f"   Positive: {sum(1 for i in all_items if i['label']==1)}")
    print(f"   Negative: {sum(1 for i in all_items if i['label']==0)}")
    print(f"   Breast: {sum(1 for i in all_items if i['dataset']=='breast')}")
    print(f"   Lung: {sum(1 for i in all_items if i['dataset']=='lung')}")

    # ================================================================
    # 3. 5-Fold stratified split
    # ================================================================

    labels = [item["label"] for item in all_items]

    # بررسی کنید که هر دو کلاس وجود داشته باشند
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        print(f"\n❌ ERROR: Only one class found: {unique_labels}")
        print(
            "   Need both positive (1) and negative (0) samples for stratified split."
        )
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_items, labels)):
        for idx in test_idx:
            all_items[idx]["fold"] = fold

    # ================================================================
    # 4. Save to JSON
    # ================================================================

    with open(output_path, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"\n✅ Split file saved to: {output_path}")

    # نمایش توزیع در هر فولد
    print("\n📊 Fold distribution:")
    for fold in range(5):
        fold_items = [i for i in all_items if i.get("fold") == fold]
        pos = sum(1 for i in fold_items if i["label"] == 1)
        neg = sum(1 for i in fold_items if i["label"] == 0)
        print(
            f"   Fold {fold}: {len(fold_items)} items ({pos} positive, {neg} negative)"
        )


if __name__ == "__main__":
    create_split_json("split_breast_lung.json")
