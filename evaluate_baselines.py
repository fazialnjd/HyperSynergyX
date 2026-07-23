# evaluate_baselines.py

"""
the first version it also put rfwr output as a feature for baseline models input
feature baseline:
[
mean_atc,
max_atc,
mean_target,
max_target,
mean_combined,
3
]
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from sklearn.dummy import DummyClassifier


def load_data(split_json):
    """Load split file and prepare features"""
    with open(split_json, "r") as f:
        items = json.load(f)

    # Load synergy scores as features (from DBRWH output)
    dataset = "breast" if "breast" in split_json else "lung"
    scores_df = pd.read_csv(f"output/{dataset}_synergy/synergy_triplets_ranked.csv")

    # Create mapping from triplet to score
    score_map = {}
    for _, row in scores_df.iterrows():
        key = tuple(sorted([row["drug_1"], row["drug_2"], row["drug_3"]]))
        score_map[key] = row["synergy_score"]

    # Create feature matrix
    X = []
    y = []

    for item in items:
        key = tuple(sorted([item["drugA"], item["drugB"], item["drugC"]]))
        score = score_map.get(key, 0.0)
        X.append([score])
        y.append(item["label"])

    return np.array(X), np.array(y)


def create_simple_features(split_json):
    """Create simple features without DBRWH (for independent evaluation)"""
    with open(split_json, "r") as f:
        items = json.load(f)

    # مسیر صحیح برای فایل‌های similarity matrices
    dataset = "breast" if "breast" in split_json else "lung"
    sim_dir = Path("output/similarity_matrices")

    sim_atc = np.load(sim_dir / f"{dataset}_S_atc.npy")
    sim_target = np.load(sim_dir / f"{dataset}_S_tgt.npy")

    # Load drug names
    with open(sim_dir / f"{dataset}_drug_names.json", "r") as f:
        drug_names = json.load(f)

    drug_to_idx = {name: i for i, name in enumerate(drug_names)}

    X = []
    y = []

    for item in items:
        a, b, c = item["drugA"], item["drugB"], item["drugC"]

        # Check if all drugs exist in similarity matrix
        if a not in drug_to_idx or b not in drug_to_idx or c not in drug_to_idx:
            X.append([0, 0, 0, 0, 0, 0])
            y.append(item["label"])
            continue

        ia, ib, ic = drug_to_idx[a], drug_to_idx[b], drug_to_idx[c]

        # Features:
        # 1. Mean ATC similarity between drug pairs
        atc_ab = sim_atc[ia, ib]
        atc_ac = sim_atc[ia, ic]
        atc_bc = sim_atc[ib, ic]
        mean_atc = np.mean([atc_ab, atc_ac, atc_bc])

        # 2. Max ATC similarity
        max_atc = max(atc_ab, atc_ac, atc_bc)

        # 3. Mean Target similarity
        tgt_ab = sim_target[ia, ib]
        tgt_ac = sim_target[ia, ic]
        tgt_bc = sim_target[ib, ic]
        mean_target = np.mean([tgt_ab, tgt_ac, tgt_bc])

        # 4. Max Target similarity
        max_target = max(tgt_ab, tgt_ac, tgt_bc)

        # 5. Combined (max of ATC and Target)
        combined_ab = max(atc_ab, tgt_ab)
        combined_ac = max(atc_ac, tgt_ac)
        combined_bc = max(atc_bc, tgt_bc)
        mean_combined = np.mean([combined_ab, combined_ac, combined_bc])

        # 6. Number of unique drugs (always 3 for triplets)
        n_unique = len(set([a, b, c]))

        X.append([mean_atc, max_atc, mean_target, max_target, mean_combined, n_unique])
        y.append(item["label"])

    return np.array(X), np.array(y)


def evaluate_model(model, model_name, X, y, n_splits=5):
    """Evaluate a model using cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)

        # Predict
        if hasattr(model_clone, "predict_proba"):
            y_score = model_clone.predict_proba(X_test)[:, 1]
        else:
            y_score = model_clone.decision_function(X_test)
        y_pred = model_clone.predict(X_test)

        # Metrics
        auroc = roc_auc_score(y_test, y_score)
        auprc = average_precision_score(y_test, y_score)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        results.append(
            {
                "fold": fold,
                "AUROC": auroc,
                "AUPRC": auprc,
                "F1": f1,
                "Precision": prec,
                "Recall": rec,
            }
        )

    # Average results
    avg = {
        "AUROC": np.mean([r["AUROC"] for r in results]),
        "AUPRC": np.mean([r["AUPRC"] for r in results]),
        "F1": np.mean([r["F1"] for r in results]),
        "Precision": np.mean([r["Precision"] for r in results]),
        "Recall": np.mean([r["Recall"] for r in results]),
        "std_AUROC": np.std([r["AUROC"] for r in results]),
    }

    return avg, results


def main():
    print("=" * 70)
    print("BASELINE MODELS EVALUATION")
    print("=" * 70)

    datasets = [("split_breast.json", "breast", 5), ("split_lung.json", "lung", 2)]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Dummy (Majority)": DummyClassifier(strategy="most_frequent"),
    }

    all_results = {}

    for split_file, dataset_name, n_splits in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()} (n_splits={n_splits})")
        print(f"{'='*70}")

        # گزینه 1: استفاده از DBRWH scores به عنوان ویژگی
        print("\n📊 Using DBRWH scores as features:")
        X_dbrwh, y = load_data(split_file)
        print(
            f"   Features shape: {X_dbrwh.shape}, Positive: {sum(y)}, Negative: {len(y)-sum(y)}"
        )

        for model_name, model in models.items():
            try:
                avg, _ = evaluate_model(model, model_name, X_dbrwh, y, n_splits)
                print(
                    f"   {model_name:<22} AUROC: {avg['AUROC']:.4f} (±{avg['std_AUROC']:.4f}) | AUPRC: {avg['AUPRC']:.4f} | F1: {avg['F1']:.4f}"
                )
            except Exception as e:
                print(f"   {model_name:<22} ❌ Error: {e}")

        # گزینه 2: استفاده از similarity features (ATC, Target)
        print("\n📊 Using similarity features (ATC + Target):")
        X_sim, y = create_simple_features(split_file)
        print(f"   Features shape: {X_sim.shape}")

        for model_name, model in models.items():
            try:
                avg, _ = evaluate_model(model, model_name, X_sim, y, n_splits)
                print(
                    f"   {model_name:<22} AUROC: {avg['AUROC']:.4f} (±{avg['std_AUROC']:.4f}) | AUPRC: {avg['AUPRC']:.4f} | F1: {avg['F1']:.4f}"
                )
            except Exception as e:
                print(f"   {model_name:<22} ❌ Error: {e}")

        all_results[dataset_name] = {"dbrwh_features": {}, "similarity_features": {}}

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

# TODO: run with smiles as a feature, as a similarity metric in rwr
