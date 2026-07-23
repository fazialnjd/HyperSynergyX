"""
baseline_models_v6.py
=====================
Baseline ML models for drug synergy triplet prediction.
Profile-based feature representation — NO augmentation, NO PCA.

Feature engineering
-------------------
Each drug is represented as an 81-dim profile vector:
    drug_profile(d) = [chem[d, 0..26], atc[d, 0..26], tgt[d, 0..26]]

Each triplet (A, B, C) is represented as:
    feature(A, B, C) = concat(profile_A, profile_B, profile_C)  → 243-dim

Similarity matrices used: chem, atc, tgt  (int excluded)

⚠ WARNING: ratio = 40 train / 243 features = 0.165 → results are discovery-only.
  High variance across folds is expected. Do NOT interpret as reliable estimates.

Models (12 total — chosen for high-dim / low-sample regime)
----------------------------------------------------------
1. LR-L1          Lasso — sparse, built-in feature selection
2. LR-L2          Ridge — stable, all features retained
3. LR-ElasticNet  L1+L2 mix — best regularization for high-dim
4. SVM-Linear     Max-margin — well-suited for high-dim
5. SVM-RBF        Nonlinear — included for comparison
6. Ridge Classifier  L2 implicit, fast
7. Naive Bayes    Feature independence assumption, no tuning needed
8. Nearest Centroid  Simplest baseline, no hyperparameters
9. Random Forest  Ensemble method, handles non-linearity well
10. XGBoost       Gradient boosting, state-of-the-art performance
11. Logistic Regression Standard L2-regularized logistic regression
12. SVM           Full SVM with multiple kernels

CV strategy
-----------
Outer: 5-fold from split_breast.json (fold field)
Inner: 3-fold StratifiedKFold GridSearchCV (scoring=AUROC)
StandardScaler fit on train, transform on test — inside each fold.

Outputs (saved to baseline_results_v6/)
----------------------------------------
metrics_v6.csv              mean±std per model
best_hyperparams_v6.csv     best params per model per fold
auroc_bar_v6.png
roc_curves_v6.png
pr_curves_v6.png

Run
---
python baseline_models_v6.py
"""

import os
import json
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.base import clone
from sklearn.metrics.pairwise import euclidean_distances
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "similarity_matrices"  # directory with split_breast.json + .npy files
SIM_DIR = "similarity_matrices"  # directory with breast_S_*.npy files
OUT_DIR = "baseline_results_v6"

PAPER_AUROC = 0.9593
PAPER_AUPRC = 0.9453
PAPER_F1 = 0.9507

AUROC_STD_WARN = 0.15
INNER_AUROC_WARN = 0.50

LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]

MODEL_COLORS = {
    "LR-L1": "#0279EE",
    "LR-L2": "#75A025",
    "LR-ElasticNet": "#FF9400",
    "SVM-Linear": "#FD9BED",
    "SVM-RBF": "#E63946",
    "Ridge Classifier": "#457B9D",
    "Naive Bayes": "#A8DADC",
    "Nearest Centroid": "#6A0572",
    "Random Forest": "#2CA02C",
    "XGBoost": "#BCBD22",
    "Logistic Regression": "#17BECF",
    "SVM": "#7F7F7F",
}


# ── Load data ──────────────────────────────────────────────────────────────
def load_data(data_dir, sim_dir):
    with open("split_breast.json") as f:
        triplets = json.load(f)
    with open(os.path.join(data_dir, "breast_drug_names.json")) as f:
        drug_names = json.load(f)

    S_chem = np.load(os.path.join(sim_dir, "breast_S_chem.npy"))
    S_atc = np.load(os.path.join(sim_dir, "breast_S_atc.npy"))
    S_tgt = np.load(os.path.join(sim_dir, "breast_S_tgt.npy"))
    # int similarity excluded

    mats = {"chem": S_chem, "atc": S_atc, "tgt": S_tgt}
    drug2idx = {d: i for i, d in enumerate(drug_names)}
    return triplets, drug_names, mats, drug2idx


# ── Feature engineering ────────────────────────────────────────────────────
def drug_profile(d_idx, mats):
    """81-dim: [chem[d,:], atc[d,:], tgt[d,:]]"""
    return np.concatenate([mats["chem"][d_idx], mats["atc"][d_idx], mats["tgt"][d_idx]])


def make_feature(iA, iB, iC, mats):
    """243-dim: concat(profile_A, profile_B, profile_C)"""
    return np.concatenate(
        [drug_profile(iA, mats), drug_profile(iB, mats), drug_profile(iC, mats)]
    )


def build_arrays(triplets, drug2idx, mats):
    """Build X (50×243), y (50,), folds (50,) — NO augmentation."""
    n = len(triplets)
    X = np.zeros((n, 243), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    folds = np.zeros(n, dtype=np.int32)

    for i, t in enumerate(triplets):
        iA = drug2idx[t["drugA"]]
        iB = drug2idx[t["drugB"]]
        iC = drug2idx[t["drugC"]]
        X[i] = make_feature(iA, iB, iC, mats)
        y[i] = t["label"]
        folds[i] = t["fold"]

    return X, y, folds


# ── Model configs ──────────────────────────────────────────────────────────
def get_model_configs():
    """
    Returns dict: model_name -> {est, grid, proba}

    GridSearchCV selects best hyperparameters via inner 3-fold CV (scoring=AUROC).
    All grids use small C / large alpha values because ratio=0.165 requires
    strong regularization.

    proba=True  → predict_proba used for scoring
    proba=False → decision_function or centroid distance used
    """
    return {
        "LR-L1": {
            "est": LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=2000, random_state=42
            ),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]},
            "proba": True,
        },
        "LR-L2": {
            "est": LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
            ),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]},
            "proba": True,
        },
        "LR-ElasticNet": {
            "est": LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=3000, random_state=42
            ),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9]},
            "proba": True,
        },
        "SVM-Linear": {
            "est": SVC(kernel="linear", probability=True, random_state=42),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]},
            "proba": True,
        },
        "SVM-RBF": {
            "est": SVC(kernel="rbf", probability=True, random_state=42),
            "grid": {"C": [0.001, 0.01, 0.1, 1.0], "gamma": ["scale", "auto"]},
            "proba": True,
        },
        "Ridge Classifier": {
            "est": RidgeClassifier(random_state=42),
            "grid": {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            "proba": False,
        },
        "Naive Bayes": {
            "est": GaussianNB(),
            "grid": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
            "proba": True,
        },
        "Nearest Centroid": {
            "est": NearestCentroid(),
            "grid": {"metric": ["euclidean", "cosine"]},
            "proba": False,
        },
        # ── مدل‌های جدید ──────────────────────────────────────────────────
        "Random Forest": {
            "est": RandomForestClassifier(random_state=42, n_jobs=-1),
            "grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample"]
            },
            "proba": True,
        },
        "XGBoost": {
            "est": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
            "grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [1, 2, 4]  # برای داده‌های نامتوازن
            },
            "proba": True,
        },
        "Logistic Regression": {
            "est": LogisticRegression(max_iter=5000, random_state=42),
            "grid": {
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "newton-cg", "sag"]
            },
            "proba": True,
        },
        "SVM": {
            "est": SVC(probability=True, random_state=42),
            "grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "kernel": ["rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
                "degree": [2, 3, 4]
            },
            "proba": True,
        },
    }


# ── Probability proxy for models without predict_proba ────────────────────
# it is used for models that does not give a number as a output and this method scale these,
# because it is necessary for calculating AUPRC
def get_prob(clf, X_test, cfg_proba):
    """Return probability-like scores for AUROC computation."""
    if cfg_proba:
        return clf.predict_proba(X_test)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        rng = scores.max() - scores.min()
        return (scores - scores.min()) / (rng + 1e-9)
    # NearestCentroid: distance to class-0 centroid minus class-1 centroid
    d0 = euclidean_distances(X_test, clf.centroids_[[0]]).ravel()
    d1 = euclidean_distances(X_test, clf.centroids_[[1]]).ravel()
    return 1.0 / (1.0 + np.exp(d0 - d1))  # sigmoid of margin


# train
# ── Nested CV ──────────────────────────────────────────────────────────────
def run_nested_cv(X, y, folds, model_configs):
    """
    Outer: 5-fold from split_breast.json
    Inner: 3-fold StratifiedKFold GridSearchCV (scoring=roc_auc)
    StandardScaler fit on train, transform on test — inside each fold.
    """
    FOLD_IDS = sorted(np.unique(folds).tolist())
    results = {}

    for model_name, cfg in model_configs.items():
        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"{'='*55}")

        fold_metrics, roc_data, pr_data, best_params_list = [], [], [], []

        for fold_id in FOLD_IDS:
            train_mask = folds != fold_id
            test_mask = folds == fold_id
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Scale inside fold — no leakage
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # GridSearchCV — finds best hyperparameters on inner folds
            gs = GridSearchCV(
                estimator=clone(cfg["est"]),
                param_grid=cfg["grid"],
                cv=inner_skf,
                scoring="roc_auc",
                refit=True,
                n_jobs=-1,
                error_score=0.0,
            )
            gs.fit(X_train_sc, y_train)
            clf = gs.best_estimator_
            best_params = gs.best_params_
            inner_auroc = gs.best_score_

            # Outer evaluation
            prob = get_prob(clf, X_test_sc, cfg["proba"])
            pred = clf.predict(X_test_sc)

            auroc = roc_auc_score(y_test, prob)
            auprc = average_precision_score(y_test, prob)
            f1 = f1_score(y_test, pred, zero_division=0)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)

            fold_metrics.append(
                {
                    "auroc": auroc,
                    "auprc": auprc,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                }
            )
            best_params_list.append(
                {"fold": fold_id, "inner_auroc": round(inner_auroc, 4), **best_params}
            )

            fpr, tpr, _ = roc_curve(y_test, prob)
            p, r, _ = precision_recall_curve(y_test, prob)
            roc_data.append((fpr, tpr, auroc))
            pr_data.append((r, p, auprc))

            if inner_auroc < INNER_AUROC_WARN:
                print(
                    f"  ⚠ Fold {fold_id}: inner_AUROC={inner_auroc:.4f} < {INNER_AUROC_WARN}"
                )
            print(
                f"  Fold {fold_id}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                f"F1={f1:.4f}  inner_AUROC={inner_auroc:.4f}  best={best_params}"
            )

        aurocs = [m["auroc"] for m in fold_metrics]
        auprcs = [m["auprc"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]
        precs = [m["precision"] for m in fold_metrics]
        recs = [m["recall"] for m in fold_metrics]

        if np.std(aurocs) > AUROC_STD_WARN:
            print(f"  ⚠ WARNING: AUROC std={np.std(aurocs):.4f} > {AUROC_STD_WARN}")

        results[model_name] = {
            "auroc": np.mean(aurocs),
            "auroc_std": np.std(aurocs),
            "auprc": np.mean(auprcs),
            "auprc_std": np.std(auprcs),
            "f1": np.mean(f1s),
            "f1_std": np.std(f1s),
            "precision": np.mean(precs),
            "recall": np.mean(recs),
            "roc_data": roc_data,
            "pr_data": pr_data,
            "best_params": best_params_list,
        }
        print(
            f"\n  → Mean AUROC={np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}  "
            f"AUPRC={np.mean(auprcs):.4f}  F1={np.mean(f1s):.4f}"
        )

    return results


# ── Save CSVs ──────────────────────────────────────────────────────────────
def save_csvs(results, out_dir):
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "model": name,
                "AUROC": round(r["auroc"], 4),
                "AUROC_std": round(r["auroc_std"], 4),
                "AUPRC": round(r["auprc"], 4),
                "AUPRC_std": round(r["auprc_std"], 4),
                "F1": round(r["f1"], 4),
                "F1_std": round(r["f1_std"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics_v6.csv"), index=False)
    print(f"Saved: metrics_v6.csv")

    hp_rows = []
    for name, r in results.items():
        for p in r["best_params"]:
            hp_rows.append({"model": name, **p})
    pd.DataFrame(hp_rows).to_csv(
        os.path.join(out_dir, "best_hyperparams_v6.csv"), index=False
    )
    print(f"Saved: best_hyperparams_v6.csv")


# ── Plots ──────────────────────────────────────────────────────────────────
def make_plots(results, out_dir):
    model_names = list(results.keys())
    colors = [MODEL_COLORS[m] for m in model_names]

    # AUROC bar
    aurocs = [results[m]["auroc"] for m in model_names]
    stds = [results[m]["auroc_std"] for m in model_names]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        model_names,
        aurocs,
        color=colors,
        width=0.55,
        yerr=stds,
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )
    ax.axhline(
        PAPER_AUROC,
        color="#000000",
        linestyle="--",
        linewidth=1.8,
        label=f"HyperSynergyX ({PAPER_AUROC})",
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (0.50)")
    for bar, val, std in zip(bars, aurocs, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + std + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Mean AUROC (5-fold CV)")
    ax.set_title(
        "AUROC — Profile 243-dim, No Augmentation (v6 + New Models)\n"
        "⚠ ratio=0.165 → discovery-only",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "auroc_bar_v6.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random", zorder=1)
    ax.axhline(
        PAPER_AUROC,
        color="#000000",
        linestyle=":",
        linewidth=1.8,
        label=f"HyperSynergyX AUROC={PAPER_AUROC}",
        zorder=2,
    )
    for i, (m, col) in enumerate(zip(model_names, colors)):
        roc_data = results[m]["roc_data"]
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in roc_data]
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        ls = LINESTYLES[i]
        for fpr, tpr, _ in roc_data:
            ax.plot(fpr, tpr, color=col, alpha=0.15, linewidth=0.8)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=col,
            linewidth=2.2,
            linestyle=ls,
            zorder=3,
            label=f"{m} ({results[m]['auroc']:.3f}±{results[m]['auroc_std']:.3f})",
        )
        ax.fill_between(
            mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=col, alpha=0.06
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Profile 243-dim (v6 + New Models)", fontsize=12)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "roc_curves_v6.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # PR curves
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(
        0.5, color="gray", linestyle=":", linewidth=1, label="Random (0.50)", zorder=1
    )
    ax.axhline(
        PAPER_AUPRC,
        color="#000000",
        linestyle=":",
        linewidth=1.8,
        label=f"HyperSynergyX AUPRC={PAPER_AUPRC}",
        zorder=2,
    )
    for i, (m, col) in enumerate(zip(model_names, colors)):
        pr_data = results[m]["pr_data"]
        mean_rec = np.linspace(0, 1, 200)
        precs = [np.interp(mean_rec, rec[::-1], prec[::-1]) for rec, prec, _ in pr_data]
        mean_prec = np.mean(precs, axis=0)
        std_prec = np.std(precs, axis=0)
        ls = LINESTYLES[i]
        for rec, prec, _ in pr_data:
            ax.plot(rec, prec, color=col, alpha=0.15, linewidth=0.8)
        ax.plot(
            mean_rec,
            mean_prec,
            color=col,
            linewidth=2.2,
            linestyle=ls,
            zorder=3,
            label=f"{m} ({results[m]['auprc']:.3f}±{results[m]['auprc_std']:.3f})",
        )
        ax.fill_between(
            mean_rec, mean_prec - std_prec, mean_prec + std_prec, color=col, alpha=0.06
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves — Profile 243-dim (v6 + New Models)", fontsize=12)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.0), framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves_v6.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Plots saved.")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  baseline_models_v6.py")
    print("  Profile 243-dim | 12 models | No augmentation")
    print("=" * 55)

    # Load
    triplets, drug_names, mats, drug2idx = load_data(DATA_DIR, SIM_DIR)

    # Features
    X, y, folds = build_arrays(triplets, drug2idx, mats)
    n_train = int(len(y) * 0.8)
    print(f"\nDataset: {len(triplets)} triplets, {len(drug_names)} drugs")
    print(f"X shape: {X.shape}")
    print(f"Labels:  {y.sum()} positive / {(y==0).sum()} negative")
    print(
        f"⚠ Sample/feature ratio: {n_train}/{X.shape[1]} = "
        f"{n_train/X.shape[1]:.3f}  → discovery-only"
    )

    # Models
    model_configs = get_model_configs()
    print(f"\nModels: {list(model_configs.keys())}")
    for name, cfg in model_configs.items():
        n_combos = 1
        for v in cfg["grid"].values():
            n_combos *= len(v)
        print(f"  {name}: {n_combos} combos × 3 inner folds")

    # Nested CV
    results = run_nested_cv(X, y, folds, model_configs)

    # Save
    print("\n" + "=" * 55)
    save_csvs(results, OUT_DIR)
    make_plots(results, OUT_DIR)

    # Summary
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"{'Model':<20} {'AUROC':>8} {'±':>6} {'AUPRC':>8} {'F1':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(
            f"{name:<20} {r['auroc']:>8.4f} {r['auroc_std']:>6.4f} "
            f"{r['auprc']:>8.4f} {r['f1']:>8.4f}"
        )
    print("-" * 55)
    print(
        f"{'HyperSynergyX':<20} {PAPER_AUROC:>8.4f} {'—':>6} "
        f"{PAPER_AUPRC:>8.4f} {PAPER_F1:>8.4f}"
    )
    print(f"\nOutputs saved to: {OUT_DIR}/")