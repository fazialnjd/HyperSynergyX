"""
baseline_models_v5.py
=====================
Profile-based feature representation for drug synergy triplet prediction.

Feature engineering
-------------------
Each drug is represented as an 81-dim profile vector:
    drug_profile(d) = [chem[d,:], atc[d,:], tgt[d,:]]   shape: (81,)

Each triplet (A, B, C) is represented as:
    feature(A,B,C) = concat(profile(A), profile(B), profile(C))  shape: (243,)

Two parallel versions
---------------------
Version A — PCA30 + 4 models (LR, RF, SVM, XGBoost)
    - PCA inside each fold: 243 → 30 dim  (ratio = 240/30 = 8 ✅)
    - Nested CV: outer 5-fold, inner 3-fold GridSearch

Version B — No PCA + 2 models (LR, SVM)
    - Full 243-dim, only regularized models
    - Nested CV: outer 5-fold, inner 3-fold GridSearch

Augmentation
------------
- Train: all 6 permutations of each non-test triplet  (240 × 243 per fold)
- Test:  original ordering only                        (10  × 243 per fold)
- Split happens BEFORE augmentation → no leakage

Similarity matrices used: chem, atc, tgt  (int excluded)

Outputs (saved to baseline_results_v5/)
----------------------------------------
- metrics_profile_pca.csv
- metrics_profile_nopca.csv
- best_hyperparams_profile_pca.csv
- best_hyperparams_profile_nopca.csv
- auroc_bar_profile_pca.png
- roc_curves_profile_pca.png
- pr_curves_profile_pca.png
- auroc_bar_profile_nopca.png
- roc_curves_profile_nopca.png
- pr_curves_profile_nopca.png
"""

import os
import json
import warnings
from itertools import permutations, product

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = (
    "similarity_matrices"  # directory containing split_breast.json and .npy files
)
OUT_DIR = "baseline_results_v5"
N_PCA = 30  # PCA components for Version A
AUROC_STD_WARN = 0.15
INNER_AUROC_WARN = 0.50

MODEL_COLORS = {
    "Logistic Regression": "#0279EE",
    "Random Forest": "#75A025",
    "XGBoost": "#FF9400",
    "SVM": "#FD9BED",
}


# ── Load data ──────────────────────────────────────────────────────────────
def load_data(data_dir):
    with open("split_breast.json") as f:
        triplets = json.load(f)
    with open(os.path.join(data_dir, "breast_drug_names.json")) as f:
        drug_names = json.load(f)
    S_chem = np.load(os.path.join(data_dir, "breast_S_chem.npy"))
    S_atc = np.load(os.path.join(data_dir, "breast_S_atc.npy"))
    S_tgt = np.load(os.path.join(data_dir, "breast_S_tgt.npy"))
    mats = {"chem": S_chem, "atc": S_atc, "tgt": S_tgt}
    drug2idx = {d: i for i, d in enumerate(drug_names)}
    return triplets, drug_names, mats, drug2idx


# ── Feature engineering ────────────────────────────────────────────────────
def drug_profile(d_idx, mats):
    """81-dim: [chem[d,:], atc[d,:], tgt[d,:]]"""
    return np.concatenate([mats["chem"][d_idx], mats["atc"][d_idx], mats["tgt"][d_idx]])


def make_profile_feature(iA, iB, iC, mats):
    """243-dim: concat profiles of A, B, C"""
    return np.concatenate(
        [drug_profile(iA, mats), drug_profile(iB, mats), drug_profile(iC, mats)]
    )


def build_arrays(triplets, drug2idx, mats):
    """Build original (50×243) and augmented (300×243) arrays."""
    PERMS = list(permutations([0, 1, 2]))
    n = len(triplets)

    X_orig = np.zeros((n, 243), dtype=np.float32)
    y_orig = np.zeros(n, dtype=np.int32)
    fold_orig = np.zeros(n, dtype=np.int32)

    for i, t in enumerate(triplets):
        iA = drug2idx[t["drugA"]]
        iB = drug2idx[t["drugB"]]
        iC = drug2idx[t["drugC"]]
        X_orig[i] = make_profile_feature(iA, iB, iC, mats)
        y_orig[i] = t["label"]
        fold_orig[i] = t["fold"]

    X_aug_list, y_aug_list, fold_aug_list = [], [], []
    for t in triplets:
        idxs = [drug2idx[t["drugA"]], drug2idx[t["drugB"]], drug2idx[t["drugC"]]]
        for perm in PERMS:
            iA, iB, iC = idxs[perm[0]], idxs[perm[1]], idxs[perm[2]]
            X_aug_list.append(make_profile_feature(iA, iB, iC, mats))
            y_aug_list.append(t["label"])
            fold_aug_list.append(t["fold"])

    X_aug = np.array(X_aug_list, dtype=np.float32)
    y_aug = np.array(y_aug_list, dtype=np.int32)
    fold_aug = np.array(fold_aug_list, dtype=np.int32)

    return X_orig, y_orig, fold_orig, X_aug, y_aug, fold_aug


# ── XGBoost manual inner CV (bypasses GridSearchCV XGBoost 2.x bug) ────────
def xgb_inner_cv(X_tr, y_tr, param_grid, inner_skf):
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    best_score, best_params = -1, None
    for combo in combos:
        params = dict(zip(keys, combo))
        scores = []
        for tr_i, val_i in inner_skf.split(X_tr, y_tr):
            mdl = XGBClassifier(
                objective="binary:logistic",
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
                **params,
            )
            mdl.fit(X_tr[tr_i], y_tr[tr_i])
            prob = mdl.predict_proba(X_tr[val_i])[:, 1]
            if len(np.unique(y_tr[val_i])) > 1:
                scores.append(roc_auc_score(y_tr[val_i], prob))
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_params = params
    return best_params, best_score


# ── Nested CV runner ───────────────────────────────────────────────────────
def run_nested_cv(
    model_configs,
    X_orig,
    y_orig,
    fold_orig,
    X_aug,
    y_aug,
    fold_aug,
    use_pca=False,
    n_pca=30,
    version_label="",
):
    FOLDS = sorted(np.unique(fold_orig).tolist())
    results = {}

    for model_name, cfg in model_configs.items():
        print(f"\n{'='*55}")
        print(f"  {model_name}  [{version_label}]")
        print(f"{'='*55}")

        fold_metrics, roc_data, pr_data, best_params_list = [], [], [], []

        for fold_id in FOLDS:
            # Split
            train_mask = fold_aug != fold_id
            test_mask = fold_orig == fold_id
            X_train_raw, y_train = X_aug[train_mask], y_aug[train_mask]
            X_test_raw, y_test = X_orig[test_mask], y_orig[test_mask]

            # Scale inside fold
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train_raw)
            X_test_sc = scaler.transform(X_test_raw)

            # Optional PCA inside fold
            if use_pca:
                pca = PCA(n_components=n_pca, random_state=42)
                X_train_in = pca.fit_transform(X_train_sc)
                X_test_in = pca.transform(X_test_sc)
            else:
                X_train_in = X_train_sc
                X_test_in = X_test_sc

            inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # Inner CV
            if model_name == "XGBoost":
                best_params, inner_auroc = xgb_inner_cv(
                    X_train_in, y_train, cfg["grid"], inner_skf
                )
                clf = XGBClassifier(
                    objective="binary:logistic",
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                    **best_params,
                )
                clf.fit(X_train_in, y_train)
            else:
                gs = GridSearchCV(
                    clone(cfg["estimator"]),
                    cfg["grid"],
                    cv=inner_skf,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                gs.fit(X_train_in, y_train)
                clf = gs.best_estimator_
                best_params = gs.best_params_
                inner_auroc = gs.best_score_

            # Outer evaluation
            prob = clf.predict_proba(X_test_in)[:, 1]
            pred = clf.predict(X_test_in)
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
                {"fold": fold_id, "inner_auroc": inner_auroc, **best_params}
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
def save_metrics_csv(results, path):
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
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}")


def save_hyperparams_csv(results, path):
    rows = []
    for name, r in results.items():
        for p in r["best_params"]:
            rows.append({"model": name, **p})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}")


# ── Plots ──────────────────────────────────────────────────────────────────
def make_plots(results, suffix, title_suffix, out_dir):
    models = list(results.keys())
    colors = [MODEL_COLORS[m] for m in models]

    # AUROC bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    aurocs = [results[m]["auroc"] for m in models]
    stds = [results[m]["auroc_std"] for m in models]
    bars = ax.bar(
        models,
        aurocs,
        color=colors,
        width=0.5,
        yerr=stds,
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )
    ax.axhline(
        0.9593,
        color="#000000",
        linestyle="--",
        linewidth=1.5,
        label="HyperSynergyX (0.9593)",
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (0.50)")
    for bar, val, std in zip(bars, aurocs, stds):
        if not (np.isnan(val) or np.isnan(std)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Mean AUROC (5-fold CV)")
    ax.set_title(f"AUROC Comparison — {title_suffix}")
    ax.legend(fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"auroc_bar_{suffix}.png"), dpi=150)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random", zorder=1)
    for m, col in zip(models, colors):
        roc_data = results[m]["roc_data"]
        if not roc_data:
            continue
        for fpr, tpr, _ in roc_data:
            ax.plot(fpr, tpr, color=col, alpha=0.35, linewidth=1.0)
        mean_fpr = np.linspace(0, 1, 200)
        tprs_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in roc_data]
        mean_tpr = np.mean(tprs_interp, axis=0)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=col,
            linewidth=2.5,
            zorder=3,
            label=f"{m} (mean={results[m]['auroc']:.3f})",
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {title_suffix}")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"roc_curves_{suffix}.png"), dpi=150)
    plt.close()

    # PR curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.axhline(
        0.5, color="gray", linestyle=":", linewidth=1, label="Random (0.50)", zorder=1
    )
    for m, col in zip(models, colors):
        pr_data = results[m]["pr_data"]
        if not pr_data:
            continue
        for rec_vals, prec_vals, _ in pr_data:
            ax.plot(rec_vals, prec_vals, color=col, alpha=0.35, linewidth=1.0)
        mean_rec = np.linspace(0, 1, 200)
        precs_interp = [
            np.interp(mean_rec, rec_vals[::-1], prec_vals[::-1])
            for rec_vals, prec_vals, _ in pr_data
        ]
        mean_prec = np.mean(precs_interp, axis=0)
        ax.plot(
            mean_rec,
            mean_prec,
            color=col,
            linewidth=2.5,
            zorder=3,
            label=f"{m} (mean={results[m]['auprc']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curves — {title_suffix}")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_curves_{suffix}.png"), dpi=150)
    plt.close()

    print(f"Plots saved for: {suffix}")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    triplets, drug_names, mats, drug2idx = load_data(DATA_DIR)
    X_orig, y_orig, fold_orig, X_aug, y_aug, fold_aug = build_arrays(
        triplets, drug2idx, mats
    )

    print(f"Dataset: {len(triplets)} triplets, {len(drug_names)} drugs")
    print(f"X_orig: {X_orig.shape}  X_aug: {X_aug.shape}")
    print(f"Sample/feature ratio (raw):   {len(y_aug[fold_aug!=0])/243:.3f}")
    print(f"Sample/feature ratio (PCA30): {len(y_aug[fold_aug!=0])/30:.1f}")

    # ── Version A: PCA30 + 4 models ───────────────────────────────────────
    MODELS_A = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=42),
            "grid": {"C": [0.01, 0.1, 1.0, 5.0], "penalty": ["l2"]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "grid": {
                "n_estimators": [100, 200],
                "max_depth": [2, 3, 4],
                "min_samples_leaf": [1, 3, 5],
            },
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=42),
            "grid": {
                "C": [0.01, 0.1, 1.0, 5.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
        "XGBoost": {
            "estimator": None,
            "grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [2, 3],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        },
    }

    print("\n\n" + "=" * 55)
    print("  VERSION A — Profile + PCA30 (4 models)")
    print("=" * 55)
    results_A = run_nested_cv(
        MODELS_A,
        X_orig,
        y_orig,
        fold_orig,
        X_aug,
        y_aug,
        fold_aug,
        use_pca=True,
        n_pca=N_PCA,
        version_label=f"Version A — PCA{N_PCA}",
    )
    save_metrics_csv(results_A, os.path.join(OUT_DIR, "metrics_profile_pca.csv"))
    save_hyperparams_csv(
        results_A, os.path.join(OUT_DIR, "best_hyperparams_profile_pca.csv")
    )
    make_plots(results_A, "profile_pca", f"Profile + PCA{N_PCA} (4 models)", OUT_DIR)

    # ── Version B: No PCA + 2 models ──────────────────────────────────────
    MODELS_B = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=42),
            "grid": {"C": [0.001, 0.01, 0.1, 1.0], "penalty": ["l2"]},
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=42),
            "grid": {
                "C": [0.001, 0.01, 0.1, 1.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
    }

    print("\n\n" + "=" * 55)
    print("  VERSION B — Profile, No PCA (LR + SVM)")
    print("=" * 55)
    results_B = run_nested_cv(
        MODELS_B,
        X_orig,
        y_orig,
        fold_orig,
        X_aug,
        y_aug,
        fold_aug,
        use_pca=False,
        version_label="Version B — No PCA, 243-dim",
    )
    save_metrics_csv(results_B, os.path.join(OUT_DIR, "metrics_profile_nopca.csv"))
    save_hyperparams_csv(
        results_B, os.path.join(OUT_DIR, "best_hyperparams_profile_nopca.csv")
    )
    make_plots(results_B, "profile_nopca", "Profile, No PCA (LR + SVM)", OUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"{'Model':<25} {'Version':<12} {'AUROC':>8} {'±':>4} {'AUPRC':>8} {'F1':>8}")
    print("-" * 65)
    for m, r in results_A.items():
        print(
            f"{m:<25} {'A (PCA30)':<12} {r['auroc']:>8.4f} {r['auroc_std']:>4.4f} "
            f"{r['auprc']:>8.4f} {r['f1']:>8.4f}"
        )
    for m, r in results_B.items():
        print(
            f"{m:<25} {'B (NoPCA)':<12} {r['auroc']:>8.4f} {r['auroc_std']:>4.4f} "
            f"{r['auprc']:>8.4f} {r['f1']:>8.4f}"
        )
    print("-" * 65)
    print(
        f"{'HyperSynergyX (paper)':<25} {'—':<12} {'0.9593':>8} {'—':>4} "
        f"{'0.9453':>8} {'0.9507':>8}"
    )
    print("\nDone. Results saved to:", OUT_DIR)
