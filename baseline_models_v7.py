"""
baseline_models_v7.py

Version 7:
- Same 243-dim profile features as v6
- No leakage augmentation
- Positive permutation augmentation inside each outer fold only
- Balanced negative sampling from train-only negative pool
- Removed Nearest Centroid
- Added: AUROC bar plot, ROC curves, PR curves, hyperparameters CSV
- Added: Random Forest, XGBoost, Logistic Regression, SVM as separate models

Pipeline:
Outer CV -> augment train -> build train negative pool -> scale -> inner GridSearch -> test
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import permutations, combinations
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── تنظیمات مسیرها و ثابت‌ها ──────────────────────────────────────────────
DATA_DIR = "similarity_matrices"
SIM_DIR = "similarity_matrices"
OUT_DIR = "baseline_results_v7"

RANDOM_STATE = 42

# رنگ و استایل برای مدل‌ها (برای نمودارها)
MODEL_COLORS = {
    "LR-L1": "#1f77b4",
    "LR-L2": "#ff7f0e",
    "LR-ElasticNet": "#2ca02c",
    "SVM-Linear": "#d62728",
    "SVM-RBF": "#9467bd",
    "Ridge Classifier": "#8c564b",
    "Naive Bayes": "#e377c2",
    "Random Forest": "#7f7f7f",
    "XGBoost": "#bcbd22",
    "Logistic Regression": "#17becf",
}

LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]


# ── بارگذاری داده‌ها ──────────────────────────────────────────────────────
def load_data():
    with open("split_breast.json") as f:
        triplets = json.load(f)

    with open(os.path.join(DATA_DIR, "breast_drug_names.json")) as f:
        drug_names = json.load(f)

    mats = {
        "chem": np.load(os.path.join(SIM_DIR, "breast_S_chem.npy")),
        "atc": np.load(os.path.join(SIM_DIR, "breast_S_atc.npy")),
        "tgt": np.load(os.path.join(SIM_DIR, "breast_S_tgt.npy")),
    }

    drug2idx = {d: i for i, d in enumerate(drug_names)}
    return triplets, mats, drug2idx


def drug_profile(idx, mats):
    """پروفایل ۲۴۳ بعدی هر دارو (۸۱+۸۱+۸۱)"""
    return np.concatenate([mats["chem"][idx], mats["atc"][idx], mats["tgt"][idx]])


def triplet_feature(t, drug2idx, mats):
    """ویژگی‌های یک triplet از سه دارو"""
    ids = [drug2idx[t["drugA"]], drug2idx[t["drugB"]], drug2idx[t["drugC"]]]
    return np.concatenate(
        [
            drug_profile(ids[0], mats),
            drug_profile(ids[1], mats),
            drug_profile(ids[2], mats),
        ]
    )


# ── Augmentation ──────────────────────────────────────────────────────────
def permute_positive_triplet(t):
    """جایگشت‌های مثبت یک triplet (۶ حالت)"""
    drugs = [t["drugA"], t["drugB"], t["drugC"]]
    out = []
    for p in permutations(drugs):
        out.append({"drugA": p[0], "drugB": p[1], "drugC": p[2], "label": 1})
    return out


def build_negative_pool(train_positive, drug2idx):
    """ساخت مخزن نمونه‌های منفی از داروهای موجود در train"""
    drugs = set()
    for t in train_positive:
        drugs.update([t["drugA"], t["drugB"], t["drugC"]])
    drugs = sorted(list(drugs))

    positives = {
        tuple(sorted([t["drugA"], t["drugB"], t["drugC"]])) for t in train_positive
    }

    pool = []
    for comb in combinations(drugs, 3):
        if comb not in positives:
            pool.append(
                {"drugA": comb[0], "drugB": comb[1], "drugC": comb[2], "label": 0}
            )

    return pool


def augment_train(train_triplets):
    """افزایش داده‌های آموزش: جایگشت مثبت‌ها + نمونه‌گیری منفی متوازن"""
    positives = [t for t in train_triplets if t["label"] == 1]
    negatives = [t for t in train_triplets if t["label"] == 0]

    aug_pos = []
    for p in positives:
        aug_pos.extend(permute_positive_triplet(p))

    pool = build_negative_pool(positives, None)
    random.seed(RANDOM_STATE)
    n_neg = len(aug_pos)
    sampled_neg = random.sample(pool, min(n_neg, len(pool)))

    return aug_pos + sampled_neg


# ── ساخت ماتریس ویژگی‌ها ─────────────────────────────────────────────────
def make_dataset(triplets, drug2idx, mats):
    X = np.array(
        [triplet_feature(t, drug2idx, mats) for t in triplets], dtype=np.float32
    )
    y = np.array([t["label"] for t in triplets])
    return X, y


# ── تعریف مدل‌ها و Grid Search ────────────────────────────────────────────
def get_models():
    return {
        "LR-L1": {
            "est": LogisticRegression(penalty="l1", solver="liblinear", max_iter=3000),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1, 1]},
        },
        "LR-L2": {
            "est": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1, 1]},
        },
        "LR-ElasticNet": {
            "est": LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=3000
            ),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1], "l1_ratio": [0.1, 0.5, 0.9]},
        },
        "SVM-Linear": {
            "est": SVC(kernel="linear", probability=True),
            "grid": {"C": [0.0001, 0.001, 0.01, 0.1, 1]},
        },
        "SVM-RBF": {
            "est": SVC(kernel="rbf", probability=True),
            "grid": {"C": [0.001, 0.01, 0.1, 1], "gamma": ["scale", "auto"]},
        },
        "Ridge Classifier": {
            "est": RidgeClassifier(),
            "grid": {"alpha": [0.1, 1, 10, 100, 1000]},
        },
        "Naive Bayes": {
            "est": GaussianNB(),
            "grid": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        },
        # ── مدل‌های جدید ──────────────────────────────────────────────────
        "Random Forest": {
            "est": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample"],
            },
        },
        "XGBoost": {
            "est": XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss"
            ),
            "grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [1, 2, 4],  # برای داده‌های نامتوازن
            },
        },
        "Logistic Regression": {
            "est": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            "grid": {
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "newton-cg", "sag"],
            },
        },
        "SVM": {
            "est": SVC(probability=True, random_state=RANDOM_STATE),
            "grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "kernel": ["rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
                "degree": [2, 3, 4],
            },
        },
    }


# ── نمودارها ──────────────────────────────────────────────────────────────
def make_plots(results_dict, hyperparams_df, out_dir):
    """
    رسم نمودارهای:
    1. AUROC bar chart
    2. ROC curves (همه فولدها)
    3. PR curves (همه فولدها)
    """
    model_names = list(results_dict.keys())
    colors = [MODEL_COLORS.get(m, "#000000") for m in model_names]

    # ── ۱. نمودار میله‌ای AUROC ──────────────────────────────────────────
    aurocs = [results_dict[m]["auroc"] for m in model_names]
    stds = [results_dict[m]["auroc_std"] for m in model_names]

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
        "AUROC — Profile 243-dim, With Augmentation (v7 + New Models)", fontsize=11
    )
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "auroc_bar_v7.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ auroc_bar_v7.png saved.")

    # ── ۲. منحنی‌های ROC ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random", zorder=1)

    for i, (m, col) in enumerate(zip(model_names, colors)):
        roc_data = results_dict[m]["roc_data"]
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_data]
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        ls = LINESTYLES[i % len(LINESTYLES)]

        # رسم خطوط نازک برای هر فولد
        for fpr, tpr in roc_data:
            ax.plot(fpr, tpr, color=col, alpha=0.15, linewidth=0.8)

        # رسم میانگین
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=col,
            linewidth=2.2,
            linestyle=ls,
            zorder=3,
            label=f"{m} ({results_dict[m]['auroc']:.3f}±{results_dict[m]['auroc_std']:.3f})",
        )
        ax.fill_between(
            mean_fpr,
            mean_tpr - std_tpr,
            mean_tpr + std_tpr,
            color=col,
            alpha=0.06,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Profile 243-dim (v7 + New Models)", fontsize=12)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "roc_curves_v7.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("✅ roc_curves_v7.png saved.")

    # ── ۳. منحنی‌های PR ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(
        0.5, color="gray", linestyle=":", linewidth=1, label="Random (0.50)", zorder=1
    )

    for i, (m, col) in enumerate(zip(model_names, colors)):
        pr_data = results_dict[m]["pr_data"]
        mean_rec = np.linspace(0, 1, 200)
        precs = [np.interp(mean_rec, rec[::-1], prec[::-1]) for rec, prec in pr_data]
        mean_prec = np.mean(precs, axis=0)
        std_prec = np.std(precs, axis=0)
        ls = LINESTYLES[i % len(LINESTYLES)]

        # رسم خطوط نازک برای هر فولد
        for rec, prec in pr_data:
            ax.plot(rec, prec, color=col, alpha=0.15, linewidth=0.8)

        # رسم میانگین
        ax.plot(
            mean_rec,
            mean_prec,
            color=col,
            linewidth=2.2,
            linestyle=ls,
            zorder=3,
            label=f"{m} ({results_dict[m]['auprc']:.3f}±{results_dict[m]['auprc_std']:.3f})",
        )
        ax.fill_between(
            mean_rec,
            mean_prec - std_prec,
            mean_prec + std_prec,
            color=col,
            alpha=0.06,
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves — Profile 243-dim (v7 + New Models)", fontsize=12)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.0), framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves_v7.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ pr_curves_v7.png saved.")


# ── اجرای اصلی ───────────────────────────────────────────────────────────
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    triplets, mats, drug2idx = load_data()
    folds = sorted(set(t["fold"] for t in triplets))

    # ذخیره نتایج هر مدل-فولد
    all_results = []
    # ذخیره بهترین هایپرپارامترهای هر مدل-فولد
    hyperparams_list = []
    # ذخیره داده‌های ROC و PR برای نمودارها
    plot_data = {}

    for name, cfg in get_models().items():
        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")

        fold_scores = []
        fold_roc_data = []
        fold_pr_data = []

        for fold in folds:
            print(f"  Fold {fold}...")

            # تقسیم train/test
            train = [t for t in triplets if t["fold"] != fold]
            test = [t for t in triplets if t["fold"] == fold]

            # augmentation روی داده‌های آموزش
            train_aug = augment_train(train)

            # ساخت ویژگی‌ها
            X_train, y_train = make_dataset(train_aug, drug2idx, mats)
            X_test, y_test = make_dataset(test, drug2idx, mats)

            # استانداردسازی
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Grid Search با ۳-fold داخلی
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            gs = GridSearchCV(
                clone(cfg["est"]),
                cfg["grid"],
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train, y_train)

            clf = gs.best_estimator_

            # ذخیره هایپرپارامترهای بهترین مدل
            hyperparams_list.append(
                {
                    "model": name,
                    "fold": fold,
                    **gs.best_params_,
                }
            )

            # پیش‌بینی
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_test)[:, 1]
            else:
                prob = clf.decision_function(X_test)

            pred = clf.predict(X_test)

            # محاسبه معیارها
            auroc = roc_auc_score(y_test, prob)
            auprc = average_precision_score(y_test, prob)
            f1 = f1_score(y_test, pred, zero_division=0)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)

            all_results.append(
                {
                    "model": name,
                    "fold": fold,
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "F1": f1,
                    "Precision": prec,
                    "Recall": rec,
                }
            )

            # ذخیره منحنی‌های ROC و PR
            fpr, tpr, _ = roc_curve(y_test, prob)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob)

            fold_roc_data.append((fpr, tpr))
            fold_pr_data.append((recall_curve, precision_curve))

        # محاسبه میانگین و انحراف معیار برای هر مدل
        model_df = pd.DataFrame([r for r in all_results if r["model"] == name])
        plot_data[name] = {
            "auroc": model_df["AUROC"].mean(),
            "auroc_std": model_df["AUROC"].std(),
            "auprc": model_df["AUPRC"].mean(),
            "auprc_std": model_df["AUPRC"].std(),
            "roc_data": fold_roc_data,
            "pr_data": fold_pr_data,
        }

    # ذخیره فایل‌های CSV
    pd.DataFrame(all_results).to_csv(
        os.path.join(OUT_DIR, "metrics_v7.csv"), index=False
    )
    print("\n✅ metrics_v7.csv saved.")

    pd.DataFrame(hyperparams_list).to_csv(
        os.path.join(OUT_DIR, "best_hyperparams_v7.csv"), index=False
    )
    print("✅ best_hyperparams_v7.csv saved.")

    # رسم نمودارها
    make_plots(plot_data, hyperparams_list, OUT_DIR)
    print("\n🎉 All done! Results saved in:", OUT_DIR)


if __name__ == "__main__":
    run()
