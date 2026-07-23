"""
baseline_models.py
==================
Baseline ML models for drug synergy triplet prediction.
Comparison against HyperSynergyX (AUROC=0.9593, AUPRC=0.9453, F1=0.9507).

Models: Logistic Regression, Random Forest, XGBoost, SVM
Features: 12 pairwise similarity features per triplet (from S_chem, S_atc, S_tgt, S_int)
Protocol: 5-fold stratified CV using split_breast.json

Outputs (saved to baseline_results/):
  baseline_results.csv       — mean±std metrics per model
  01_metrics_comparison.png  — grouped bar chart
  02_roc_curves.png          — ROC curves per model
  03_pr_curves.png           — Precision-Recall curves per model

Run:
    pip install xgboost  # if not installed
    python baseline_models.py

feature matrix:
(A,B,C)

↓


[
chem_AB,
chem_AC,
chem_BC,

atc_AB,
atc_AC,
atc_BC,

tgt_AB,
tgt_AC,
tgt_BC,

int_AB,
int_AC,
int_BC
]

"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    print("WARNING: xgboost not installed. Run: pip install xgboost")
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SPLIT_FILE = Path("split_breast.json")
SIM_DIR = Path("similarity_matrices")
OUTPUT_DIR = Path("baseline_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── HyperSynergyX paper reference numbers ───────────────────────────────────
PAPER = {
    "model": "HyperSynergyX (paper)",
    "AUROC": 0.9593,
    "AUPRC": 0.9453,
    "F1": 0.9507,
    "Precision": None,
    "Recall": None,
}

# ── Style ────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#0279EE",
    "Random Forest": "#75A025",
    "XGBoost": "#FF9400",
    "SVM": "#FD9BED",
    "HyperSynergyX (paper)": "#000000",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})


# ============================================================================
# 1. LOAD DATA
# ============================================================================
def load_data():
    print("=" * 65)
    print("LOADING DATA")
    print("=" * 65)

    # Load split
    with open(SPLIT_FILE) as f:
        items = json.load(f)
    print(f"  split_breast.json: {len(items)} triplets")

    # Load similarity matrices
    drug_names = json.load(open(SIM_DIR / "breast_drug_names.json"))
    drug_idx = {d: i for i, d in enumerate(drug_names)}

    mats = {
        "chem": np.load(SIM_DIR / "breast_S_chem.npy"),
        "atc": np.load(SIM_DIR / "breast_S_atc.npy"),
        "tgt": np.load(SIM_DIR / "breast_S_tgt.npy"),
        "int": np.load(SIM_DIR / "breast_S_int.npy"),
    }
    print(
        f"  Similarity matrices loaded: {list(mats.keys())}, "
        f"shape {next(iter(mats.values())).shape}"
    )

    return items, drug_idx, mats


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
def build_features(items, drug_idx, mats):
    """
    For each triplet (A, B, C) build 12 pairwise similarity features:
      [chem_AB, chem_AC, chem_BC,
       atc_AB,  atc_AC,  atc_BC,
       tgt_AB,  tgt_AC,  tgt_BC,
       int_AB,  int_AC,  int_BC]
    """
    feat_names = []
    for sim in ["chem", "atc", "tgt", "int"]:
        for pair in ["AB", "AC", "BC"]:
            feat_names.append(f"{sim}_{pair}")

    rows, labels, folds = [], [], []
    skipped = 0

    for item in items:
        dA = item["drugA"]
        dB = item["drugB"]
        dC = item["drugC"]

        # Skip if any drug not in similarity matrix
        if any(d not in drug_idx for d in [dA, dB, dC]):
            skipped += 1
            continue

        iA, iB, iC = drug_idx[dA], drug_idx[dB], drug_idx[dC]

        row = []
        for sim_key in ["chem", "atc", "tgt", "int"]:
            M = mats[sim_key]
            row.extend([M[iA, iB], M[iA, iC], M[iB, iC]])

        rows.append(row)
        labels.append(item["label"])
        folds.append(item.get("fold", 0))

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=int)
    folds = np.array(folds, dtype=int)

    if skipped:
        print(f"  WARNING: {skipped} triplets skipped (drug not in similarity matrix)")

    print(
        f"\n  Feature matrix: {X.shape}  |  "
        f"Positive: {y.sum()}  Negative: {(y==0).sum()}"
    )
    print(f"  Features: {feat_names}")
    return X, y, folds, feat_names


# ============================================================================
# 3. MODELS
# ============================================================================
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    return models


# ============================================================================
# 4. CROSS-VALIDATION
# ============================================================================
def run_cv(X, y, folds, models):
    """
    5-fold CV using the fold assignments from split_breast.json.
    Returns per-model dict of per-fold metrics + mean ROC/PR curve data.
    """
    n_folds = len(np.unique(folds))
    print(f"\n{'='*65}")
    print(f"CROSS-VALIDATION  ({n_folds} folds)")
    print(f"{'='*65}")

    all_results = {}

    for model_name, clf in models.items():
        print(f"\n  [{model_name}]")

        fold_metrics = {m: [] for m in ["AUROC", "AUPRC", "F1", "Precision", "Recall"]}
        roc_data, pr_data = [], []

        for fold_id in range(n_folds):
            test_mask = folds == fold_id
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # Scale inside fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            # Metrics
            auroc = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            fold_metrics["AUROC"].append(auroc)
            fold_metrics["AUPRC"].append(auprc)
            fold_metrics["F1"].append(f1)
            fold_metrics["Precision"].append(prec)
            fold_metrics["Recall"].append(rec)

            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data.append((fpr, tpr, auroc))

            # PR curve data
            p, r, _ = precision_recall_curve(y_test, y_prob)
            pr_data.append((r, p, auprc))

            print(
                f"    Fold {fold_id}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                f"F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
            )

        # Summarise
        summary = {}
        for metric, vals in fold_metrics.items():
            summary[f"{metric}_mean"] = np.mean(vals)
            summary[f"{metric}_std"] = np.std(vals)

        print(
            f"    → AUROC={summary['AUROC_mean']:.4f}±{summary['AUROC_std']:.4f}  "
            f"AUPRC={summary['AUPRC_mean']:.4f}±{summary['AUPRC_std']:.4f}  "
            f"F1={summary['F1_mean']:.4f}±{summary['F1_std']:.4f}"
        )

        all_results[model_name] = {
            "summary": summary,
            "roc_data": roc_data,
            "pr_data": pr_data,
        }

    return all_results


# ============================================================================
# 5. SAVE CSV
# ============================================================================
def save_csv(all_results, out_path):
    rows = []

    # Paper reference row
    rows.append(
        {
            "Model": PAPER["model"],
            "AUROC": f"{PAPER['AUROC']:.4f}",
            "AUPRC": f"{PAPER['AUPRC']:.4f}",
            "F1": f"{PAPER['F1']:.4f}",
            "Precision": "—",
            "Recall": "—",
            "AUROC_std": "—",
            "AUPRC_std": "—",
            "F1_std": "—",
            "Precision_std": "—",
            "Recall_std": "—",
        }
    )

    for model_name, res in all_results.items():
        s = res["summary"]
        rows.append(
            {
                "Model": model_name,
                "AUROC": f"{s['AUROC_mean']:.4f}",
                "AUPRC": f"{s['AUPRC_mean']:.4f}",
                "F1": f"{s['F1_mean']:.4f}",
                "Precision": f"{s['Precision_mean']:.4f}",
                "Recall": f"{s['Recall_mean']:.4f}",
                "AUROC_std": f"{s['AUROC_std']:.4f}",
                "AUPRC_std": f"{s['AUPRC_std']:.4f}",
                "F1_std": f"{s['F1_std']:.4f}",
                "Precision_std": f"{s['Precision_std']:.4f}",
                "Recall_std": f"{s['Recall_std']:.4f}",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ============================================================================
# 6. PLOT — METRICS BAR CHART
# ============================================================================
def plot_metrics(all_results, out_path):
    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    model_names = list(all_results.keys())
    all_names = ["HyperSynergyX (paper)"] + model_names

    n_metrics = len(metrics)
    n_models = len(all_names)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, name in enumerate(all_names):
        if name == "HyperSynergyX (paper)":
            vals = [PAPER.get(m) for m in metrics]
            errs = [0] * n_metrics
        else:
            s = all_results[name]["summary"]
            vals = [s[f"{m}_mean"] for m in metrics]
            errs = [s[f"{m}_std"] for m in metrics]

        offset = (i - n_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(name, "#888888")
        hatch = "//" if name == "HyperSynergyX (paper)" else ""

        bars = ax.bar(
            x + offset,
            vals,
            width * 0.9,
            label=name,
            color=color,
            alpha=0.85,
            hatch=hatch,
            edgecolor="white",
            yerr=[e if e > 0 else None for e in errs],
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )

        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Baseline Models vs HyperSynergyX — Breast Cancer\n" "(5-fold CV, mean ± std)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.95, color="gray", ls="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# 7. PLOT — ROC CURVES
# ============================================================================
def plot_roc(all_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.50)")

    # HyperSynergyX reference
    ax.axhline(
        PAPER["AUROC"],
        color=MODEL_COLORS["HyperSynergyX (paper)"],
        ls=":",
        lw=2,
        label=f"HyperSynergyX AUROC={PAPER['AUROC']:.4f} (paper)",
    )

    for model_name, res in all_results.items():
        color = MODEL_COLORS.get(model_name, "#888888")
        mean_auroc = res["summary"]["AUROC_mean"]
        std_auroc = res["summary"]["AUROC_std"]

        # Interpolate all fold ROC curves to common FPR grid, then average
        fpr_grid = np.linspace(0, 1, 200)
        tprs = []
        for fpr, tpr, _ in res["roc_data"]:
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        ax.plot(
            fpr_grid,
            mean_tpr,
            color=color,
            lw=2,
            label=f"{model_name} (AUC={mean_auroc:.4f}±{std_auroc:.4f})",
        )
        ax.fill_between(
            fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.12
        )

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC Curves — Breast Cancer Baselines\n(mean ± std across 5 folds)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# 8. PLOT — PR CURVES
# ============================================================================
def plot_pr(all_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # HyperSynergyX reference
    ax.axhline(
        PAPER["AUPRC"],
        color=MODEL_COLORS["HyperSynergyX (paper)"],
        ls=":",
        lw=2,
        label=f"HyperSynergyX AUPRC={PAPER['AUPRC']:.4f} (paper)",
    )

    for model_name, res in all_results.items():
        color = MODEL_COLORS.get(model_name, "#888888")
        mean_auprc = res["summary"]["AUPRC_mean"]
        std_auprc = res["summary"]["AUPRC_std"]

        # Interpolate all fold PR curves to common recall grid
        recall_grid = np.linspace(0, 1, 200)
        precs = []
        for r, p, _ in res["pr_data"]:
            # PR curve: recall is decreasing — flip for interp
            precs.append(np.interp(recall_grid, r[::-1], p[::-1]))
        mean_prec = np.mean(precs, axis=0)
        std_prec = np.std(precs, axis=0)

        ax.plot(
            recall_grid,
            mean_prec,
            color=color,
            lw=2,
            label=f"{model_name} (AP={mean_auprc:.4f}±{std_auprc:.4f})",
        )
        ax.fill_between(
            recall_grid,
            mean_prec - std_prec,
            mean_prec + std_prec,
            color=color,
            alpha=0.12,
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curves — Breast Cancer Baselines\n"
        "(mean ± std across 5 folds)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # 1. Load
    items, drug_idx, mats = load_data()

    # 2. Features
    print(f"\n{'='*65}")
    print("FEATURE ENGINEERING")
    print(f"{'='*65}")
    X, y, folds, feat_names = build_features(items, drug_idx, mats)

    # 3. Models
    models = get_models()
    print(f"\n  Models: {list(models.keys())}")

    # 4. CV
    all_results = run_cv(X, y, folds, models)

    # 5. Save CSV
    print(f"\n{'='*65}")
    print("SAVING OUTPUTS")
    print(f"{'='*65}")
    df = save_csv(all_results, OUTPUT_DIR / "baseline_results.csv")

    # 6. Plots
    plot_metrics(all_results, OUTPUT_DIR / "01_metrics_comparison.png")
    plot_roc(all_results, OUTPUT_DIR / "02_roc_curves.png")
    plot_pr(all_results, OUTPUT_DIR / "03_pr_curves.png")

    # 7. Print final table
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")
    print(df.to_string(index=False))

    print(f"\n{'='*65}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*65}")
