"""
baseline_models.py
==================
Baseline ML models for drug synergy triplet prediction.
Comparison against HyperSynergyX (AUROC=0.9593, AUPRC=0.9453, F1=0.9507).

Models: Logistic Regression, Random Forest, XGBoost, SVM
Features: 4 mean-aggregated similarity features per triplet
          mean_chem/atc/tgt/int = mean of 3 pairwise similarities (AB, AC, BC)
          Rationale: 50 samples total → 4 features gives 10 samples/feature (safe threshold)

Hyperparameter tuning: Nested CV (inner 3-fold GridSearchCV on train set per outer fold)
                       Scoring: AUROC — best params selected per fold, reported in output
Outer CV: 5-fold stratified (from split_breast.json)

Outputs (saved to baseline_results/):
  baseline_results.csv         — mean±std metrics per model
  best_hyperparams.csv         — best hyperparams found per model per fold
  01_metrics_comparison.png    — grouped bar chart
  02_roc_curves.png            — ROC curves per model
  03_pr_curves.png             — Precision-Recall curves per model

Run:
    pip install xgboost  # if not installed
    python baseline_models.py
feature matrix = Triplet
mean_chem   mean_atc   mean_tgt   mean_int
------------------------------------------------------------
(A,B,C)            0.666       0.42       0.71       0.55
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

# ── HyperSynergyX paper reference ───────────────────────────────────────────
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

    with open(SPLIT_FILE) as f:
        items = json.load(f)
    print(f"  split_breast.json : {len(items)} triplets")

    drug_names = json.load(open(SIM_DIR / "breast_drug_names.json"))
    drug_idx = {d: i for i, d in enumerate(drug_names)}

    mats = {
        "chem": np.load(SIM_DIR / "breast_S_chem.npy"),
        "atc": np.load(SIM_DIR / "breast_S_atc.npy"),
        "tgt": np.load(SIM_DIR / "breast_S_tgt.npy"),
        "int": np.load(SIM_DIR / "breast_S_int.npy"),
    }
    print(
        f"  Similarity matrices: {list(mats.keys())}, "
        f"shape {next(iter(mats.values())).shape}"
    )
    return items, drug_idx, mats


# ============================================================================
# 2. FEATURE ENGINEERING  (4 mean-aggregated features)
# ============================================================================
def build_features(items, drug_idx, mats):
    """
    4 features per triplet (A, B, C):
      mean_chem = mean(S_chem[A,B], S_chem[A,C], S_chem[B,C])
      mean_atc  = mean(S_atc[A,B],  S_atc[A,C],  S_atc[B,C])
      mean_tgt  = mean(S_tgt[A,B],  S_tgt[A,C],  S_tgt[B,C])
      mean_int  = mean(S_int[A,B],  S_int[A,C],  S_int[B,C])
    """
    feat_names = ["mean_chem", "mean_atc", "mean_tgt", "mean_int"]
    rows, labels, folds = [], [], []
    skipped = 0

    for item in items:
        dA, dB, dC = item["drugA"], item["drugB"], item["drugC"]
        if any(d not in drug_idx for d in [dA, dB, dC]):
            skipped += 1
            continue
        iA, iB, iC = drug_idx[dA], drug_idx[dB], drug_idx[dC]
        row = []
        for key in ["chem", "atc", "tgt", "int"]:
            M = mats[key]
            row.append((M[iA, iB] + M[iA, iC] + M[iB, iC]) / 3.0)
        rows.append(row)
        labels.append(item["label"])
        folds.append(item.get("fold", 0))

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=int)
    folds = np.array(folds, dtype=int)

    if skipped:
        print(f"  WARNING: {skipped} triplets skipped (drug not in matrix)")

    n_train_per_fold = int(len(rows) * 0.8)
    print(f"\n  Feature matrix      : {X.shape}")
    print(f"  Positive / Negative : {y.sum()} / {(y==0).sum()}")
    print(f"  Features            : {feat_names}")
    print(
        f"  Samples/feature     : {n_train_per_fold/len(feat_names):.1f} "
        f"(train per fold / n_features)"
    )
    return X, y, folds, feat_names


# ============================================================================
# 3. MODEL + PARAM GRID DEFINITIONS
# ============================================================================
def get_model_configs():
    """
    Returns dict of:
      model_name -> {"pipeline": Pipeline, "param_grid": dict}

    Each pipeline has two steps:
      scaler  : StandardScaler  (fit on train, transform test — no leakage)
      clf     : the classifier

    Param grid keys use the "clf__" prefix (sklearn Pipeline convention).
    Grid is intentionally compact: with 40 train samples and inner 3-fold CV
    (≈27 inner-train samples), large grids overfit the inner loop.
    """
    configs = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    configs["Logistic Regression"] = {
        "pipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "param_grid": {
            "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0],
            "clf__penalty": ["l2"],  # l1 needs solver change; keep simple
        },
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    configs["Random Forest"] = {
        "pipeline": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),  # RF doesn't need scaling but keeps API uniform
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced", random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [2, 3, 4, None],
            "clf__min_samples_leaf": [1, 3, 5],
        },
    }

    # ── SVM ──────────────────────────────────────────────────────────────────
    configs["SVM"] = {
        "pipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(probability=True, class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "param_grid": {
            "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
    }

    # ── XGBoost ──────────────────────────────────────────────────────────────
    if HAS_XGB:
        configs["XGBoost"] = {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        XGBClassifier(
                            random_state=42, eval_metric="logloss", verbosity=0
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample": [0.8, 1.0],
            },
        }

    return configs


# ============================================================================
# 4. NESTED CV
# ============================================================================
def run_nested_cv(X, y, folds, configs):
    """
    Outer loop : 5-fold from split_breast.json (fold field)
    Inner loop : 3-fold StratifiedKFold GridSearchCV on train set
                 scoring = AUROC (roc_auc)

    For each outer fold:
      1. Fit GridSearchCV on X_train → finds best hyperparams
      2. Refit best estimator on full X_train
      3. Evaluate on X_test

    Returns per-model results dict + best_params log.
    """
    n_folds = len(np.unique(folds))
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    print(f"\n{'='*65}")
    print(f"NESTED CV  (outer={n_folds}-fold, inner=3-fold GridSearch, scoring=AUROC)")
    print(f"{'='*65}")

    all_results = {}
    all_params = []  # for best_hyperparams.csv

    for model_name, cfg in configs.items():
        print(f"\n  [{model_name}]")

        fold_metrics = {m: [] for m in ["AUROC", "AUPRC", "F1", "Precision", "Recall"]}
        roc_data, pr_data = [], []
        best_params_per_fold = []

        for fold_id in range(n_folds):
            test_mask = folds == fold_id
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            # ── Inner GridSearchCV ───────────────────────────────────────────
            gs = GridSearchCV(
                estimator=cfg["pipeline"],
                param_grid=cfg["param_grid"],
                cv=inner_cv,
                scoring="roc_auc",
                refit=True,  # refit best on full X_train
                n_jobs=-1,
                error_score=0.0,  # don't crash on degenerate folds
            )
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            best_inner_score = gs.best_score_
            best_params_per_fold.append(best_params)

            # ── Evaluate on outer test fold ──────────────────────────────────
            y_prob = gs.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

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

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data.append((fpr, tpr, auroc))
            p, r, _ = precision_recall_curve(y_test, y_prob)
            pr_data.append((r, p, auprc))

            # Pretty-print best params (strip "clf__" prefix for readability)
            clean = {k.replace("clf__", ""): v for k, v in best_params.items()}
            print(
                f"    Fold {fold_id}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                f"F1={f1:.4f}  |  inner_AUROC={best_inner_score:.4f}  "
                f"best={clean}"
            )

            # Log for CSV
            row = {
                "model": model_name,
                "fold": fold_id,
                "outer_AUROC": auroc,
                "inner_AUROC": best_inner_score,
            }
            row.update(clean)
            all_params.append(row)

        # ── Summarise across folds ───────────────────────────────────────────
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

    return all_results, all_params


# ============================================================================
# 5. SAVE CSV
# ============================================================================
def save_csv(all_results, all_params, results_path, params_path):
    # ── Metrics table ────────────────────────────────────────────────────────
    rows = [
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
    ]
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
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # ── Best hyperparams table ───────────────────────────────────────────────
    df_params = pd.DataFrame(all_params)
    df_params.to_csv(params_path, index=False)
    print(f"  Saved: {params_path}")

    return df_metrics, df_params


# ============================================================================
# 6. PLOT — METRICS BAR CHART
# ============================================================================
def plot_metrics(all_results, out_path):
    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    all_names = ["HyperSynergyX (paper)"] + list(all_results.keys())
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

        # Plot each bar individually so None values are skipped cleanly
        for j, (v, e) in enumerate(zip(vals, errs)):
            if v is None:
                continue  # skip metrics not reported for this model (e.g. paper has no P/R)
            yerr_val = [[e], [e]] if e and e > 0 else None
            bar = ax.bar(
                x[j] + offset,
                v,
                width * 0.9,
                color=color,
                alpha=0.85,
                hatch=hatch,
                edgecolor="white",
                yerr=yerr_val,
                capsize=3,
                error_kw={"elinewidth": 1.2, "ecolor": "black"},
                label=name if j == 0 else "_nolegend_",
            )
            ax.text(
                x[j] + offset,
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Baseline Models vs HyperSynergyX — Breast Cancer\n"
        "(nested 5-fold CV, mean ± std | n=50 samples, 4 features, inner GridSearch)",
        fontsize=11,
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
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.50)")
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
        fpr_grid = np.linspace(0, 1, 200)
        tprs = [np.interp(fpr_grid, fpr, tpr) for fpr, tpr, _ in res["roc_data"]]
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
        "ROC Curves — Breast Cancer Baselines\n"
        "(mean ± std, nested CV with inner GridSearch)",
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
        recall_grid = np.linspace(0, 1, 200)
        precs = [np.interp(recall_grid, r[::-1], p[::-1]) for r, p, _ in res["pr_data"]]
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
        "(mean ± std, nested CV with inner GridSearch)",
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

    # 3. Model configs (pipeline + param grid)
    configs = get_model_configs()
    print(f"\n  Models to tune: {list(configs.keys())}")
    for name, cfg in configs.items():
        n_combos = 1
        for v in cfg["param_grid"].values():
            n_combos *= len(v)
        print(f"    {name}: {n_combos} hyperparameter combinations × 3 inner folds")

    # 4. Nested CV
    all_results, all_params = run_nested_cv(X, y, folds, configs)

    # 5. Save outputs
    print(f"\n{'='*65}")
    print("SAVING OUTPUTS")
    print(f"{'='*65}")
    df_metrics, df_params = save_csv(
        all_results,
        all_params,
        OUTPUT_DIR / "baseline_results.csv",
        OUTPUT_DIR / "best_hyperparams.csv",
    )

    # 6. Plots
    plot_metrics(all_results, OUTPUT_DIR / "01_metrics_comparison.png")
    plot_roc(all_results, OUTPUT_DIR / "02_roc_curves.png")
    plot_pr(all_results, OUTPUT_DIR / "03_pr_curves.png")

    # 7. Print summary
    print(f"\n{'='*65}")
    print("RESULTS SUMMARY")
    print(f"{'='*65}")
    print(df_metrics.to_string(index=False))

    print(f"\n{'='*65}")
    print("BEST HYPERPARAMETERS (per model per fold)")
    print(f"{'='*65}")
    print(df_params.to_string(index=False))

    print(f"\n{'='*65}")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*65}")
