"""
baseline_models_v4.py
=====================
Baseline ML models for drug synergy triplet prediction.
Comparison against HyperSynergyX (AUROC=0.9593, AUPRC=0.9453, F1=0.9507).

Changes vs v3:
  - FIX: XGBoost uses manual inner CV loop (fixes GridSearchCV compatibility bug
    with XGBoost 2.x — "Got a regressor with response_method=predict_proba")
  - FIX: RF max_depth=None removed from grid (prevents overfitting)
  - FIX: ROC/PR plots skip models with no successful folds (prevents crash)
  - FIX: nan values in bar chart skipped cleanly
  - BP:  Fold variance warning if std(AUROC) > 0.15
  - BP:  inner_AUROC sanity check (warns if < 0.5)
  - BP:  Calibration summary printed per model

Features: pairwise 9-dim (chem_AB/AC/BC, atc_AB/AC/BC, tgt_AB/AC/BC)
Augmentation: train x6 permutations, test x1 (original only — no leakage)
Outer CV: 5-fold from split_breast.json
Inner CV: 3-fold StratifiedKFold GridSearchCV (scoring=AUROC)

Outputs (saved to baseline_results/):
  baseline_results.csv
  best_hyperparams.csv
  01_metrics_comparison.png
  02_roc_curves.png
  03_pr_curves.png

Run:
  pip install xgboost  # optional
  python baseline_models_v4.py
"""

import itertools
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations

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

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLIT_FILE = Path("split_breast.json")
SIM_DIR = Path("similarity_matrices")
OUTPUT_DIR = Path("baseline_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── HyperSynergyX reference ───────────────────────────────────────────────────
PAPER = {
    "model": "HyperSynergyX (paper)",
    "AUROC": 0.9593,
    "AUPRC": 0.9453,
    "F1": 0.9507,
    "Precision": None,
    "Recall": None,
}

# ── Thresholds ────────────────────────────────────────────────────────────────
AUROC_STD_WARN = 0.15
INNER_AUROC_WARN = 0.50

# ── Style ─────────────────────────────────────────────────────────────────────
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
    }
    print(
        f"  Similarity matrices: {list(mats.keys())}, "
        f"shape {next(iter(mats.values())).shape}"
    )
    return items, drug_idx, mats


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
def make_feature_vector(iA, iB, iC, mats):
    """
    9-dim pairwise feature برای ترتیب (A, B, C):
      [chem_AB, chem_AC, chem_BC,
       atc_AB,  atc_AC,  atc_BC,
       tgt_AB,  tgt_AC,  tgt_BC]
    """
    row = []
    for key in ["chem", "atc", "tgt"]:
        M = mats[key]
        row.extend([M[iA, iB], M[iA, iC], M[iB, iC]])
    return row


def build_features(items, drug_idx, mats):
    feat_names = [
        f"{s}_{p}" for s in ["chem", "atc", "tgt"] for p in ["AB", "AC", "BC"]
    ]
    triplets = []
    skipped = 0

    for item in items:
        dA, dB, dC = item["drugA"], item["drugB"], item["drugC"]
        if any(d not in drug_idx for d in [dA, dB, dC]):
            skipped += 1
            continue
        triplets.append(
            (
                drug_idx[dA],
                drug_idx[dB],
                drug_idx[dC],
                item["label"],
                item.get("fold", 0),
            )
        )

    if skipped:
        print(f"  WARNING: {skipped} triplets skipped (drug not in matrix)")

    print(f"\n  Triplets loaded : {len(triplets)}")
    print(f"  Features ({len(feat_names)}): {feat_names}")
    return triplets, feat_names


def augment_dataset(triplets, mats):
    """
    Train dataset: هر triplet -> همه 6 جایگشت
    ترتیب جایگشت‌ها deterministic (itertools.permutations) -> reproducible
    """
    X, y, folds = [], [], []
    for iA, iB, iC, label, fold in triplets:
        for perm in permutations([iA, iB, iC]):
            X.append(make_feature_vector(*perm, mats))
            y.append(label)
            folds.append(fold)
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=int),
        np.array(folds, dtype=int),
    )


def original_dataset(triplets, mats):
    """
    Test dataset: فقط جایگشت اصلی (A,B,C) — بدون augmentation
    """
    X, y, folds = [], [], []
    for iA, iB, iC, label, fold in triplets:
        X.append(make_feature_vector(iA, iB, iC, mats))
        y.append(label)
        folds.append(fold)
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=int),
        np.array(folds, dtype=int),
    )


# ============================================================================
# 3. MODEL CONFIGS
# ============================================================================
def get_model_configs():
    configs = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    configs["Logistic Regression"] = {
        "type": "pipeline",
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
            "clf__C": [0.01, 0.1, 1.0, 5.0],
            "clf__penalty": ["l2"],
        },
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    # max_depth=None حذف شد — با dataset کوچیک همیشه overfit می‌کنه
    configs["Random Forest"] = {
        "type": "pipeline",
        "pipeline": Pipeline(
            [
                ("scaler", StandardScaler()),
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
            "clf__max_depth": [2, 3, 4],
            "clf__min_samples_leaf": [1, 3, 5],
        },
    }

    # ── SVM ──────────────────────────────────────────────────────────────────
    configs["SVM"] = {
        "type": "pipeline",
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
            "clf__C": [0.01, 0.1, 1.0, 5.0],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
    }

    # ── XGBoost ──────────────────────────────────────────────────────────────
    # type="raw": manual inner CV — fix برای XGBoost 2.x GridSearchCV bug
    if HAS_XGB:
        configs["XGBoost"] = {
            "type": "raw",
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [2, 3],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        }

    return configs


# ============================================================================
# 4. NESTED CV
# ============================================================================
def fit_and_predict(cfg, X_train, y_train, X_test, inner_cv):
    """
    Pipeline models (LR, RF, SVM): GridSearchCV روی sklearn Pipeline
    Raw models (XGBoost): manual inner CV loop — سازگار با همه نسخه‌های XGBoost
    Returns: y_prob, best_params, best_inner_score
    """
    if cfg["type"] == "pipeline":
        gs = GridSearchCV(
            estimator=cfg["pipeline"],
            param_grid=cfg["param_grid"],
            cv=inner_cv,
            scoring="roc_auc",
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )
        gs.fit(X_train, y_train)
        y_prob = gs.predict_proba(X_test)[:, 1]
        best_params = gs.best_params_
        best_inner = gs.best_score_

    else:
        # ── XGBoost: manual grid search + inner CV ───────────────────────────
        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(X_train)
        Xte_sc = scaler.transform(X_test)

        keys = list(cfg["param_grid"].keys())
        combos = list(itertools.product(*cfg["param_grid"].values()))

        best_score = -1.0
        best_params = {}
        best_model = None

        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

        for combo in combos:
            params = dict(zip(keys, combo))
            fold_scores = []

            for tr_idx, val_idx in inner_skf.split(Xtr_sc, y_train):
                Xi_tr = Xtr_sc[tr_idx]
                yi_tr = y_train[tr_idx]
                Xi_val = Xtr_sc[val_idx]
                yi_val = y_train[val_idx]
                try:
                    mdl = XGBClassifier(
                        objective="binary:logistic",
                        use_label_encoder=False,
                        eval_metric="logloss",
                        verbosity=0,
                        random_state=42,
                        **params,
                    )
                    mdl.fit(Xi_tr, yi_tr)
                    prob = mdl.predict_proba(Xi_val)[:, 1]
                    fold_scores.append(roc_auc_score(yi_val, prob))
                except Exception:
                    fold_scores.append(0.0)

            mean_s = float(np.mean(fold_scores))
            if mean_s > best_score:
                best_score = mean_s
                best_params = params
                # refit روی کل train با بهترین params
                best_model = XGBClassifier(
                    objective="binary:logistic",
                    use_label_encoder=False,
                    eval_metric="logloss",
                    verbosity=0,
                    random_state=42,
                    **params,
                )
                best_model.fit(Xtr_sc, y_train)

        y_prob = best_model.predict_proba(Xte_sc)[:, 1]
        best_inner = best_score

    return y_prob, best_params, best_inner


def run_nested_cv(X_aug, y_aug, fold_aug, X_orig, y_orig, fold_orig, configs):
    n_folds = len(np.unique(fold_orig))
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    print(f"\n{'='*65}")
    print(f"NESTED CV (outer={n_folds}-fold, inner=3-fold GridSearch, scoring=AUROC)")
    print(f"{'='*65}")

    all_results = {}
    all_params = []

    for model_name, cfg in configs.items():
        print(f"\n  [{model_name}]")

        fold_metrics = {m: [] for m in ["AUROC", "AUPRC", "F1", "Precision", "Recall"]}
        roc_data, pr_data = [], []
        all_y_true, all_y_prob = [], []

        for fold_id in range(n_folds):
            train_mask = fold_aug != fold_id
            test_mask = fold_orig == fold_id

            X_train = X_aug[train_mask]
            y_train = y_aug[train_mask]
            X_test = X_orig[test_mask]
            y_test = y_orig[test_mask]

            try:
                y_prob, best_params, best_inner = fit_and_predict(
                    cfg, X_train, y_train, X_test, inner_cv
                )
            except Exception as e:
                print(f"    ERROR fold {fold_id}: {e}")
                continue

            if best_inner < INNER_AUROC_WARN:
                print(
                    f"    WARNING fold {fold_id}: inner_AUROC={best_inner:.4f} < "
                    f"{INNER_AUROC_WARN} — GridSearch may have failed"
                )

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

            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(y_prob.tolist())

            clean = {k: v for k, v in best_params.items()}
            print(
                f"    Fold {fold_id}: AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                f"F1={f1:.4f} | inner_AUROC={best_inner:.4f} best={clean}"
            )

            row = {
                "model": model_name,
                "fold": fold_id,
                "outer_AUROC": auroc,
                "inner_AUROC": best_inner,
            }
            row.update(clean)
            all_params.append(row)

        # ── Summary ──────────────────────────────────────────────────────────
        summary = {}
        for metric, vals in fold_metrics.items():
            summary[f"{metric}_mean"] = np.mean(vals) if vals else float("nan")
            summary[f"{metric}_std"] = np.std(vals) if vals else float("nan")

        print(
            f"    -> AUROC={summary['AUROC_mean']:.4f}+-{summary['AUROC_std']:.4f} "
            f"AUPRC={summary['AUPRC_mean']:.4f}+-{summary['AUPRC_std']:.4f} "
            f"F1={summary['F1_mean']:.4f}+-{summary['F1_std']:.4f}"
        )

        if not np.isnan(summary["AUROC_std"]) and summary["AUROC_std"] > AUROC_STD_WARN:
            print(
                f"    WARNING: AUROC std={summary['AUROC_std']:.4f} > "
                f"{AUROC_STD_WARN} — high fold-to-fold variance (small dataset)"
            )

        if all_y_prob:
            arr_true = np.array(all_y_true)
            arr_prob = np.array(all_y_prob)
            print(
                f"    Calibration: mean_pred={arr_prob.mean():.3f} vs "
                f"mean_true={arr_true.mean():.3f} "
                f"(diff={abs(arr_prob.mean()-arr_true.mean()):.3f})"
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

        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else "nan"

        rows.append(
            {
                "Model": model_name,
                "AUROC": fmt(s["AUROC_mean"]),
                "AUPRC": fmt(s["AUPRC_mean"]),
                "F1": fmt(s["F1_mean"]),
                "Precision": fmt(s["Precision_mean"]),
                "Recall": fmt(s["Recall_mean"]),
                "AUROC_std": fmt(s["AUROC_std"]),
                "AUPRC_std": fmt(s["AUPRC_std"]),
                "F1_std": fmt(s["F1_std"]),
                "Precision_std": fmt(s["Precision_std"]),
                "Recall_std": fmt(s["Recall_std"]),
            }
        )
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

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
            errs = [0.0] * n_metrics
        else:
            s = all_results[name]["summary"]
            vals = [s[f"{m}_mean"] for m in metrics]
            errs = [s[f"{m}_std"] for m in metrics]

        offset = (i - n_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(name, "#888888")
        hatch = "//" if name == "HyperSynergyX (paper)" else ""

        for j, (v, e) in enumerate(zip(vals, errs)):
            # nan یا None رو skip کن
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            yerr_val = [[e], [e]] if (e and not np.isnan(e) and e > 0) else None
            ax.bar(
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
        "(nested 5-fold CV, mean+-std | pairwise 9-dim features + permutation augmentation)",
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
        if not res["roc_data"]:
            print(f"  SKIP {model_name}: no successful folds for ROC plot")
            continue
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
            label=f"{model_name} (AUC={mean_auroc:.4f}+-{std_auroc:.4f})",
        )
        ax.fill_between(
            fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.12
        )

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(
        "ROC Curves — Breast Cancer Baselines\n"
        "(mean+-std, nested CV | pairwise features + permutation augmentation)",
        fontsize=11,
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
        if not res["pr_data"]:
            print(f"  SKIP {model_name}: no successful folds for PR plot")
            continue
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
            label=f"{model_name} (AP={mean_auprc:.4f}+-{std_auprc:.4f})",
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
        "(mean+-std, nested CV | pairwise features + permutation augmentation)",
        fontsize=11,
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
    triplets, feat_names = build_features(items, drug_idx, mats)

    # 3. Augmentation
    X_aug, y_aug, fold_aug = augment_dataset(triplets, mats)
    X_orig, y_orig, fold_orig = original_dataset(triplets, mats)

    print(
        f"\n  Augmented (train) : {X_aug.shape}  "
        f"({len(triplets)} triplets x 6 permutations)"
    )
    print(f"  Original  (test)  : {X_orig.shape}  " f"({len(triplets)} triplets x 1)")
    print(
        f"  Train/feature ratio per fold: "
        f"{int(len(triplets)*0.8)*6}/{len(feat_names)} = "
        f"{int(len(triplets)*0.8)*6/len(feat_names):.1f}"
    )

    # 4. Model configs
    configs = get_model_configs()
    print(f"\n  Models: {list(configs.keys())}")
    for name, cfg in configs.items():
        n = 1
        for v in cfg["param_grid"].values():
            n *= len(v)
        print(f"    {name}: {n} combinations x 3 inner folds")

    # 5. Nested CV
    all_results, all_params = run_nested_cv(
        X_aug, y_aug, fold_aug, X_orig, y_orig, fold_orig, configs
    )

    # 6. Save outputs
    print(f"\n{'='*65}")
    print("SAVING OUTPUTS")
    print(f"{'='*65}")
    df_metrics, df_params = save_csv(
        all_results,
        all_params,
        OUTPUT_DIR / "baseline_results.csv",
        OUTPUT_DIR / "best_hyperparams.csv",
    )
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

""""

مدل	AUROC	AUPRC	F1
HyperSynergyX (paper)	0.9593	0.9453	0.9507
SVM	0.776±0.178	0.860±0.102	0.763±0.193
XGBoost	0.760±0.080	0.790±0.117	0.700±0.066
Random Forest	0.720±0.091	0.769±0.100	0.630±0.118
Logistic Regression	0.720±0.091	0.755±0.087	0.610±0.176

"""
