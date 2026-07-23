# run_eval.py
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from pathlib import Path


def compute_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    threshold = np.median(y_score)
    y_pred = (y_score >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "threshold": threshold,
    }


def evaluate_split(split_json, dataset_name):
    """Evaluate on a split file"""
    with open(split_json, "r") as f:
        items = json.load(f)

    # Load synergy scores
    scores_df = pd.read_csv(
        f"output/{dataset_name}_synergy/synergy_triplets_ranked.csv"
    )

    # Create mapping
    score_map = {}
    for _, row in scores_df.iterrows():
        key = tuple(sorted([row["drug_1"], row["drug_2"], row["drug_3"]]))
        score_map[key] = row["synergy_score"]

    # Get folds
    folds = set(item["fold"] for item in items if "fold" in item)
    folds = sorted(folds)

    results = []
    for fold in folds:
        test_items = [item for item in items if item.get("fold") == fold]
        y_true = []
        y_score = []
        for item in test_items:
            key = tuple(sorted([item["drugA"], item["drugB"], item["drugC"]]))
            y_true.append(item["label"])
            y_score.append(score_map.get(key, 0.0))

        metrics = compute_metrics(y_true, y_score)
        metrics["fold"] = fold
        results.append(metrics)
        print(
            f"Fold {fold}: AUROC={metrics['AUROC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}"
        )

    avg = {
        k: np.mean([r[k] for r in results])
        for k in ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    }
    print(f"\n{'='*50}")
    print(f"AVERAGE for {dataset_name.upper()}:")
    print(f"  AUROC: {avg['AUROC']:.4f}")
    print(f"  AUPRC: {avg['AUPRC']:.4f}")
    print(f"  F1:    {avg['F1']:.4f}")
    return avg


if __name__ == "__main__":
    print("=" * 60)
    print("BREAST CANCER EVALUATION")
    print("=" * 60)
    evaluate_split("split_breast.json", "breast")

    print("\n" + "=" * 60)
    print("LUNG CANCER EVALUATION")
    print("=" * 60)
    evaluate_split("split_lung.json", "lung")


""""
==================================================
AVERAGE for BREAST:
  AUROC: 0.9120
  AUPRC: 0.9211
  F1:    0.8400      ← اینجا هست

==================================================
AVERAGE for LUNG:
  AUROC: 1.0000
  AUPRC: 1.0000
  F1:    1.0000      ← اینجا هست
  lung overfit

"""


# TODO:
# جرا با enrichment pathway که تارگتای مهم رو برمیداره
# بررسی چرا ماتریس coincidence ۴ تا ۱ داره برای برخی درحالی ک تو فایل اصلی نیس
# اجرا با تعداد داده های بیشتر
# درست کردن یه فایل پرزنتیشن قوی با نمودارهای ویژوالیزیشن حالت گزارشای ماشین لرنینگ صنایع کوتاه مختصر مفید
# اجرا با alpha بزرگتر
# داکیومنت با توضیح معماری های بیس لاینها
"""
بیس لاین ها - ولی اینو چک کن انگار از اسمتیاز سینرژی ای ک با روش رندوم والک زدم استفاده کرده

خب اره ب عنوان ی فیچر ورودی دادیم ودیدیم رندوم نیس و معنا داره
وقتی حذفش کردیم دقت خیلی پایین اومده

📊 جدول مقایسه نهایی
روش	AUROC	AUPRC	F1	نتیجه
DBRWH (خودتان)	0.9120	0.9211	0.8400	✅ عالی
LR on DBRWH features	0.9120	0.9211	0.6322	⚠️ F1 پایین‌تر
SVM on DBRWH features	0.8640	0.8431	0.8539	✅ F1 خوب
RF on similarity features	0.6960	0.7255	0.6718	❌ ضعیف

"""
