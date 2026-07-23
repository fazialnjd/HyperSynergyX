import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path(
    "/home/faezeh/uni/progect/HyperSynergyX/2.General Information of Drug Combination.tsv"
)
OUTDIR = Path("18_may_filtered_based_on_article")
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT, sep="\t", low_memory=False)
print(f"Total rows loaded: {len(df):,}")

# ── حذف رکوردهایی که DrugID_4 مقدار دارد ──────────────────────────────────────────
if "DrugID_4" in df.columns:
    mask_has_drug4 = ~(
        df["DrugID_4"].astype(str).str.strip().isin([".", "", "nan", "NaN", "None"])
    )
    print(f"Rows with DrugID_4 value: {mask_has_drug4.sum():,}")
    df = df[~mask_has_drug4].copy()
    print(f"Rows after removing DrugID_4: {len(df):,}")

# ── ستون‌های سینرژی ──────────────────────────────────────────────────────────────
synergy_cols = ["Cell Line", "ZIP", "Bliss", "Loewe", "HSA"]
for col in synergy_cols:
    if col not in df.columns:
        df[col] = np.nan

for col in ["ZIP", "Bliss", "Loewe", "HSA"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace(".", np.nan), errors="coerce")


# ── تابع تشخیص تعداد داروها ──────────────────────────────────────────────────────
def drug_count(row):
    if str(row.get("DrugID_2", ".")).strip() == ".":
        return 1
    if str(row.get("DrugID_3", ".")).strip() == ".":
        return 2
    return 3


df["n_drugs"] = df.apply(drug_count, axis=1)

# حذف تک‌دارویی‌ها
df_valid = df[df["n_drugs"] >= 2].copy()
df_2drug = df_valid[df_valid["n_drugs"] == 2]
df_3drug = df_valid[df_valid["n_drugs"] == 3]

print(f"\n2-drug combinations: {len(df_2drug):,}")
print(f"3-drug combinations: {len(df_3drug):,}")

# ── فیلتر دقیق بر اساس نوع سرطان ──────────────────────────────────────────────────
# عبارات مجاز برای سرطان پستان
breast_keywords = [
    "breast cancer",
    "breast neoplasms",
    "breast adenocarcinoma",
    "breast carcinoma",
]

# عبارات مجاز برای سرطان ریه
lung_keywords = [
    "lung cancer",
    "lung neoplasms",
    "lung adenocarcinoma",
    "lung carcinoma",
]


def exact_cancer_filter(frame, keywords):
    """فیلتر دقیق: فقط رکوردهایی که Disease_Entry دقیقاً با یکی از keywords مطابقت داشته باشد"""
    if frame.empty:
        return frame
    # تبدیل به lower case برای مقایسه یکسان
    pattern = "|".join(keywords)
    return frame[
        frame["Disease_Entry"]
        .str.lower()
        .str.contains(pattern, case=False, na=False, regex=True)
    ]


# فیلترها
breast_2 = exact_cancer_filter(df_2drug, breast_keywords)
breast_3 = exact_cancer_filter(df_3drug, breast_keywords)
lung_2 = exact_cancer_filter(df_2drug, lung_keywords)
lung_3 = exact_cancer_filter(df_3drug, lung_keywords)

# ── گزارش تعداد رکوردهای حذف شده در فیلتر ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Filtering Report:")
print(f"  Breast cancer (exact): {len(breast_2) + len(breast_3):,} rows")
print(f"  Lung cancer (exact):   {len(lung_2) + len(lung_3):,} rows")


# ── نرمال‌سازی نام داروها (تبدیل حرف اول به کوچک) ──────────────────────────────────
def normalize_drug_names(df):
    """تبدیل حرف اول نام داروها به کوچک"""
    for col in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # تبدیل حرف اول به کوچک و بقیه حروف را به همان شکل نگه دار
            df[col] = df[col].apply(lambda x: x[0].lower() + x[1:] if len(x) > 0 else x)
    return df


# اعمال نرمال‌سازی روی همه دیتافریم‌ها
breast_2 = normalize_drug_names(breast_2)
breast_3 = normalize_drug_names(breast_3)
lung_2 = normalize_drug_names(lung_2)
lung_3 = normalize_drug_names(lung_3)
df_2drug = normalize_drug_names(df_2drug)
df_3drug = normalize_drug_names(df_3drug)


# ── ذخیره فایل‌ها ────────────────────────────────────────────────────────────────
for name, data in [
    ("breast_2drug", breast_2),
    ("breast_3drug", breast_3),
    ("lung_2drug", lung_2),
    ("lung_3drug", lung_3),
    ("all_2drug", df_2drug),
    ("all_3drug", df_3drug),
]:
    data.to_csv(OUTDIR / f"{name}.tsv", sep="\t", index=False)
    print(f"Saved: {name}.tsv ({len(data):,} rows)")

# ── گزارش پر بودن ستون‌ها ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Column completeness (sample: breast_2drug):")
for col in synergy_cols:
    if col in breast_2.columns:
        non_null = breast_2[col].notna().sum()
        print(
            f"  {col:15s}: {non_null}/{len(breast_2)} ({100*non_null/len(breast_2):.1f}%)"
            if len(breast_2) > 0
            else f"  {col:15s}: N/A (empty file)"
        )

# ── نمایش نمونه‌هایی از Disease_Entry در فایل‌های فیلتر شده ────────────────────────
print("\n" + "=" * 70)
print("Sample Disease_Entry values in filtered files:")
print("-" * 50)
if len(breast_2) > 0:
    print("Breast cancer samples:", breast_2["Disease_Entry"].unique()[:5])
if len(lung_2) > 0:
    print("Lung cancer samples:  ", lung_2["Disease_Entry"].unique()[:5])


"""

filtered based n name in article:
breast 62
lung 40
"""
