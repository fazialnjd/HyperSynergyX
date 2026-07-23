import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# تنظیمات اولیه
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ── مسیرها ──────────────────────────────────────────────────────────────────────
INPUT = Path(
    "/home/faezeh/uni/progect/HyperSynergyX/2.General Information of Drug Combination.tsv"
)
OUTDIR = Path("18_may_filtered")
OUTDIR.mkdir(exist_ok=True)

# ── بارگذاری داده ─────────────────────────────────────────────────────────────────
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


# ── فیلتر بر اساس نوع سرطان ──────────────────────────────────────────────────────
def cancer_filter(frame, keyword):
    return frame[frame["Disease_Entry"].str.contains(keyword, case=False, na=False)]


breast_2 = cancer_filter(df_2drug, "breast")
breast_3 = cancer_filter(df_3drug, "breast")
lung_2 = cancer_filter(df_2drug, "lung")
lung_3 = cancer_filter(df_3drug, "lung")

# ── ذخیره فایل‌های اصلی ──────────────────────────────────────────────────────────
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
        )

# ────────────────────────────────────────────────────────────────────────────────
# ── بخش دوم: حذف رکوردهای تکراری و رسم نمودار ──────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Checking for duplicates and generating visualizations...")
print("=" * 70)

OUTDIR_DEDUP = Path("18_may_filtered_deduplicated")
OUTDIR_DEDUP.mkdir(exist_ok=True)

# لیست فایل‌هایی که می‌خواهیم پردازش کنیم
files_to_process = [
    "breast_2drug.tsv",
    "breast_3drug.tsv",
    "lung_2drug.tsv",
    "lung_3drug.tsv",
    "all_2drug.tsv",
    "all_3drug.tsv",
]

# ذخیره آمار برای گزارش
stats = []

for file_name in files_to_process:
    input_path = OUTDIR / file_name
    if not input_path.exists():
        print(f"⚠️ File not found: {input_path}")
        continue

    print(f"\nProcessing: {file_name}")
    df_temp = pd.read_csv(input_path, sep="\t", low_memory=False)
    original_count = len(df_temp)

    # حذف رکوردهای تکراری بر اساس ستون DrugCom_ID (یا همه ستون‌ها)
    if "DrugCom_ID" in df_temp.columns:
        df_dedup = df_temp.drop_duplicates(subset=["DrugCom_ID"], keep="first")
    else:
        df_dedup = df_temp.drop_duplicates()

    dedup_count = len(df_dedup)
    removed_count = original_count - dedup_count

    print(
        f"  Original: {original_count:,}, After dedup: {dedup_count:,}, Removed: {removed_count:,}"
    )

    # ذخیره فایل بدون تکرار
    df_dedup.to_csv(OUTDIR_DEDUP / file_name, sep="\t", index=False)

    stats.append(
        {
            "File": file_name,
            "Original": original_count,
            "After_Dedup": dedup_count,
            "Removed": removed_count,
            "Removed_Pct": (
                100 * removed_count / original_count if original_count > 0 else 0
            ),
        }
    )

# ── رسم نمودارها ─────────────────────────────────────────────────────────────────
stats_df = pd.DataFrame(stats)

if len(stats_df) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 1. نمودار میله‌ای مقایسه قبل و بعد
    ax1 = axes[0]
    x = np.arange(len(stats_df))
    width = 0.35
    ax1.bar(
        x - width / 2, stats_df["Original"], width, label="Original", color="steelblue"
    )
    ax1.bar(
        x + width / 2,
        stats_df["After_Dedup"],
        width,
        label="After Dedup",
        color="coral",
    )
    ax1.set_xlabel("File")
    ax1.set_ylabel("Number of Rows")
    ax1.set_title("Before vs After Deduplication")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["File"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. درصد رکوردهای حذف شده
    ax2 = axes[1]
    bars = ax2.barh(stats_df["File"], stats_df["Removed_Pct"], color="orange")
    ax2.set_xlabel("Removed Percentage (%)")
    ax2.set_title("Percentage of Duplicates Removed")
    for bar, pct in zip(bars, stats_df["Removed_Pct"]):
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%",
            va="center",
            fontsize=9,
        )

    # 3. تعداد رکوردهای حذف شده
    ax3 = axes[2]
    bars = ax3.bar(stats_df["File"], stats_df["Removed"], color="tomato")
    ax3.set_xlabel("File")
    ax3.set_ylabel("Number of Duplicates Removed")
    ax3.set_title("Count of Duplicates Removed")
    ax3.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, stats_df["Removed"]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(val)}",
            ha="center",
            fontsize=9,
        )

    # 4. نمودار دایره‌ای کل
    ax4 = axes[3]
    total_original = stats_df["Original"].sum()
    total_dedup = stats_df["After_Dedup"].sum()
    total_removed = stats_df["Removed"].sum()
    sizes = [total_dedup, total_removed]
    labels = [f"Unique\n({total_dedup:,})", f"Duplicates\n({total_removed:,})"]
    ax4.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    ax4.set_title(
        f"Total Across All Files\n(Original: {total_original:,} → Unique: {total_dedup:,})"
    )

    # 5. نمودار سرطان پستان
    ax5 = axes[4]
    breast_stats = stats_df[stats_df["File"].str.contains("breast")]
    if len(breast_stats) > 0:
        x = np.arange(len(breast_stats))
        ax5.bar(
            x - width / 2,
            breast_stats["Original"],
            width,
            label="Original",
            color="steelblue",
        )
        ax5.bar(
            x + width / 2,
            breast_stats["After_Dedup"],
            width,
            label="After Dedup",
            color="coral",
        )
        ax5.set_xlabel("Breast Cancer Files")
        ax5.set_ylabel("Number of Rows")
        ax5.set_title("Breast Cancer Dataset")
        ax5.set_xticks(x)
        ax5.set_xticklabels(breast_stats["File"], rotation=45, ha="right")
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "No breast cancer files", ha="center", va="center")

    # 6. نمودار سرطان ریه
    ax6 = axes[5]
    lung_stats = stats_df[stats_df["File"].str.contains("lung")]
    if len(lung_stats) > 0:
        x = np.arange(len(lung_stats))
        ax6.bar(
            x - width / 2,
            lung_stats["Original"],
            width,
            label="Original",
            color="steelblue",
        )
        ax6.bar(
            x + width / 2,
            lung_stats["After_Dedup"],
            width,
            label="After Dedup",
            color="coral",
        )
        ax6.set_xlabel("Lung Cancer Files")
        ax6.set_ylabel("Number of Rows")
        ax6.set_title("Lung Cancer Dataset")
        ax6.set_xticks(x)
        ax6.set_xticklabels(lung_stats["File"], rotation=45, ha="right")
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, "No lung cancer files", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(
        OUTDIR_DEDUP / "deduplication_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(OUTDIR_DEDUP / "deduplication_analysis.pdf", bbox_inches="tight")
    print(f"\n✅ Charts saved: {OUTDIR_DEDUP}/deduplication_analysis.png/pdf")

    # ── ذخیره آمار ────────────────────────────────────────────────────────────────
    stats_df.to_csv(OUTDIR_DEDUP / "deduplication_stats.csv", index=False)
    print(f"✅ Statistics saved: {OUTDIR_DEDUP}/deduplication_stats.csv")

    # ── نمایش آمار ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY - DUPLICATE REMOVAL REPORT")
    print("=" * 70)
    print(
        f"\n{'File':<25} {'Original':>12} {'After Dedup':>12} {'Removed':>12} {'Removed %':>10}"
    )
    print("-" * 75)
    for _, row in stats_df.iterrows():
        print(
            f"{row['File']:<25} {row['Original']:>12,} {row['After_Dedup']:>12,} "
            f"{row['Removed']:>12,} {row['Removed_Pct']:>9.2f}%"
        )
    print("-" * 75)
    total_original = stats_df["Original"].sum()
    total_dedup = stats_df["After_Dedup"].sum()
    total_removed = stats_df["Removed"].sum()
    print(
        f"{'TOTAL':<25} {total_original:>12,} {total_dedup:>12,} "
        f"{total_removed:>12,} {100*total_removed/total_original:>9.2f}%"
    )
else:
    print("No data to visualize.")

print("\n" + "=" * 70)
print("✅ ALL DONE!")
print(f"📁 Filtered files: {OUTDIR}/")
print(f"📁 Deduplicated files: {OUTDIR_DEDUP}/")
print("=" * 70)

"""
generally:
breast 65
lung 50
"""
