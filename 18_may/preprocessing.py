# remove_duplicates_and_visualize.py
# Run from: /home/faezeh/uni/progect/HyperSynergyX/
# Usage:    python remove_duplicates_and_visualize.py

# we does not need to this script; data is already clean.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# تنظیم ظاهر نمودارها
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# ── مسیرها ──────────────────────────────────────────────────────────────────────
INPUT_DIR = Path("18_may_filtered")
OUTPUT_DIR = Path("18_may_filtered_deduplicated")
OUTPUT_DIR.mkdir(exist_ok=True)

# لیست فایل‌هایی که می‌خواهیم پردازش کنیم
files_to_process = [
    "breast_2drug.tsv",
    "breast_3drug.tsv",
    "lung_2drug.tsv",
    "lung_3drug.tsv",
    "all_2drug.tsv",
    "all_3drug.tsv",
]


# ── تابع تشخیص ستون‌های کلیدی برای تشخیص تکرار ────────────────────────────────────
def get_key_columns(df):
    """ستون‌هایی که برای تشخیص ترکیب یکتای دارویی استفاده می‌شوند"""
    key_cols = []

    # ستون‌های دارو
    drug_cols = [
        "DrugID_1",
        "Drug_Name_1",
        "DrugID_2",
        "Drug_Name_2",
        "DrugID_3",
        "Drug_Name_3",
        "DrugID_4",
        "Drug_Name_4",
    ]

    for col in drug_cols:
        if col in df.columns:
            key_cols.append(col)

    # ستون سلول لاین (اگر وجود داشته باشد)
    if "Cell Line" in df.columns:
        key_cols.append("Cell Line")

    # اگر ستون خاصی برای یکتاسازی وجود دارد
    if "DrugCom_ID" in df.columns:
        key_cols = ["DrugCom_ID"]  # اگر ID یکتا داریم، از آن استفاده کن

    return key_cols


# ── ذخیره آمار برای گزارش ───────────────────────────────────────────────────────
stats = []

# ── پردازش هر فایل ──────────────────────────────────────────────────────────────
for file_name in files_to_process:
    input_path = INPUT_DIR / file_name
    if not input_path.exists():
        print(f"⚠️ File not found: {input_path}")
        continue

    print(f"\n{'='*70}")
    print(f"Processing: {file_name}")
    print("=" * 70)

    # بارگذاری داده
    df = pd.read_csv(input_path, sep="\t", low_memory=False)
    original_count = len(df)
    print(f"Original rows: {original_count:,}")

    # تعیین ستون‌های کلیدی برای حذف تکراری
    key_cols = get_key_columns(df)
    print(f"Key columns for deduplication: {key_cols}")

    # حذف رکوردهای تکراری
    df_deduplicated = df.drop_duplicates(subset=key_cols, keep="first")
    dedup_count = len(df_deduplicated)
    removed_count = original_count - dedup_count

    print(f"After deduplication: {dedup_count:,}")
    print(
        f"Removed duplicates: {removed_count:,} ({100*removed_count/original_count:.2f}%)"
    )

    # ذخیره فایل بدون تکرار
    output_path = OUTPUT_DIR / file_name
    df_deduplicated.to_csv(output_path, sep="\t", index=False)
    print(f"Saved to: {output_path}")

    # ذخیره آمار
    stats.append(
        {
            "File": file_name,
            "Original": original_count,
            "After_Dedup": dedup_count,
            "Removed": removed_count,
            "Removed_Pct": 100 * removed_count / original_count,
        }
    )

# ── نمودار 1: میله‌ای تعداد رکوردهای حذف شده ──────────────────────────────────────
print("\n" + "=" * 70)
print("Generating visualizations...")
print("=" * 70)

stats_df = pd.DataFrame(stats)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# 1. نمودار میله‌ای مقایسه قبل و بعد
ax1 = axes[0]
x = np.arange(len(stats_df))
width = 0.35
ax1.bar(x - width / 2, stats_df["Original"], width, label="Original", color="steelblue")
ax1.bar(
    x + width / 2, stats_df["After_Dedup"], width, label="After Dedup", color="coral"
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
colors = [
    "green" if p < 5 else "orange" if p < 10 else "red" for p in stats_df["Removed_Pct"]
]
bars = ax2.barh(stats_df["File"], stats_df["Removed_Pct"], color=colors)
ax2.set_xlabel("Removed Percentage (%)")
ax2.set_title("Percentage of Duplicates Removed")
for bar, pct in zip(bars, stats_df["Removed_Pct"]):
    ax2.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{pct:.1f}%",
        va="center",
        fontsize=10,
    )

# 3. تعداد رکوردهای حذف شده به صورت عددی
ax3 = axes[2]
bars = ax3.bar(stats_df["File"], stats_df["Removed"], color="tomato")
ax3.set_xlabel("File")
ax3.set_ylabel("Number of Duplicates Removed")
ax3.set_title("Count of Duplicates Removed")
ax3.tick_params(axis="x", rotation=45)
for bar, val in zip(bars, stats_df["Removed"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{int(val)}",
        ha="center",
        fontsize=9,
    )

# 4. نمودار دایره‌ای برای کل داده‌ها (جمع همه فایل‌ها)
ax4 = axes[3]
total_original = stats_df["Original"].sum()
total_dedup = stats_df["After_Dedup"].sum()
total_removed = stats_df["Removed"].sum()
sizes = [total_dedup, total_removed]
labels = [f"Unique\n({total_dedup:,})", f"Duplicates\n({total_removed:,})"]
colors_pie = ["#2ecc71", "#e74c3c"]
ax4.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie, startangle=90)
ax4.set_title(
    f"Total Across All Files\n(Original: {total_original:,} → Unique: {total_dedup:,})"
)

# 5. نمودار مقایسه برای فایل‌های سرطان پستان
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

# 6. نمودار مقایسه برای فایل‌های سرطان ریه
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
plt.savefig(OUTPUT_DIR / "deduplication_analysis.png", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "deduplication_analysis.pdf", bbox_inches="tight")
print(f"✅ Chart saved: {OUTPUT_DIR}/deduplication_analysis.png/pdf")

# ── نمایش آمار در کنسول ─────────────────────────────────────────────────────────
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
print(
    f"{'TOTAL':<25} {total_original:>12,} {total_dedup:>12,} "
    f"{total_removed:>12,} {100*total_removed/total_original:>9.2f}%"
)

# ── ذخیره گزارش آمار به CSV ──────────────────────────────────────────────────────
stats_df.to_csv(OUTPUT_DIR / "deduplication_stats.csv", index=False)
print(f"\n✅ Statistics saved: {OUTPUT_DIR}/deduplication_stats.csv")

# ── نمودار اضافی: توزیع داده‌ها قبل و بعد ────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(stats_df))
width = 0.35
ax.bar(
    x - width / 2,
    stats_df["Original"],
    width,
    label="Original",
    color="steelblue",
    alpha=0.8,
)
ax.bar(
    x + width / 2,
    stats_df["After_Dedup"],
    width,
    label="After Dedup",
    color="coral",
    alpha=0.8,
)
ax.set_xlabel("File")
ax.set_ylabel("Number of Rows (log scale)")
ax.set_title("Deduplication Results (Log Scale)")
ax.set_xticks(x)
ax.set_xticklabels(stats_df["File"], rotation=45, ha="right")
ax.set_yscale("log")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "deduplication_logscale.png", dpi=150, bbox_inches="tight")
print(f"✅ Log-scale chart saved: {OUTPUT_DIR}/deduplication_logscale.png")

print("\n" + "=" * 70)
print("✅ ALL DONE!")
print(f"📁 Deduplicated files saved in: {OUTPUT_DIR}/")
print("=" * 70)
