# filter_drug_combinations.py
# Run from: /home/faezeh/uni/progect/HyperSynergyX/
# Usage:    python filter_drug_combinations.py

import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT = Path("2.General Information of Drug Combination.tsv")
OUTDIR = Path("filtered")
OUTDIR.mkdir(exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, sep="\t", low_memory=False)
print(f"Total rows loaded : {len(df):,}")

# ── classify drug count ────────────────────────────────────────────────────────
# A "." in DrugID_2 → only 1 drug  (remove)
# A "." in DrugID_3 → exactly 2 drugs
# A real value in DrugID_3 → at least 3 drugs


def drug_count(row):
    if str(row["DrugID_2"]).strip() == ".":
        return 1
    if str(row["DrugID_3"]).strip() == ".":
        return 2
    return 3


df["n_drugs"] = df.apply(drug_count, axis=1)

# ── remove single-drug rows ────────────────────────────────────────────────────
df_valid = df[df["n_drugs"] >= 2].copy()
print(
    f"Rows with ≥2 drugs : {len(df_valid):,}  "
    f"(removed {(df['n_drugs'] == 1).sum():,} single-drug rows)"
)

# ── split by drug count ────────────────────────────────────────────────────────
df_2drug = df_valid[df_valid["n_drugs"] == 2]
df_3drug = df_valid[df_valid["n_drugs"] == 3]

print(f"\n── Overall counts ──────────────────────────")
print(f"  2-drug combinations : {len(df_2drug):,}")
print(f"  3-drug combinations : {len(df_3drug):,}")


# ── cancer-type filter ─────────────────────────────────────────────────────────
def cancer_filter(frame, keyword):
    return frame[frame["Disease_Entry"].str.contains(keyword, case=False, na=False)]


for cancer in ["breast", "lung"]:
    sub2 = cancer_filter(df_2drug, cancer)
    sub3 = cancer_filter(df_3drug, cancer)

    print(f"\n── {cancer.capitalize()} cancer ─────────────────────────")
    print(f"  2-drug : {len(sub2):,} rows")
    print(f"  3-drug : {len(sub3):,} rows")

    sub2.to_csv(OUTDIR / f"{cancer}_2drug.tsv", sep="\t", index=False)
    sub3.to_csv(OUTDIR / f"{cancer}_3drug.tsv", sep="\t", index=False)
    print(f"  Saved  : filtered/{cancer}_2drug.tsv  |  filtered/{cancer}_3drug.tsv")

# ── also save the full (all-cancer) splits ─────────────────────────────────────
df_2drug.to_csv(OUTDIR / "all_2drug.tsv", sep="\t", index=False)
df_3drug.to_csv(OUTDIR / "all_3drug.tsv", sep="\t", index=False)
print(f"\n── All-cancer splits saved ─────────────────")
print(f"  filtered/all_2drug.tsv  ({len(df_2drug):,} rows)")
print(f"  filtered/all_3drug.tsv  ({len(df_3drug):,} rows)")

# TODO: normalize: make them unique

"""
====================================================================================================================================
| Cancer Type | Experiments Count | Taee Type | Source    | Description                        |
|===================================================================================================================================
| Breast       | 80                | 3         | drugmap   | count in code                      |
| Breast       | 1701              | 2         | drugmap   | count in code                      |
| Breast       | 51                | 3         | overall   | mentioned in article               |
| Breast       | 23                | 3         | drugmap   | mentioned in article               |
| Breast       | 440               | 2         | drugmap   | mentioned in article               |
| Lung         | 56                | 3         | drugmap   | count in code                      |
| Lung         | 3062              | 2         | drugmap   | count in code                      |
| Lung         | 82                | 3         | overall   | mentioned in article               |
| Lung         | 0                 | 3         | drugmap   | mentioned in article               |
| Lung         | 841               | 2         | drugmap   | mentioned in article               |
====================================================================================================================================

breast 80 > 23
lung 56 > 0 

# checing the dataset
based on cell line there is no cell line which is related to the lung .
there is just:
روده - پستان - تخمدان
"""
