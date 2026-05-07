import pandas as pd
import json


def filter_drug_combinations_by_cancer(tsv_path: str, output_json_path: str) -> dict:
    """
    Reads a drug combination TSV file, filters unique rows for lung cancer
    and breast cancer (case-insensitive substring match on Disease_Entry),
    and stores the results in a JSON file with keys 'lung' and 'breast'.

    Parameters
    ----------
    tsv_path : str
        Path to the input TSV file.
    output_json_path : str
        Path where the output JSON file will be saved.

    Returns
    -------
    dict
        Dictionary with keys 'lung' and 'breast', each containing a list
        of unique matching records.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")

    # ── 2. Global deduplication ───────────────────────────────────────────────
    df_unique = df.drop_duplicates()
    print(f"Rows after deduplication: {len(df_unique):,}")

    # ── 3. Filter per cancer keyword ─────────────────────────────────────────
    results = {}
    for keyword in ["lung", "breast"]:
        mask = df_unique["Disease_Entry"].str.contains(keyword, case=False, na=False)
        subset = df_unique[mask].drop_duplicates()
        results[keyword] = subset.to_dict(orient="records")
        print(f"  '{keyword}' → {len(subset):,} unique rows")

    # ── 4. Save to JSON ───────────────────────────────────────────────────────
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {output_json_path}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TSV_PATH = "/home/faezeh/uni/progect/HyperSynergyX/2.General Information of Drug Combination.tsv"
    OUTPUT_JSON = "/home/faezeh/uni/progect/HyperSynergyX/cancer_drug_combinations.json"

    results = filter_drug_combinations_by_cancer(TSV_PATH, OUTPUT_JSON)

    # Quick summary
    for cancer_type, records in results.items():
        diseases = list({r["Disease_Entry"] for r in records})
        print(
            f"\n[{cancer_type.upper()}] {len(records)} entries | disease labels: {diseases[:5]}"
        )

"""
output:
[LUNG] 3116 entries | example: disease labels: ['Lung Cancer', 
'Minimally invasive lung adenocarcinoma', 
'Head and neck cancer; Mesothelioma; Renal cell carcinoma; Small-cell lung cancer; Solid 
tumour/cancer'
, 'Small Cell Lung Cancer'
, 'Cystic fibrosis; Acute lung injury']

[BREAST] 1779 entries | example: disease labels: 
['Breast and ovarian cancer syndrome'
,'Anatomic Stage I Breast Cancer AJCC v8', 
'Breast Neoplasms', 
'Anatomic Stage II Breast Cancer AJCC v8', 
'Metastatic Breast Carcinoma']
"""
