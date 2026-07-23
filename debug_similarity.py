# debug_similarity.py
# اسکریپت دیباگ برای بررسی داده‌های ورودی محاسبه شباهت

import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

# ============================================================================
# مسیرهای فایل‌ها (مطابق با کد اصلی شما)
# ============================================================================

# مسیر فایل‌های JSON ورودی
breast_json = Path("drug_info_output/strictly_cleaned_no_chembl/breast_3drug_drug_info_strictly_cleaned.json")
lung_json = Path("drug_info_output/strictly_cleaned_no_chembl/lung_3drug_drug_info_strictly_cleaned.json")

# مسیر فایل‌های خروجی (ماتریس‌های ذخیره شده)
output_dir = Path("similarity_matrices")
breast_drug_names_file = output_dir / "breast_drug_names.json"
lung_drug_names_file = output_dir / "lung_drug_names.json"

# ============================================================================
# توابع کمکی (کپی شده از کد اصلی)
# ============================================================================

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    """Convert SMILES to Morgan fingerprint"""
    if not smiles or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_from_binary_vectors(v_a, v_b):
    """Tanimoto similarity between two binary vectors"""
    dot = float(np.dot(v_a, v_b))
    norm_a_sq = float(np.dot(v_a, v_a))
    norm_b_sq = float(np.dot(v_b, v_b))
    denom = norm_a_sq + norm_b_sq - dot
    if denom == 0.0:
        return 1.0
    return dot / denom


def load_drugs_from_cleaned_json(json_path):
    """Load drug information from cleaned JSON file (همان کد اصلی)"""
    with open(json_path, "r") as f:
        data = json.load(f)

    atc_map = {}
    target_map = {}
    smiles_map = {}

    for combo_id, combo_data in data.items():
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if drug_info and isinstance(drug_info, dict) and drug_info.get("drug"):
                    drug_name = drug_info["drug"]

                    # ATC codes
                    if drug_name not in atc_map:
                        atc_codes = drug_info.get("atc_codes", [])
                        atc_map[drug_name] = atc_codes if isinstance(atc_codes, list) else []

                    # Targets
                    if drug_name not in target_map:
                        targets = drug_info.get("curated_targets", [])
                        target_map[drug_name] = targets if targets else []

                    # SMILES
                    if drug_name not in smiles_map:
                        smiles = drug_info.get("smiles", "")
                        smiles_map[drug_name] = smiles if smiles else ""

    drug_names = sorted(list(atc_map.keys()))
    return drug_names, atc_map, target_map, smiles_map


# ============================================================================
# تابع اصلی دیباگ
# ============================================================================

def debug_dataset(cancer_name, json_path):
    """بررسی کامل یک دیتاست"""
    print("\n" + "=" * 70)
    print(f"DEBUGGING: {cancer_name.upper()} CANCER DATASET")
    print("=" * 70)
    
    # بررسی وجود فایل
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return
    
    print(f"JSON file exists: {json_path}")
    print(f"File size: {json_path.stat().st_size / 1024:.2f} KB")
    
    # بارگذاری داده
    print("\n--- Loading data from JSON ---")
    drug_names, atc_map, target_map, smiles_map = load_drugs_from_cleaned_json(json_path)
    
    n_drugs = len(drug_names)
    print(f"\nNumber of unique drugs: {n_drugs}")
    print(f"First 10 drugs: {drug_names[:10]}")
    
    # ========================================================================
    # 1. بررسی SMILES
    # ========================================================================
    print("\n" + "-" * 50)
    print("1. SMILES VALIDITY CHECK")
    print("-" * 50)
    
    valid_smiles_count = 0
    invalid_smiles_list = []
    smiles_values = []
    
    for drug in drug_names:
        smiles = smiles_map.get(drug, "")
        is_valid = False
        
        if smiles and smiles.strip():
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is not None:
                is_valid = True
                smiles_values.append(smiles.strip())
        
        if is_valid:
            valid_smiles_count += 1
        else:
            invalid_smiles_list.append(drug)
    
    print(f"Valid SMILES: {valid_smiles_count}/{n_drugs}")
    
    if invalid_smiles_list:
        print(f"Drugs with INVALID or MISSING SMILES (first 10):")
        for drug in invalid_smiles_list[:10]:
            smiles_val = smiles_map.get(drug, "")
            print(f"  - {drug}: SMILES='{smiles_val[:50] if smiles_val else 'MISSING'}'")
    
    # بررسی یکسان بودن SMILES ها
    unique_smiles = set(smiles_values)
    print(f"\nUnique SMILES strings: {len(unique_smiles)} out of {valid_smiles_count}")
    
    if len(unique_smiles) <= 1 and valid_smiles_count > 1:
        print("⚠️  WARNING: All drugs have the SAME SMILES string!")
        if smiles_values:
            print(f"   The common SMILES is: {smiles_values[0][:100]}")
    
    # ========================================================================
    # 2. بررسی فینگرپرینت‌ها
    # ========================================================================
    print("\n" + "-" * 50)
    print("2. FINGERPRINT CHECK")
    print("-" * 50)
    
    fps = {}
    for drug in drug_names:
        smiles = smiles_map.get(drug, "")
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            fps[drug] = fp
    
    print(f"Valid fingerprints: {len(fps)}/{n_drugs}")
    
    if len(fps) >= 2:
        # محاسبه شباهت برای چند جفت
        drug_list = list(fps.keys())
        print("\nPairwise Tanimoto similarities (first 5 pairs):")
        
        for i in range(min(5, len(drug_list) - 1)):
            drug_a = drug_list[i]
            drug_b = drug_list[i + 1]
            sim = tanimoto_from_binary_vectors(fps[drug_a], fps[drug_b])
            print(f"  {drug_a[:25]:25s} vs {drug_b[:25]:25s} : {sim:.6f}")
        
        # بررسی یکسان بودن همه فینگرپرینت‌ها
        fp_list = list(fps.values())
        all_identical = all(np.array_equal(fp_list[0], fp) for fp in fp_list[1:])
        
        print(f"\nAll fingerprints identical? {all_identical}")
        
        if all_identical:
            print("⚠️  CRITICAL WARNING: All fingerprints are IDENTICAL!")
            print("   This explains why all chemical similarities are the same (0.77).")
            print("   The self-similarity should be 1.0, but here it's different:")
            self_sim = tanimoto_from_binary_vectors(fp_list[0], fp_list[0])
            print(f"   Self-similarity (same vector): {self_sim}")
            
            # بررسی محتوای فینگرپرینت
            nonzero_bits = np.count_nonzero(fp_list[0])
            print(f"   Number of non-zero bits in fingerprint: {nonzero_bits}/{len(fp_list[0])}")
            
            if nonzero_bits == 0:
                print("   → All fingerprints are ALL ZEROS!")
            elif 0 < nonzero_bits < len(fp_list[0]):
                print(f"   → Fingerprints have {nonzero_bits} bits set, but all drugs share EXACTLY the same pattern")
    
    # ========================================================================
    # 3. بررسی ATC codes
    # ========================================================================
    print("\n" + "-" * 50)
    print("3. ATC CODES CHECK")
    print("-" * 50)
    
    atc_counts = []
    for drug in drug_names:
        atc_list = atc_map.get(drug, [])
        atc_counts.append(len(atc_list))
    
    drugs_without_atc = sum(1 for c in atc_counts if c == 0)
    print(f"Drugs with ATC codes: {n_drugs - drugs_without_atc}/{n_drugs}")
    print(f"Average ATC codes per drug: {np.mean(atc_counts):.2f}")
    
    # بررسی یکسان بودن ATC ها
    atc_sets = [set(atc_map.get(drug, [])) for drug in drug_names]
    unique_atc_sets = [list(s) for s in set(tuple(sorted(s)) for s in atc_sets if s)]
    
    print(f"Unique ATC patterns: {len(unique_atc_sets)}")
    
    if len(unique_atc_sets) <= 1 and n_drugs > 1:
        print("⚠️  WARNING: All drugs have identical ATC codes!")
    
    # ========================================================================
    # 4. بررسی Target genes
    # ========================================================================
    print("\n" + "-" * 50)
    print("4. TARGET GENES CHECK")
    print("-" * 50)
    
    target_counts = []
    target_sets = []
    
    for drug in drug_names:
        targets = target_map.get(drug, [])
        if targets and isinstance(targets[0], dict):
            genes = {t.get("gene_symbol") for t in targets if t.get("gene_symbol")}
        else:
            genes = set(targets) if targets else set()
        target_sets.append(genes)
        target_counts.append(len(genes))
    
    drugs_without_targets = sum(1 for c in target_counts if c == 0)
    print(f"Drugs with targets: {n_drugs - drugs_without_targets}/{n_drugs}")
    print(f"Average targets per drug: {np.mean(target_counts):.2f}")
    
    # بررسی یکسان بودن target sets
    unique_target_sets = len(set(frozenset(s) for s in target_sets))
    print(f"Unique target patterns: {unique_target_sets}")
    
    if unique_target_sets <= 1 and n_drugs > 1:
        print("⚠️  WARNING: All drugs have identical target sets!")
    
    # ========================================================================
    # 5. بررسی فایل‌های خروجی ذخیره شده (اگر وجود داشته باشند)
    # ========================================================================
    print("\n" + "-" * 50)
    print("5. CHECKING SAVED SIMILARITY MATRICES")
    print("-" * 50)
    
    saved_drugs_file = output_dir / f"{cancer_name}_drug_names.json"
    saved_chem_file = output_dir / f"{cancer_name}_S_chem.npy"
    saved_atc_file = output_dir / f"{cancer_name}_S_atc.npy"
    saved_tgt_file = output_dir / f"{cancer_name}_S_tgt.npy"
    saved_int_file = output_dir / f"{cancer_name}_S_int.npy"
    
    if saved_drugs_file.exists():
        with open(saved_drugs_file, "r") as f:
            saved_drug_names = json.load(f)
        print(f"Saved drug names file exists: {len(saved_drug_names)} drugs")
        
        if saved_drug_names != drug_names:
            print("⚠️  WARNING: Saved drug names don't match current extraction!")
            print(f"   Saved first 5: {saved_drug_names[:5]}")
            print(f"   Current first 5: {drug_names[:5]}")
    else:
        print(f"Saved drug names file NOT FOUND: {saved_drugs_file}")
    
    if saved_chem_file.exists():
        chem_mat = np.load(saved_chem_file)
        print(f"\nSaved chemical similarity matrix shape: {chem_mat.shape}")
        print(f"  Unique values in matrix: {len(np.unique(chem_mat))}")
        print(f"  Min value: {chem_mat.min():.6f}")
        print(f"  Max value: {chem_mat.max():.6f}")
        print(f"  Mean value: {chem_mat.mean():.6f}")
        print(f"  Std value: {chem_mat.std():.6f}")
        
        if np.std(chem_mat) < 1e-6:
            print("  ⚠️  MATRIX HAS NO VARIATION - all values are identical!")
    else:
        print(f"Chemical similarity file NOT FOUND: {saved_chem_file}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY & DIAGNOSIS")
    print("=" * 70)
    
    if len(fps) >= 2 and all(np.array_equal(list(fps.values())[0], fp) for fp in list(fps.values())[1:]):
        print("\n❌ PROBLEM FOUND: All fingerprints are IDENTICAL")
        print("   Cause: All drugs have the same or invalid SMILES strings")
        print("   Solution: Check the JSON file structure and SMILES extraction")
    
    if len(unique_atc_sets) <= 1 and n_drugs > 1:
        print("\n❌ PROBLEM FOUND: All drugs have identical ATC codes")
    
    if unique_target_sets <= 1 and n_drugs > 1:
        print("\n❌ PROBLEM FOUND: All drugs have identical target sets")
    
    if len(fps) == 0:
        print("\n❌ CRITICAL: No valid fingerprints could be generated")
        print("   Check if RDKit is working and SMILES strings are valid")
    
    return {
        "n_drugs": n_drugs,
        "valid_smiles": valid_smiles_count,
        "n_unique_smiles": len(unique_smiles),
        "fingerprints_identical": len(fps) >= 2 and all(np.array_equal(list(fps.values())[0], fp) for fp in list(fps.values())[1:]) if len(fps) >= 2 else False,
    }


# ============================================================================
# اجرای اصلی
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SIMILARITY DATA DEBUG SCRIPT")
    print("=" * 70)
    print(f"\nChecking paths:")
    print(f"  Breast JSON: {breast_json}")
    print(f"  Lung JSON:   {lung_json}")
    print(f"  Output dir:  {output_dir}")
    
    # بررسی Breast Cancer
    debug_dataset("breast", breast_json)
    
    # بررسی Lung Cancer
    debug_dataset("lung", lung_json)
    
    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)