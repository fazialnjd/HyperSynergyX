import json
from pathlib import Path
from copy import deepcopy

# ────────────────────────────────────────────────────────────────────────────────
# مسیر فایل‌ها
# ────────────────────────────────────────────────────────────────────────────────
BREAST_FILE = Path("drug_info_output/breast_3drug_drug_info.json")
LUNG_FILE = Path("drug_info_output/lung_3drug_drug_info.json")
OUTPUT_DIR = Path("drug_info_output/cleaned")
OUTPUT_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────────
# داروهایی که باید حذف شوند (بدون SMILES یا بدون ChEMBL ID)
# ────────────────────────────────────────────────────────────────────────────────
def identify_incomplete_drugs(all_drugs):
    """شناسایی داروهایی که فیلدهای ضروری (SMILES و ChEMBL ID) را ندارند"""

    drugs_to_remove = []
    reasons = {}

    for drug_name, info in all_drugs.items():
        has_chembl = info.get("chembl_id") is not None
        has_smiles = info.get("smiles") and info.get("smiles") != ""

        missing = []
        if not has_chembl:
            missing.append("chembl_id")
        if not has_smiles:
            missing.append("smiles")

        if missing:
            drugs_to_remove.append(drug_name)
            reasons[drug_name] = missing

    return drugs_to_remove, reasons


def remove_empty_combinations(input_file, output_file, drugs_to_remove):
    """حذف ترکیباتی که شامل داروهای مشکل‌دار هستند"""

    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return None, 0, 0

    with open(input_file, "r") as f:
        data = json.load(f)

    original_count = len(data)
    removed_combos = 0
    cleaned_data = {}

    for combo_id, combo_data in data.items():
        keep = True

        # بررسی هر سه داروی ترکیب
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if drug_info and drug_info.get("drug") in drugs_to_remove:
                    keep = False
                    removed_combos += 1
                    break

        if keep:
            cleaned_data[combo_id] = combo_data

    # ذخیره فایل پاک‌شده
    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    return cleaned_data, original_count, removed_combos


def analyze_cleaned_data(cleaned_data, file_name):
    """تحلیل داده‌های پاک‌شده"""

    if not cleaned_data:
        return

    all_drugs = {}
    for combo_id, combo_data in cleaned_data.items():
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if drug_info and drug_info.get("drug"):
                    drug_name = drug_info["drug"]
                    if drug_name not in all_drugs:
                        all_drugs[drug_name] = drug_info

    # آمار
    complete_count = 0
    missing_atc_count = 0
    missing_targets_count = 0

    for drug_name, info in all_drugs.items():
        has_smiles = info.get("smiles") and info.get("smiles") != ""
        has_chembl = info.get("chembl_id") is not None
        has_atc = info.get("atc_codes") and len(info.get("atc_codes", [])) > 0
        has_targets = (
            info.get("curated_targets") and len(info.get("curated_targets", [])) > 0
        )

        if has_smiles and has_chembl and has_atc and has_targets:
            complete_count += 1
        if not has_atc:
            missing_atc_count += 1
        if not has_targets:
            missing_targets_count += 1

    print(f"\n📊 {file_name}:")
    print(f"   Total unique drugs: {len(all_drugs)}")
    print(f"   ✅ Complete (all fields): {complete_count}")
    print(f"   ⚠️ Missing ATC only: {missing_atc_count}")
    print(f"   ⚠️ Missing Targets only: {missing_targets_count}")

    return all_drugs


def generate_final_report():
    """گزارش نهایی از داده‌های پاک‌شده"""

    print("=" * 70)
    print("CLEANING DRUG COMBINATION DATA")
    print("=" * 70)

    # ابتدا اطلاعات کامل داروها را از هر دو فایل استخراج می‌کنیم
    all_drugs = {}

    for file_path in [BREAST_FILE, LUNG_FILE]:
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
            for combo_id, combo_data in data.items():
                for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
                    if drug_key in combo_data:
                        drug_info = combo_data[drug_key]
                        if drug_info and drug_info.get("drug"):
                            drug_name = drug_info["drug"]
                            if drug_name not in all_drugs:
                                all_drugs[drug_name] = drug_info

    # شناسایی داروهای مشکل‌دار
    drugs_to_remove, reasons = identify_incomplete_drugs(all_drugs)

    print(f"\n🔍 Problematic drugs identified ({len(drugs_to_remove)} drugs):")
    for drug in drugs_to_remove:
        print(f"   - {drug}: missing {', '.join(reasons[drug])}")

    # پاک کردن فایل‌ها
    print("\n" + "=" * 70)
    print("REMOVING COMBINATIONS WITH PROBLEMATIC DRUGS")
    print("=" * 70)

    # پردازش فایل سرطان پستان
    breast_cleaned, breast_original, breast_removed = remove_empty_combinations(
        BREAST_FILE, OUTPUT_DIR / "breast_3drug_drug_info_cleaned.json", drugs_to_remove
    )

    # پردازش فایل سرطان ریه
    lung_cleaned, lung_original, lung_removed = remove_empty_combinations(
        LUNG_FILE, OUTPUT_DIR / "lung_3drug_drug_info_cleaned.json", drugs_to_remove
    )

    # ────────────────────────────────────────────────────────────────────────────
    # گزارش نهایی
    # ────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"\n📁 Breast Cancer:")
    print(f"   Original combinations: {breast_original}")
    print(f"   Removed combinations: {breast_removed}")
    print(f"   Remaining combinations: {len(breast_cleaned) if breast_cleaned else 0}")

    print(f"\n📁 Lung Cancer:")
    print(f"   Original combinations: {lung_original}")
    print(f"   Removed combinations: {lung_removed}")
    print(f"   Remaining combinations: {len(lung_cleaned) if lung_cleaned else 0}")

    # تحلیل داده‌های پاک‌شده
    print("\n" + "=" * 70)
    print("CLEANED DATA ANALYSIS")
    print("=" * 70)

    breast_drugs = analyze_cleaned_data(breast_cleaned, "Breast Cancer")
    lung_drugs = analyze_cleaned_data(lung_cleaned, "Lung Cancer")

    # داروهای حذف شده
    print("\n" + "=" * 70)
    print("REMOVED DRUGS LIST")
    print("=" * 70)
    for drug in drugs_to_remove:
        print(f"   - {drug}")

    # ────────────────────────────────────────────────────────────────────────────
    # جمع‌بندی نهایی
    # ────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    total_original = breast_original + lung_original
    total_remaining = (len(breast_cleaned) if breast_cleaned else 0) + (
        len(lung_cleaned) if lung_cleaned else 0
    )

    print(f"\n📊 Total combinations:")
    print(f"   Original: {total_original}")
    print(f"   Remaining: {total_remaining}")
    print(
        f"   Removed: {total_original - total_remaining} ({100*(total_original - total_remaining)/total_original:.1f}%)"
    )

    print(f"\n✅ Cleaned files saved in: {OUTPUT_DIR}/")
    print(f"\n💡 These drugs were removed because they lack:")
    print(f"   - ChEMBL ID")
    print(f"   - SMILES")
    print("\n💡 Remaining data is clean and ready for model building.")


def verify_cleaned_files():
    """بررسی فایل‌های پاک‌شده و نمایش نمونه"""

    print("\n" + "=" * 70)
    print("VERIFYING CLEANED FILES")
    print("=" * 70)

    cleaned_breast = OUTPUT_DIR / "breast_3drug_drug_info_cleaned.json"
    cleaned_lung = OUTPUT_DIR / "lung_3drug_drug_info_cleaned.json"

    if cleaned_breast.exists():
        with open(cleaned_breast, "r") as f:
            data = json.load(f)

        print(f"\n📁 Breast Cancer Cleaned File Sample:")
        first_key = list(data.keys())[0] if data else None
        if first_key:
            first_combo = data[first_key]
            print(f"   DrugCom_ID: {first_key}")
            for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
                if drug_key in first_combo:
                    drug_info = first_combo[drug_key]
                    if drug_info:
                        print(f"   {drug_key}: {drug_info.get('drug')}")
                        print(f"      chembl_id: {drug_info.get('chembl_id')}")
                        print(f"      smiles: {drug_info.get('smiles', '')[:60]}...")
                        print(f"      atc_codes: {drug_info.get('atc_codes')}")
                        print(
                            f"      curated_targets: {len(drug_info.get('curated_targets', []))} targets"
                        )


if __name__ == "__main__":
    generate_final_report()
    verify_cleaned_files()
