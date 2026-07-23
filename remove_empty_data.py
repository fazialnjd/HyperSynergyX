import json
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# مسیر فایل‌ها
# ────────────────────────────────────────────────────────────────────────────────
BREAST_FILE = Path("drug_info_output/cleaned/breast_3drug_drug_info_cleaned.json")
LUNG_FILE = Path("drug_info_output/cleaned/lung_3drug_drug_info_cleaned.json")
OUTPUT_DIR = Path("drug_info_output/strictly_cleaned_no_chembl")
OUTPUT_DIR.mkdir(exist_ok=True)


def is_field_empty(value):
    """بررسی خالی بودن فیلد"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip() == "."
    if isinstance(value, list):
        return len(value) == 0
    return False


def has_valid_targets(targets):
    """
    بررسی می‌کند که آیا در لیست curated_targets حداقل یک آیتم با gene_symbol معتبر وجود دارد
    """
    if not targets or not isinstance(targets, list) or len(targets) == 0:
        return False

    for t in targets:
        if t and not is_field_empty(t.get("gene_symbol")):
            return True
    return False


def is_drug_valid(drug_info):
    """
    بررسی اعتبار یک دارو بر اساس معیارهای سختگیرانه (بدون chembl_id):
    1. smiles باید داشته باشد
    2. atc_codes باید داشته باشد (لیست غیرخالی)
    3. curated_targets: اگر وجود دارد، حداقل یک gene_symbol معتبر باید داشته باشد
    """
    if not drug_info:
        return False

    # شرط 1: smiles نباید خالی باشد
    if is_field_empty(drug_info.get("smiles")):
        return False

    # شرط 2: atc_codes نباید خالی باشد
    if is_field_empty(drug_info.get("atc_codes")):
        return False

    # شرط 3: curated_targets - اگر لیست خالی نبود، باید حداقل یک gene_symbol معتبر داشته باشد
    targets = drug_info.get("curated_targets", [])
    if targets and len(targets) > 0:
        if not has_valid_targets(targets):
            return False

    return True


def clean_combination_file(input_file, output_file):
    """حذف ترکیباتی که حداقل یک داروی نامعتبر دارند"""

    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return None, 0, 0

    with open(input_file, "r") as f:
        data = json.load(f)

    original_combos = len(data)
    removed_combos = 0
    cleaned_data = {}

    for combo_id, combo_data in data.items():
        combo_valid = True

        # بررسی هر سه داروی ترکیب
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if not is_drug_valid(drug_info):
                    combo_valid = False
                    break

        if combo_valid:
            cleaned_data[combo_id] = combo_data
        else:
            removed_combos += 1

    # ذخیره فایل پاک‌شده
    with open(output_file, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    return cleaned_data, original_combos, removed_combos


def analyze_cleaned_file(file_path, name):
    """تحلیل فایل پاک‌شده و نمایش آمار"""

    if not file_path.exists():
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    # جمع‌آوری آمار داروها
    all_drugs = {}
    for combo_id, combo_data in data.items():
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if drug_info and drug_info.get("drug"):
                    drug_name = drug_info["drug"]
                    if drug_name not in all_drugs:
                        all_drugs[drug_name] = drug_info

    # آمار فیلدها
    valid_targets_count = 0
    has_atc_count = 0
    has_smiles_count = 0

    for drug_name, info in all_drugs.items():
        if not is_field_empty(info.get("smiles")):
            has_smiles_count += 1
        if not is_field_empty(info.get("atc_codes")):
            has_atc_count += 1

        targets = info.get("curated_targets", [])
        if has_valid_targets(targets):
            valid_targets_count += 1

    print(f"\n📊 {name}:")
    print(f"   Total combinations: {len(data)}")
    print(f"   Total unique drugs: {len(all_drugs)}")
    print(f"   ✅ Drugs with SMILES: {has_smiles_count}")
    print(f"   ✅ Drugs with ATC codes: {has_atc_count}")
    print(f"   ✅ Drugs with valid curated_targets: {valid_targets_count}")


def generate_strict_clean_report():
    """گزارش نهایی از پاکسازی سختگیرانه"""

    print("=" * 70)
    print("STRICT CLEANING (without chembl_id filter)")
    print("=" * 70)

    print("\n📋 Removal criteria (ALL must be present):")
    print("   1. ✅ smiles must exist (non-empty)")
    print("   2. ✅ atc_codes must exist (non-empty list)")
    print(
        "   3. ✅ curated_targets: if non-empty, at least one gene_symbol must be valid"
    )
    print("   (chembl_id is IGNORED - not checked)")
    print(
        "   (If ANY drug in a combination fails these, the entire combination is removed)"
    )

    # پاکسازی فایل سرطان پستان
    print("\n" + "-" * 50)
    breast_cleaned, breast_original, breast_removed = clean_combination_file(
        BREAST_FILE, OUTPUT_DIR / "breast_3drug_drug_info_strictly_cleaned.json"
    )

    # پاکسازی فایل سرطان ریه
    lung_cleaned, lung_original, lung_removed = clean_combination_file(
        LUNG_FILE, OUTPUT_DIR / "lung_3drug_drug_info_strictly_cleaned.json"
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

    # تحلیل فایل‌های پاک‌شده
    print("\n" + "=" * 70)
    print("CLEANED DATA ANALYSIS")
    print("=" * 70)

    analyze_cleaned_file(
        OUTPUT_DIR / "breast_3drug_drug_info_strictly_cleaned.json", "Breast Cancer"
    )
    analyze_cleaned_file(
        OUTPUT_DIR / "lung_3drug_drug_info_strictly_cleaned.json", "Lung Cancer"
    )

    # ────────────────────────────────────────────────────────────────────────────
    total_original = breast_original + lung_original
    total_remaining = (len(breast_cleaned) if breast_cleaned else 0) + (
        len(lung_cleaned) if lung_cleaned else 0
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n📊 Total combinations:")
    print(f"   Original: {total_original}")
    print(f"   Remaining: {total_remaining}")
    print(
        f"   Removed: {total_original - total_remaining} ({100*(total_original - total_remaining)/total_original:.1f}%)"
    )

    print(f"\n✅ Strictly cleaned files saved in: {OUTPUT_DIR}/")


def verify_sample():
    """بررسی یک نمونه از فایل پاک‌شده"""

    sample_file = OUTPUT_DIR / "breast_3drug_drug_info_strictly_cleaned.json"
    if not sample_file.exists():
        return

    with open(sample_file, "r") as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("SAMPLE VERIFICATION (First combination)")
    print("=" * 70)

    first_key = list(data.keys())[0] if data else None
    if first_key:
        first_combo = data[first_key]
        print(f"\nDrugCom_ID: {first_key}")
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in first_combo:
                drug_info = first_combo[drug_key]
                if drug_info:
                    print(f"\n  {drug_key}: {drug_info.get('drug')}")
                    print(f"    smiles: {drug_info.get('smiles', '')[:60]}...")
                    print(f"    atc_codes: {drug_info.get('atc_codes')}")
                    targets = drug_info.get("curated_targets", [])
                    print(f"    curated_targets: {len(targets)} targets")
                    if targets:
                        for t in targets[:2]:
                            print(
                                f"      - {t.get('gene_symbol')}: {t.get('action_type')}"
                            )


if __name__ == "__main__":
    generate_strict_clean_report()
    verify_sample()
