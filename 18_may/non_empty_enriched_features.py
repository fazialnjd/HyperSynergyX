import json
from pathlib import Path


def check_drug_completeness(drug_info):
    """
    بررسی می‌کند که یک دارو دارای اطلاعات کامل (smiles, atc_codes, curated_targets) باشد یا نه
    """
    if not drug_info:
        return False
    # بررسی وجود فیلدها و غیر خالی بودن آن‌ها
    has_smiles = drug_info.get("smiles") is not None and drug_info.get("smiles") != ""
    has_atc = (
        drug_info.get("atc_codes") is not None
        and len(drug_info.get("atc_codes", [])) > 0
    )
    has_targets = (
        drug_info.get("curated_targets") is not None
        and len(drug_info.get("curated_targets", [])) > 0
    )
    return has_smiles and has_atc and has_targets


def filter_complete_combinations(input_json_path, output_json_path):
    """
    فیلتر کردن ترکیباتی که هر سه داروی آن‌ها اطلاعات کامل دارند

    Parameters:
    -----------
    input_json_path : Path
        مسیر فایل JSON ورودی (خروجی مرحله قبل)
    output_json_path : Path
        مسیر ذخیره فایل JSON خروجی (فقط ترکیبات کامل)

    Returns:
    --------
    dict
        دیکشنری شامل فقط ترکیبات کامل
    """

    # بارگذاری فایل JSON
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nProcessing: {input_json_path.name}")
    print(f"Total combinations: {len(data)}")

    # دیکشنری برای ذخیره ترکیبات کامل
    complete_combinations = {}

    # آمارگیری
    stats = {
        "total": len(data),
        "complete": 0,
        "incomplete": 0,
        "missing_drugs_count": 0,
        "missing_smiles": [],
        "missing_atc": [],
        "missing_targets": [],
    }

    for drug_com_id, combo_data in data.items():
        # بررسی سه دارو
        drugs_status = {}
        all_complete = True

        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                is_complete = check_drug_completeness(drug_info)
                drugs_status[drug_key] = {
                    "name": drug_info.get("drug") if drug_info else None,
                    "is_complete": is_complete,
                }

                if not is_complete:
                    all_complete = False
                    stats["missing_drugs_count"] += 1

                    # ثبت دقیق چه چیزی کم است
                    if not drug_info.get("smiles"):
                        stats["missing_smiles"].append(
                            drug_info.get("drug") if drug_info else "Unknown"
                        )
                    if (
                        not drug_info.get("atc_codes")
                        or len(drug_info.get("atc_codes", [])) == 0
                    ):
                        stats["missing_atc"].append(
                            drug_info.get("drug") if drug_info else "Unknown"
                        )
                    if (
                        not drug_info.get("curated_targets")
                        or len(drug_info.get("curated_targets", [])) == 0
                    ):
                        stats["missing_targets"].append(
                            drug_info.get("drug") if drug_info else "Unknown"
                        )
            else:
                all_complete = False
                stats["missing_drugs_count"] += 1

        # اگر هر سه دارو کامل بودند، به خروجی اضافه کن
        if all_complete:
            complete_combinations[drug_com_id] = combo_data
            stats["complete"] += 1
        else:
            stats["incomplete"] += 1

    # ذخیره فایل خروجی
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(complete_combinations, f, indent=2, ensure_ascii=False)

    # نمایش گزارش
    print("\n" + "=" * 70)
    print("FILTERING REPORT")
    print("=" * 70)
    print(f"Total combinations:        {stats['total']}")
    print(f"✅ Complete combinations:  {stats['complete']}")
    print(f"❌ Incomplete combinations: {stats['incomplete']}")
    print(f"   (missing at least one drug's data)")

    print("\n" + "-" * 70)
    print("MISSING DATA DETAILS:")
    print(f"  Drugs with missing SMILES:      {len(set(stats['missing_smiles']))}")
    print(f"  Drugs with missing ATC codes:   {len(set(stats['missing_atc']))}")
    print(f"  Drugs with missing targets:     {len(set(stats['missing_targets']))}")

    # نمایش نمونه داروهای مشکل‌دار
    if stats["missing_smiles"]:
        print(
            f"\n  Sample drugs missing SMILES: {list(set(stats['missing_smiles']))[:5]}"
        )
    if stats["missing_atc"]:
        print(f"  Sample drugs missing ATC:    {list(set(stats['missing_atc']))[:5]}")
    if stats["missing_targets"]:
        print(
            f"  Sample drugs missing targets: {list(set(stats['missing_targets']))[:5]}"
        )

    print(f"\n✅ Saved to: {output_json_path}")

    return complete_combinations


# ── اجرای اصلی ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_DIR = Path("drug_info_output")
    OUTPUT_DIR = Path("drug_info_output/filtered_complete")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # پردازش فایل سرطان پستان
    breast_input = INPUT_DIR / "breast_3drug_drug_info.json"
    breast_complete = filter_complete_combinations(
        breast_input, OUTPUT_DIR / "breast_3drug_complete.json"
    )

    # پردازش فایل سرطان ریه
    lung_input = INPUT_DIR / "lung_3drug_drug_info.json"

    lung_complete = filter_complete_combinations(
        lung_input, OUTPUT_DIR / "lung_3drug_complete.json"
    )

    # ── نمایش خلاصه نهایی ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if "breast_complete" in locals():
        print(f"Breast Cancer - Complete combinations: {len(breast_complete)} / 62")
    if "lung_complete" in locals():
        print(f"Lung Cancer - Complete combinations:   {len(lung_complete)} / 40")
    print(f"\n✅ Output files in: {OUTPUT_DIR}/")
