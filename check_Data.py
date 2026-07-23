import json
from pathlib import Path
from collections import defaultdict

# ────────────────────────────────────────────────────────────────────────────────
# مسیر فایل‌ها
# ────────────────────────────────────────────────────────────────────────────────
BREAST_FILE = Path("drug_info_output/breast_3drug_drug_info.json")
LUNG_FILE = Path("drug_info_output/lung_3drug_drug_info.json")


def extract_all_drugs_from_json(file_path):
    """استخراج تمام داروها از فایل JSON خروجی (ساختار تودرتو)"""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return {}

    with open(file_path, "r") as f:
        data = json.load(f)

    drugs_dict = {}

    for combo_id, combo_data in data.items():
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if drug_info and drug_info.get("drug"):
                    drug_name = drug_info["drug"]
                    if drug_name not in drugs_dict:
                        drugs_dict[drug_name] = drug_info

    return drugs_dict


def safe_len(obj):
    """امن‌تر کردن len() برای None و انواع دیگر"""
    if obj is None:
        return 0
    if isinstance(obj, list):
        return len(obj)
    return 0


def analyze_all_fields():
    """بررسی کامل همه فیلدهای مهم داروها از فایل‌های خروجی"""

    print("=" * 70)
    print("COMPLETE DRUG FIELDS ANALYSIS (from JSON files)")
    print("=" * 70)

    # استخراج داروها از هر دو فایل
    breast_drugs = extract_all_drugs_from_json(BREAST_FILE)
    lung_drugs = extract_all_drugs_from_json(LUNG_FILE)

    # ترکیب داروها (با اولویت breast در صورت تداخل)
    all_drugs = {**lung_drugs, **breast_drugs}

    print(f"\n📊 Total unique drugs in JSON files: {len(all_drugs)}")

    # دسته‌بندی
    complete_all = []  # هر ۴ فیلد را دارند
    missing_chembl = []  # chembl_id ندارند
    missing_smiles = []  # smiles ندارند
    missing_atc = []  # atc_codes ندارند
    missing_targets = []  # curated_targets خالی یا None
    all_missing_details = []  # جزئیات کامل

    for drug_name, info in all_drugs.items():
        has_chembl = info.get("chembl_id") is not None
        has_smiles = info.get("smiles") and info.get("smiles") != ""

        # مدیریت atc_codes (ممکن است None باشد)
        atc_codes = info.get("atc_codes")
        has_atc = (
            atc_codes is not None and len(atc_codes) > 0
            if isinstance(atc_codes, list)
            else False
        )

        # مدیریت curated_targets (ممکن است None باشد)
        targets = info.get("curated_targets")
        has_targets = (
            targets is not None and len(targets) > 0
            if isinstance(targets, list)
            else False
        )

        missing = []
        if not has_chembl:
            missing.append("chembl")
        if not has_smiles:
            missing.append("smiles")
        if not has_atc:
            missing.append("atc")
        if not has_targets:
            missing.append("targets")

        if len(missing) == 0:
            complete_all.append(drug_name)
        else:
            all_missing_details.append(
                {
                    "drug": drug_name,
                    "missing": missing,
                    "chembl_id": info.get("chembl_id"),
                    "has_smiles": has_smiles,
                    "has_atc": has_atc,
                    "has_targets": has_targets,
                }
            )

            if not has_chembl:
                missing_chembl.append(drug_name)
            if not has_smiles:
                missing_smiles.append(drug_name)
            if not has_atc:
                missing_atc.append(drug_name)
            if not has_targets:
                missing_targets.append(drug_name)

    # گزارش کلی
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Complete (all 4 fields):      {len(complete_all)} drugs")
    print(f"⚠️ Missing at least one field:   {len(all_missing_details)} drugs")

    print(f"\n{'='*70}")
    print("FIELD-BY-FIELD BREAKDOWN")
    print(f"{'='*70}")
    print(f"   Missing ChEMBL ID:    {len(missing_chembl)} drugs")
    print(f"   Missing SMILES:       {len(missing_smiles)} drugs")
    print(f"   Missing ATC codes:    {len(missing_atc)} drugs")
    print(f"   Missing Targets:      {len(missing_targets)} drugs")

    # ────────────────────────────────────────────────────────────────────────────
    # داروهای بدون curated_targets
    # ────────────────────────────────────────────────────────────────────────────
    if missing_targets:
        print(f"\n{'='*70}")
        print(f"DRUGS WITHOUT CURATED_TARGETS ({len(missing_targets)} drugs)")
        print(f"{'='*70}")

        for drug in missing_targets[:25]:
            info = all_drugs.get(drug, {})
            has_chembl = "✅" if info.get("chembl_id") else "❌"
            has_smiles = "✅" if info.get("smiles") else "❌"
            atc_codes = info.get("atc_codes")
            has_atc = "✅" if (atc_codes and len(atc_codes) > 0) else "❌"
            print(
                f"  {drug:30s} | chembl: {has_chembl} | smiles: {has_smiles} | atc: {has_atc}"
            )

        if len(missing_targets) > 25:
            print(f"  ... and {len(missing_targets)-25} more")

    # ────────────────────────────────────────────────────────────────────────────
    # داروهایی که چند فیلد را همزمان ندارند
    # ────────────────────────────────────────────────────────────────────────────
    severe_missing = [d for d in all_missing_details if len(d["missing"]) >= 2]
    if severe_missing:
        print(f"\n{'='*70}")
        print(f"DRUGS WITH MULTIPLE MISSING FIELDS ({len(severe_missing)} drugs)")
        print(f"{'='*70}")
        for d in severe_missing[:15]:
            print(f"  {d['drug']:30s} | missing: {', '.join(d['missing'])}")

    # ────────────────────────────────────────────────────────────────────────────
    # داروهای کاملاً کامل (برای اطمینان)
    # ────────────────────────────────────────────────────────────────────────────
    if complete_all:
        print(f"\n{'='*70}")
        print(f"COMPLETELY PERFECT DRUGS (all 4 fields) - {len(complete_all)} drugs")
        print(f"{'='*70}")
        for drug in complete_all[:20]:
            info = all_drugs.get(drug, {})
            targets = info.get("curated_targets")
            targets_count = len(targets) if targets else 0
            atc_codes = info.get("atc_codes")
            atc_count = len(atc_codes) if atc_codes else 0
            print(f"  {drug:30s} | atc: {atc_count} | targets: {targets_count}")
        if len(complete_all) > 20:
            print(f"  ... and {len(complete_all)-20} more")

    # ────────────────────────────────────────────────────────────────────────────
    # داروهای خاص (برای بررسی)
    # ────────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SPECIFIC DRUGS CHECK")
    print(f"{'='*70}")

    drugs_to_check = [
        "carboplatin",
        "colestipol",
        "colesevelam",
        "dTI-015",
        "motexafin gadolinium",
        "erlotinib",
        "gemcitabine",
    ]
    for drug in drugs_to_check:
        if drug in all_drugs:
            info = all_drugs[drug]
            print(f"\n✅ {drug}:")
            print(f"   chembl_id: {info.get('chembl_id')}")
            print(f"   smiles: {'✅' if info.get('smiles') else '❌'}")
            atc_codes = info.get("atc_codes")
            print(f"   atc_codes: {atc_codes if atc_codes else '❌'}")
            targets = info.get("curated_targets")
            targets_count = len(targets) if targets else 0
            print(f"   curated_targets: {targets_count} targets")
        else:
            print(f"\n❌ {drug}: NOT FOUND IN DATA")

    # ────────────────────────────────────────────────────────────────────────────
    # ذخیره گزارش کامل
    # ────────────────────────────────────────────────────────────────────────────
    simple_report = {
        "total_drugs": len(all_drugs),
        "complete_all_fields": len(complete_all),
        "missing_chembl": missing_chembl,
        "missing_smiles": missing_smiles,
        "missing_atc": missing_atc,
        "missing_targets": missing_targets,
        "multiple_missing": [
            {"drug": d["drug"], "missing": d["missing"]} for d in severe_missing
        ],
    }

    output_file = Path("drug_info_output/complete_analysis_report.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(simple_report, f, indent=2)
    print(f"\n📁 Full report saved to: {output_file}")

    return simple_report, all_drugs


def export_drugs_summary(all_drugs):
    """خروجی گرفتن از خلاصه داروها"""

    summary = []
    for drug_name, info in all_drugs.items():
        atc_codes = info.get("atc_codes")
        targets = info.get("curated_targets")

        summary.append(
            {
                "drug": drug_name,
                "chembl_id": info.get("chembl_id"),
                "has_smiles": bool(info.get("smiles")),
                "has_atc": (
                    bool(atc_codes and len(atc_codes) > 0)
                    if isinstance(atc_codes, list)
                    else False
                ),
                "has_targets": (
                    bool(targets and len(targets) > 0)
                    if isinstance(targets, list)
                    else False
                ),
                "atc_count": (
                    len(atc_codes) if (atc_codes and isinstance(atc_codes, list)) else 0
                ),
                "targets_count": (
                    len(targets) if (targets and isinstance(targets, list)) else 0
                ),
            }
        )

    # مرتب‌سازی بر اساس نام
    summary.sort(key=lambda x: x["drug"])

    output_file = Path("drug_info_output/drugs_summary.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Drugs summary saved to: {output_file}")

    # نمایش در کنسول
    print("\n" + "=" * 70)
    print("DRUGS SUMMARY (first 20)")
    print("=" * 70)
    for d in summary[:20]:
        status = []
        if d["has_smiles"]:
            status.append("smiles")
        if d["has_atc"]:
            status.append("atc")
        if d["has_targets"]:
            status.append("targets")
        print(f"  {d['drug']:30s} | {', '.join(status)}")
    if len(summary) > 20:
        print(f"  ... and {len(summary)-20} more")


if __name__ == "__main__":
    report, all_drugs = analyze_all_fields()
    export_drugs_summary(all_drugs)

    # جمع‌بندی نهایی
    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    if report["missing_smiles"]:
        print(
            f"❌ {len(report['missing_smiles'])} drugs missing SMILES - cannot use in siamese network"
        )
    else:
        print(f"✅ ALL {report['total_drugs']} drugs have SMILES!")

    if report["missing_chembl"]:
        print(f"⚠️ {len(report['missing_chembl'])} drugs missing ChEMBL ID")

    if report["missing_atc"]:
        print(f"ℹ️ {len(report['missing_atc'])} drugs missing ATC codes - optional")

    if report["missing_targets"]:
        print(
            f"ℹ️ {len(report['missing_targets'])} drugs missing curated_targets - optional"
        )

    print("\n💡 RECOMMENDATION:")
    if not report["missing_smiles"]:
        print("   ✅ You can proceed with building hypergraphs and random walk models.")
        print("   ✅ SMILES are available for all drugs (siamese network possible).")
        print("   ⚠️ ATC codes and targets are optional and not required.")
    else:
        print("   ❌ Please fix missing SMILES first.")
