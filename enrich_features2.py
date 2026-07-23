import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import json
from pathlib import Path
import time
import urllib.parse
import os

# ────────────────────────────────────────────────────────────────────────────────
# DRUG NAME MAPPING (برای داروهایی که با نام اصلی در ChEMBL پیدا نمی‌شوند)
# ────────────────────────────────────────────────────────────────────────────────
DRUG_NAME_MAPPING = {
    # کدهای توسعه → نام علمی
    "dTI-015": "busulfan",
    "lEE011": "ribociclib",
    "lY2835219": "abemaciclib",
    "hKI-272": "neratinib",
    "aBT-888": "veliparib",
    "gDC-0068": "ipatasertib",
    "pF-03084014": "nirogacestat",
    "pF-05212384": "gedatolisib",
    "rAD-1901": "elacestrant",
    "mK-1775": "adavosertib",
    "aBT-869": "linifanib",
    "endoAngio-GT": "endostatin",
    "cATECHIN": "catechin",
    "icotinib hydrochloride": "icotinib",
    "motexafin gadolinium": "motexafin",
    "indocyanine green": "indocyanine green",
    "cycloleucine": "cycloleucine",
    "zoledronate": "zoledronic acid",
    "colestipol": "colestipol",
    "colesevelam": "colesevelam",
    "motesanib": "motesanib",
    "buparlisib": "buparlisib",
    "thiotepa": "thiotepa",
    "palbociclib": "palbociclib",
    "fulvestrant": "fulvestrant",
    "epirubicin": "epirubicin",
    "cyclophosphamide": "cyclophosphamide",
    "carboplatin": "carboplatin",
    "gemcitabine": "gemcitabine",
    "curcumin": "curcumin",
    "pirfenidone": "pirfenidone",
    "vandetanib": "vandetanib",
    "bortezomib": "bortezomib",
    "lurbinectedin": "lurbinectedin",
    "itraconazole": "itraconazole",
}

# ────────────────────────────────────────────────────────────────────────────────
# CACHE برای ذخیره نتایج و ادامه از نقطه توقف
# ────────────────────────────────────────────────────────────────────────────────
CACHE_FILE = Path("drug_info_cache.json")


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def normalize_drug_name(drug_name):
    """نرمال‌سازی نام دارو برای جستجو در ChEMBL"""
    if not drug_name:
        return drug_name
    drug_name = drug_name.strip().lower()
    if drug_name in DRUG_NAME_MAPPING:
        return DRUG_NAME_MAPPING[drug_name]
    return drug_name


# ---------------------------
# Resilient session with retries (افزایش یافته)
# ---------------------------
def make_session(retries=10, backoff=2.0, timeout=60):
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session._default_timeout = timeout
    return session


SESSION = make_session()


# ---------------------------
# 1. ChEMBL search (با retry اختصاصی و تاخیر بیشتر)
# TODO: return the first valid  molcule of list, i have to check chemble sorting.
# ---------------------------
# input: drug name, output: molcule: molcule.atc, molcule.smile, molecule.chembl_id -> get target
def search_chembl_with_retry(drug_name, max_attempts=3):
    """جستجوی دارو با چندین بار تلاش مجدد"""
    original_name = drug_name
    drug_name = normalize_drug_name(drug_name)

    if drug_name != original_name:
        print(f"    Mapping: '{original_name}' -> '{drug_name}'")

    encoded_name = urllib.parse.quote(drug_name)
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={encoded_name}&format=json"

    for attempt in range(max_attempts):
        try:
            r = SESSION.get(url, timeout=45)
            if r.status_code == 200:
                data = r.json()
                mols = data.get("molecules", [])
                if mols:
                    # برگرداندن اولین مولکول معتبر
                    for mol in mols:
                        if mol.get("pref_name") and mol.get("molecule_chembl_id"):
                            return mol
                    return mols[0]
                else:
                    if attempt < max_attempts - 1:
                        print(
                            f"    No molecules, retrying ({attempt+2}/{max_attempts})..."
                        )
                        time.sleep(2)
                        continue
                    return None
            else:
                if attempt < max_attempts - 1:
                    print(
                        f"    HTTP {r.status_code}, retrying ({attempt+2}/{max_attempts})..."
                    )
                    time.sleep(2)
                    continue
                return None
        except requests.exceptions.Timeout:
            if attempt < max_attempts - 1:
                print(f"    Timeout, retrying ({attempt+2}/{max_attempts})...")
                time.sleep(3)
                continue
            return None
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"    Error: {e}, retrying ({attempt+2}/{max_attempts})...")
                time.sleep(2)
                continue
            return None
    return None


def get_smiles(mol):
    return mol.get("molecule_structures", {}).get("canonical_smiles") if mol else None


def get_atc(mol):
    return mol.get("atc_classifications", []) if mol else []


# ---------------------------
# 2. Curated targets — PRIMARY: Open Targets MOA
# ---------------------------
def get_targets_opentargets(chembl_id):
    if not chembl_id:
        return None
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    query = """
    query DrugMOA($chemblId: String!) {
      drug(chemblId: $chemblId) {
        mechanismsOfAction {
          rows {
            mechanismOfAction
            actionType
            targets {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """
    try:
        r = SESSION.post(
            url,
            json={"query": query, "variables": {"chemblId": chembl_id}},
            timeout=30,
        )
        if r.status_code != 200:
            return None
        rows = (
            r.json()
            .get("data", {})
            .get("drug", {})
            .get("mechanismsOfAction", {})
            .get("rows", [])
        )
        targets = []
        for row in rows:
            for t in row.get("targets", []):
                targets.append(
                    {
                        "gene_symbol": t.get("approvedSymbol"),
                        "name": t.get("approvedName"),
                        "ensembl_id": t.get("id"),
                        "action_type": row.get("actionType"),
                        "mechanism": row.get("mechanismOfAction"),
                        "source": "OpenTargets",
                    }
                )
        return targets if targets else None
    except Exception as e:
        print(f"  [!] Open Targets error: {e}")
        return None


# ---------------------------
# 3. Curated targets — FALLBACK: ChEMBL mechanism endpoint
# ---------------------------
def get_targets_chembl_moa(chembl_id):
    if not chembl_id:
        return []
    url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism.json?molecule_chembl_id={chembl_id}&limit=100"
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code != 200:
            return []
        mechanisms = r.json().get("mechanisms", [])
        targets = []
        for m in mechanisms:
            targets.append(
                {
                    "gene_symbol": None,
                    "name": m.get("target_name"),
                    "ensembl_id": None,
                    "action_type": m.get("action_type"),
                    "mechanism": m.get("mechanism_of_action"),
                    "source": "ChEMBL-MOA",
                }
            )
        return targets
    except Exception as e:
        print(f"  [!] ChEMBL MOA fallback failed: {e}")
        return []


# ---------------------------
# 4. Unified target fetcher
# ---------------------------
def get_curated_targets(chembl_id):
    if not chembl_id:
        return []
    targets = get_targets_opentargets(chembl_id)
    if targets is None:
        targets = get_targets_chembl_moa(chembl_id)
    return targets


# ---------------------------
# 5. Main pipeline (با کش کردن نتایج)
# ---------------------------
def get_drug_info(drug_name, cache):
    """دریافت اطلاعات دارو با استفاده از cache"""
    # ابتدا در cache جستجو کن
    if drug_name in cache:
        return cache[drug_name]

    # جستجو در ChEMBL
    mol = search_chembl_with_retry(drug_name)

    if mol is None:
        result = {
            "drug": drug_name,
            "chembl_id": None,
            "smiles": None,
            "atc_codes": None,
            "curated_targets": None,
            "error": "Not found in ChEMBL",
        }
    else:
        chembl_id = mol.get("molecule_chembl_id")
        result = {
            "drug": drug_name,
            "chembl_id": chembl_id,
            "smiles": get_smiles(mol),
            "atc_codes": get_atc(mol),
            "curated_targets": get_curated_targets(chembl_id),
        }

    # ذخیره در cache
    cache[drug_name] = result
    save_cache(cache)
    return result


# ---------------------------
# 6. Process drug combinations
# ---------------------------
def process_drug_combinations_to_dict(file_path):
    df = pd.read_csv(file_path, sep="\t")
    print(f"\nProcessing: {file_path.name}")
    print(f"Total rows: {len(df)}")

    unique_drugs = set()
    for col in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
        if col in df.columns:
            unique_drugs.update(df[col].dropna().unique())

    print(f"Unique drugs found: {len(unique_drugs)}")

    # بارگذاری cache قبلی
    cache = load_cache()
    print(f"Loaded {len(cache)} cached drug infos")

    drugs_cache = {}
    drug_list = list(unique_drugs)

    for idx, drug_name in enumerate(drug_list):
        print(f"  Fetching info for: {drug_name} ({idx+1}/{len(drug_list)})")

        # اگر در cache هست، مستقیماً استفاده کن
        if drug_name in cache:
            drugs_cache[drug_name] = cache[drug_name]
            print(f"    (cached)")
        else:
            try:
                drug_info = get_drug_info(drug_name, cache)
                drugs_cache[drug_name] = drug_info
                if drug_info.get("chembl_id"):
                    print(f"    ✅ Found: {drug_info['chembl_id']}")
                else:
                    print(f"    ❌ Not found")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                drugs_cache[drug_name] = {
                    "drug": drug_name,
                    "chembl_id": None,
                    "smiles": None,
                    "atc_codes": None,
                    "curated_targets": None,
                    "error": str(e),
                }

        # تاخیر بیشتر (۱ ثانیه) برای جلوگیری از blocking
        time.sleep(1.0)

    # ساخت دیکشنری نهایی
    result = {}
    exclude_cols = [
        "Drug_Name_1",
        "Drug_Name_2",
        "Drug_Name_3",
        "DrugID_1",
        "DrugID_2",
        "DrugID_3",
        "DrugID_4",
    ]
    other_columns = [
        col for col in df.columns if col not in exclude_cols and col != "DrugCom_ID"
    ]

    for _, row in df.iterrows():
        drug_com_id = row["DrugCom_ID"]
        entry = {}
        for col in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if col in df.columns and pd.notna(row[col]):
                drug_name = row[col]
                entry[col] = drugs_cache.get(
                    drug_name,
                    {
                        "drug": drug_name,
                        "chembl_id": None,
                        "smiles": None,
                        "atc_codes": None,
                        "curated_targets": None,
                        "error": "Not in cache",
                    },
                )
            else:
                entry[col] = None
        for col in other_columns:
            if col in df.columns:
                value = row[col]
                entry[col] = None if pd.isna(value) else value
        result[drug_com_id] = entry

    print(f"✅ Processed {len(result)} combinations")
    return result


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Saved to: {output_path}")


# ── اجرای اصلی ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = Path("18_may_filtered_based_on_article")
    OUTPUT_DIR = Path("drug_info_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # پردازش فایل سرطان پستان
    breast_file = BASE_DIR / "breast_3drug.tsv"
    if breast_file.exists():
        breast_result = process_drug_combinations_to_dict(breast_file)
        save_to_json(breast_result, OUTPUT_DIR / "breast_3drug_drug_info.json")
    else:
        print(f"File not found: {breast_file}")

    # پردازش فایل سرطان ریه
    lung_file = BASE_DIR / "lung_3drug.tsv"
    if lung_file.exists():
        lung_result = process_drug_combinations_to_dict(lung_file)
        save_to_json(lung_result, OUTPUT_DIR / "lung_3drug_drug_info.json")
    else:
        print(f"File not found: {lung_file}")

    # آمار نهایی
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    cache = load_cache()
    total_cached = len(cache)
    successful = sum(1 for v in cache.values() if v.get("chembl_id"))
    print(f"Total unique drugs in cache: {total_cached}")
    print(f"Successfully found: {successful} ({100*successful/total_cached:.1f}%)")
    print(
        f"Not found: {total_cached - successful} ({100*(total_cached-successful)/total_cached:.1f}%)"
    )

    print(f"\n✅ Output files in: {OUTPUT_DIR}/")
    print(f"✅ Cache file: {CACHE_FILE}")
