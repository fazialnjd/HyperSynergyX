import numpy as np
import pandas as pd
import requests
import json
import os
from collections import defaultdict

# ============================================================================
# SIMILARITY CALCULATION FUNCTIONS (from paper equations)
# ============================================================================


def tanimoto_similarity(vec_a, vec_b):
    """
    Equation (1): Tanimoto similarity between binary fingerprint vectors.

    ChemSim(v_a, v_b) = (v_a · v_b) / (||v_a||^2 + ||v_b||^2 - v_a · v_b)
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a_sq = np.dot(vec_a, vec_a)
    norm_b_sq = np.dot(vec_b, vec_b)

    denominator = norm_a_sq + norm_b_sq - dot_product
    if denominator == 0:
        return 0.0

    return float(dot_product) / float(denominator)


def jaccard_similarity(set_a, set_b):
    """
    Jaccard similarity between two sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set_a and not set_b:
        return 1.0

    set_a = set(set_a) if not isinstance(set_a, set) else set_a
    set_b = set(set_b) if not isinstance(set_b, set) else set_b

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def build_chemical_similarity(drug_names, smiles_map):
    """
    Build chemical similarity matrix using Tanimoto similarity on RDKit fingerprints.

    Equation (1) from paper.

    Args:
        drug_names: list of drug names
        smiles_map: dict mapping drug name -> SMILES string

    Returns:
        NxN numpy array of chemical similarities
    """
    n = len(drug_names)
    chem_sim = np.zeros((n, n), dtype=float)

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print(
            "Warning: RDKit not installed. Returning identity matrix for chemical similarity."
        )
        return np.eye(n, dtype=float)

    # Build fingerprints for all drugs
    fingerprints = {}
    for drug in drug_names:
        smiles = smiles_map.get(drug, "")
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints[drug] = np.array(fp, dtype=float)
                else:
                    fingerprints[drug] = np.zeros(2048, dtype=float)
            except:
                fingerprints[drug] = np.zeros(2048, dtype=float)
        else:
            fingerprints[drug] = np.zeros(2048, dtype=float)

    # Compute pairwise Tanimoto similarities
    for i, drug_a in enumerate(drug_names):
        for j, drug_b in enumerate(drug_names):
            if i == j:
                chem_sim[i, j] = 1.0
            else:
                fp_a = fingerprints.get(drug_a, np.zeros(2048, dtype=float))
                fp_b = fingerprints.get(drug_b, np.zeros(2048, dtype=float))
                chem_sim[i, j] = tanimoto_similarity(fp_a, fp_b)

    return chem_sim


def build_atc_similarity(drug_names, atc_map):
    """
    Build ATC similarity matrix using Jaccard similarity across 3 ATC levels.

    Equations (2) & (3) from paper.

    Args:
        drug_names: list of drug names
        atc_map: dict mapping drug name -> list/set of ATC codes

    Returns:
        NxN numpy array of ATC similarities
    """
    n = len(drug_names)
    atc_sim = np.zeros((n, n), dtype=float)

    # Extract ATC codes by level (first 3 levels: e.g., "C09CA" -> levels C, C09, C09C)
    atc_by_level = {}
    for drug in drug_names:
        atc_by_level[drug] = [set(), set(), set()]  # levels 1, 2, 3

        atc_codes = atc_map.get(drug, [])
        if isinstance(atc_codes, str):
            atc_codes = [atc_codes]

        for code in atc_codes:
            if code and len(str(code).strip()) > 0:
                code_str = str(code).strip()
                # Level 1: first character (e.g., "C")
                if len(code_str) >= 1:
                    atc_by_level[drug][0].add(code_str[0])
                # Level 2: first 3 characters (e.g., "C09")
                if len(code_str) >= 3:
                    atc_by_level[drug][1].add(code_str[:3])
                # Level 3: first 4 characters (e.g., "C09C")
                if len(code_str) >= 4:
                    atc_by_level[drug][2].add(code_str[:4])

    # Compute pairwise Jaccard similarities averaged across levels
    for i, drug_a in enumerate(drug_names):
        for j, drug_b in enumerate(drug_names):
            if i == j:
                atc_sim[i, j] = 1.0
            else:
                level_sims = []
                for level in range(3):
                    sim = jaccard_similarity(
                        atc_by_level[drug_a][level], atc_by_level[drug_b][level]
                    )
                    level_sims.append(sim)
                atc_sim[i, j] = np.mean(level_sims)

    return atc_sim


def build_target_similarity(drug_names, target_map):
    """
    Build target similarity matrix using Jaccard similarity of protein targets.

    Equation (4) from paper.

    Args:
        drug_names: list of drug names
        target_map: dict mapping drug name -> list/set of protein target IDs

    Returns:
        NxN numpy array of target similarities
    """
    n = len(drug_names)
    target_sim = np.zeros((n, n), dtype=float)

    # Normalize target sets
    targets = {}
    for drug in drug_names:
        target_list = target_map.get(drug, [])
        targets[drug] = set(target_list) if target_list else set()

    # Compute pairwise Jaccard similarities
    for i, drug_a in enumerate(drug_names):
        for j, drug_b in enumerate(drug_names):
            if i == j:
                target_sim[i, j] = 1.0
            else:
                sim = jaccard_similarity(targets[drug_a], targets[drug_b])
                target_sim[i, j] = sim

    return target_sim


def build_integrated_similarity(chem_sim, atc_sim, target_sim):
    """
    Build integrated similarity matrix as element-wise maximum.

    Equation (5) from paper.

    Args:
        chem_sim: NxN chemical similarity matrix
        atc_sim: NxN ATC similarity matrix
        target_sim: NxN target similarity matrix

    Returns:
        NxN numpy array (element-wise maximum)
    """
    return np.maximum(np.maximum(chem_sim, atc_sim), target_sim)


# ============================================================================
# DRUGBANK DATA FETCHING FUNCTIONS
# ============================================================================


def fetch_drug_from_drugbank(drug_name, cache_file="drugbank_cache.json"):
    """
    Fetch drug information from DrugBank API.

    Args:
        drug_name: str, name of drug to fetch
        cache_file: str, path to cache file to avoid repeated requests

    Returns:
        dict with keys: 'smiles', 'atc_codes', 'targets'
    """
    # Load cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except:
            cache = {}

    # Check cache first
    if drug_name in cache:
        return cache[drug_name]

    result = {"smiles": "", "atc_codes": [], "targets": []}

    try:
        # Query DrugBank via their search API
        url = f"https://www.drugbank.ca/api/v1/drugs/search"
        params = {"q": drug_name, "type": "name"}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data and len(data) > 0:
            drug = data[0]

            # Extract SMILES (canonical smiles)
            if "smiles" in drug:
                result["smiles"] = drug["smiles"]

            # Extract ATC codes
            if "atc_codes" in drug:
                result["atc_codes"] = drug["atc_codes"]

            # Extract targets (protein names or IDs)
            if "targets" in drug:
                targets = drug["targets"]
                if isinstance(targets, list):
                    result["targets"] = [
                        t.get("name", t.get("id", ""))
                        for t in targets
                        if isinstance(t, dict)
                    ]
                else:
                    result["targets"] = []

        # Save to cache
        cache[drug_name] = result
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Warning: Failed to fetch {drug_name} from DrugBank: {e}")
        # Try fallback with requests_html for web scraping if API fails
        result = _fetch_drugbank_webscrape(drug_name)
        cache[drug_name] = result
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    return result


def _fetch_drugbank_webscrape(drug_name):
    """
    Fallback: web scrape DrugBank HTML page for drug data.
    """
    result = {"smiles": "", "atc_codes": [], "targets": []}

    try:
        # Construct URL for DrugBank search
        search_url = f"https://www.drugbank.ca/unearth/q?utf8=%E2%9C%93&query={drug_name}&commit=Search"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers, timeout=10)

        # Parse HTML for first match
        if response.status_code == 200 and "drugbank.ca/drugs/" in response.text:
            # Extract first drug link
            import re

            matches = re.findall(r"/drugs/(DB\d+)", response.text)
            if matches:
                drug_id = matches[0]
                drug_url = f"https://www.drugbank.ca/drugs/{drug_id}"
                drug_page = requests.get(drug_url, headers=headers, timeout=10)

                # Extract SMILES
                smiles_match = re.search(
                    r'SMILES["\s:]+([A-Za-z0-9\[\]()=@#$%\\/-]+)', drug_page.text
                )
                if smiles_match:
                    result["smiles"] = smiles_match.group(1)

                # Extract ATC codes
                atc_matches = re.findall(r"([A-Z]\d[A-Z]{0,2}\w?\d?)", drug_page.text)
                result["atc_codes"] = list(set(atc_matches))

    except Exception as e:
        print(f"Warning: Web scrape failed for {drug_name}: {e}")

    return result


# ردیف‌های مربوط به breast یا lung را با استفاده از Disease_Entry و Cell Line
# پیدا کن، سپس همه اسم داروهای یکتای آن subset را استخراج کن
def load_drug_data_from_tsv(tsv_path, disease_type="breast"):
    """
    Load drug combination data from TSV and filter by disease type.

    Args:
        tsv_path: path to "2.General Information of Drug Combination.tsv"
        disease_type: 'breast' or 'lung'

    Returns:
        list of unique drug names appearing in that disease type
    """
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")

    # Filter by disease type
    if disease_type.lower() == "breast":
        mask = df["Disease_Entry"].str.contains(
            r"breast|mammary",
            case=False,
            na=False,
            regex=True,
        ) | df["Cell Line"].str.contains(
            r"\bMCF[- ]?7\b|\bZR[- ]?75\b|\bT47D\b|\bMDA[- ]?MB[- ]?\d+\b|\bSUM\d+\b|\bBT[- ]?\d+\b",
            case=False,
            na=False,
            regex=True,
        )

    elif disease_type.lower() == "lung":
        mask = df["Disease_Entry"].str.contains(
            r"lung|NSCLC|SCLC|non[- ]small",
            case=False,
            na=False,
            regex=True,
        ) | df["Cell Line"].str.contains(
            r"\bA549\b|\bH226\b|\bH1299\b|\bH322\b|\bH460\b",
            case=False,
            na=False,
            regex=True,
        )
    else:
        mask = pd.Series([True] * len(df))

    filtered_df = df[mask]

    # Extract unique drug names
    drug_names = set()
    for col in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
        if col in filtered_df.columns:
            names = filtered_df[col].dropna().astype(str)
            names = names[names != "."]
            drug_names.update(names.unique())

    return sorted(list(drug_names))


def build_breast(root):
    """
    Build breast cancer dataset with ATC and target similarities.

    Loads from "2.General Information of Drug Combination.tsv",
    fetches drug data from DrugBank, and computes similarity matrices.

    Args:
        root: root directory containing the TSV file

    Returns:
        (drug_names, S_atc, S_tgt) where:
            - drug_names: list of unique drug names
            - S_atc: NxN ATC similarity matrix
            - S_tgt: NxN target similarity matrix
    """
    tsv_path = os.path.join(root, "2.General Information of Drug Combination.tsv")

    if not os.path.exists(tsv_path):
        print(f"Warning: TSV file not found at {tsv_path}")
        return [], np.zeros((0, 0)), np.zeros((0, 0))

    # Load drug names
    drug_names = load_drug_data_from_tsv(tsv_path, disease_type="breast")

    if not drug_names:
        print("No breast cancer drugs found in TSV")
        return [], np.zeros((0, 0)), np.zeros((0, 0))

    print(f"Building breast cancer dataset with {len(drug_names)} unique drugs...")

    # Fetch data for each drug
    atc_map = {}
    target_map = {}

    for drug in drug_names:
        print(f"  Fetching data for {drug}...")
        drug_data = fetch_drug_from_drugbank(
            drug, cache_file=os.path.join(root, "drugbank_cache_breast.json")
        )
        atc_map[drug] = drug_data["atc_codes"]
        target_map[drug] = drug_data["targets"]

    # Build similarity matrices
    S_atc = build_atc_similarity(drug_names, atc_map)
    S_tgt = build_target_similarity(drug_names, target_map)

    print(f"Breast cancer dataset ready: {len(drug_names)} drugs, {S_atc.shape}")

    return drug_names, S_atc, S_tgt


def build_lung(root):
    """
    Build lung cancer dataset with ATC and target similarities.

    Loads from "2.General Information of Drug Combination.tsv",
    fetches drug data from DrugBank, and computes similarity matrices.

    Args:
        root: root directory containing the TSV file

    Returns:
        (drug_names, S_atc, S_tgt) where:
            - drug_names: list of unique drug names
            - S_atc: NxN ATC similarity matrix
            - S_tgt: NxN target similarity matrix
    """
    tsv_path = os.path.join(root, "2.General Information of Drug Combination.tsv")

    if not os.path.exists(tsv_path):
        print(f"Warning: TSV file not found at {tsv_path}")
        return [], np.zeros((0, 0)), np.zeros((0, 0))

    # Load drug names
    drug_names = load_drug_data_from_tsv(tsv_path, disease_type="lung")

    if not drug_names:
        print("No lung cancer drugs found in TSV")
        return [], np.zeros((0, 0)), np.zeros((0, 0))

    print(f"Building lung cancer dataset with {len(drug_names)} unique drugs...")

    # Fetch data for each drug
    atc_map = {}
    target_map = {}

    for drug in drug_names:
        print(f"  Fetching data for {drug}...")
        drug_data = fetch_drug_from_drugbank(
            drug, cache_file=os.path.join(root, "drugbank_cache_lung.json")
        )
        atc_map[drug] = drug_data["atc_codes"]
        target_map[drug] = drug_data["targets"]

    # Build similarity matrices
    S_atc = build_atc_similarity(drug_names, atc_map)
    S_tgt = build_target_similarity(drug_names, target_map)

    print(f"Lung cancer dataset ready: {len(drug_names)} drugs, {S_atc.shape}")

    return drug_names, S_atc, S_tgt


def build_feature_map_atc(atc_path):
    """
    Load ATC codes from Excel file and build drug -> ATC codes mapping.

    Args:
        atc_path: path to ATC Excel file (e.g., "ATC_Breast.xlsx")

    Returns:
        dict mapping drug name -> list of ATC codes
    """
    feature_map = {}

    if not os.path.exists(atc_path):
        print(f"Warning: ATC file not found at {atc_path}")
        return feature_map

    try:
        df = pd.read_excel(atc_path)

        # Assume first column is drug name, remaining columns are ATC codes
        if len(df.columns) > 0:
            drug_col = df.columns[0]
            atc_cols = df.columns[1:]

            for _, row in df.iterrows():
                drug_name = str(row[drug_col]).strip()
                atc_codes = []

                for col in atc_cols:
                    val = str(row[col]).strip()
                    if val and val != "nan" and val != ".":
                        atc_codes.append(val)

                feature_map[drug_name] = atc_codes

    except Exception as e:
        print(f"Error reading ATC file {atc_path}: {e}")

    return feature_map


def build_feature_map_target(target_path):
    """
    Load protein targets from Excel file and build drug -> targets mapping.

    Args:
        target_path: path to Target Excel file (e.g., "Target_Breast.xlsx")

    Returns:
        dict mapping drug name -> list of protein target names/IDs
    """
    feature_map = {}

    if not os.path.exists(target_path):
        print(f"Warning: Target file not found at {target_path}")
        return feature_map

    try:
        df = pd.read_excel(target_path)

        # Assume first column is drug name, remaining columns are targets
        if len(df.columns) > 0:
            drug_col = df.columns[0]
            target_cols = df.columns[1:]

            for _, row in df.iterrows():
                drug_name = str(row[drug_col]).strip()
                targets = []

                for col in target_cols:
                    val = str(row[col]).strip()
                    if val and val != "nan" and val != ".":
                        targets.append(val)

                feature_map[drug_name] = targets

    except Exception as e:
        print(f"Error reading Target file {target_path}: {e}")

    return feature_map


if __name__ == "__main__":
    # Example usage
    root = "."

    # Build breast cancer dataset
    names_breast, S_atc_breast, S_tgt_breast = build_breast(root)
    print(f"Breast: {len(names_breast)} drugs")
    print(f"S_atc shape: {S_atc_breast.shape}, S_tgt shape: {S_tgt_breast.shape}")

    # Build lung cancer dataset
    names_lung, S_atc_lung, S_tgt_lung = build_lung(root)
    print(f"Lung: {len(names_lung)} drugs")
    print(f"S_atc shape: {S_atc_lung.shape}, S_tgt shape: {S_tgt_lung.shape}")
