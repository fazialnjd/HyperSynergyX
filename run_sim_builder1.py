import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
# SIMILARITY CALCULATION FUNCTIONS
# ============================================================================


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

    # Extract ATC codes by level (first 3 levels: e.g., "L01BC05" -> levels L, L01, L01B)
    atc_by_level = {}
    for drug in drug_names:
        atc_by_level[drug] = [set(), set(), set()]  # levels 1, 2, 3

        atc_codes = atc_map.get(drug, [])
        if isinstance(atc_codes, str):
            atc_codes = [atc_codes]

        for code in atc_codes:
            if code and len(str(code).strip()) > 0:
                code_str = str(code).strip()
                # Level 1: first character (e.g., "L")
                if len(code_str) >= 1:
                    atc_by_level[drug][0].add(code_str[0])
                # Level 2: first 3 characters (e.g., "L01")
                if len(code_str) >= 3:
                    atc_by_level[drug][1].add(code_str[:3])
                # Level 3: first 4 characters (e.g., "L01B")
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
        target_map: dict mapping drug name -> list/set of gene_symbols

    Returns:
        NxN numpy array of target similarities
    """
    n = len(drug_names)
    target_sim = np.zeros((n, n), dtype=float)

    # Normalize target sets (extract gene_symbols)
    targets = {}
    for drug in drug_names:
        target_list = target_map.get(drug, [])
        if target_list and isinstance(target_list[0], dict):
            # Extract gene_symbol from dict
            gene_set = set()
            for t in target_list:
                if isinstance(t, dict) and t.get("gene_symbol"):
                    gene_set.add(t["gene_symbol"])
            targets[drug] = gene_set
        else:
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
# MAIN: Load from cleaned JSON and compute similarities
# ============================================================================


def load_drugs_from_cleaned_json(json_path):
    """
    Load drug information from cleaned JSON file.

    Args:
        json_path: path to cleaned JSON file (e.g., breast_3drug_drug_info_strictly_cleaned.json)

    Returns:
        drug_names: list of unique drug names
        atc_map: dict mapping drug name -> list of ATC codes
        target_map: dict mapping drug name -> list of targets (with gene_symbol)
        smiles_map: dict mapping drug name -> SMILES string
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    atc_map = {}
    target_map = {}
    smiles_map = {}

    for combo_id, combo_data in data.items():
        for drug_key in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            if drug_key in combo_data:
                drug_info = combo_data[drug_key]
                if drug_info and drug_info.get("drug"):
                    drug_name = drug_info["drug"]

                    # ATC codes
                    if drug_name not in atc_map:
                        atc_codes = drug_info.get("atc_codes", [])
                        if isinstance(atc_codes, list):
                            atc_map[drug_name] = atc_codes
                        else:
                            atc_map[drug_name] = []

                    # Targets (curated_targets with gene_symbol)
                    if drug_name not in target_map:
                        targets = drug_info.get("curated_targets", [])
                        target_map[drug_name] = targets if targets else []

                    # SMILES
                    if drug_name not in smiles_map:
                        smiles = drug_info.get("smiles", "")
                        smiles_map[drug_name] = smiles if smiles else ""

    # Convert to sorted list for consistent ordering
    drug_names = sorted(list(atc_map.keys()))

    return drug_names, atc_map, target_map, smiles_map


def build_similarities_from_cleaned_json(json_path, output_prefix=None):
    """
    Build ATC and Target similarity matrices from cleaned JSON file.

    Args:
        json_path: path to cleaned JSON file
        output_prefix: optional prefix for saving output files

    Returns:
        dict containing:
            - drug_names: list of drug names
            - S_atc: ATC similarity matrix
            - S_tgt: Target similarity matrix
            - S_int: Integrated similarity (max of ATC and Target)
            - n_drugs: number of drugs
    """
    print("=" * 70)
    print(f"Loading data from: {json_path}")
    print("=" * 70)

    # Load data
    drug_names, atc_map, target_map, smiles_map = load_drugs_from_cleaned_json(
        json_path
    )

    print(f"\n📊 Found {len(drug_names)} unique drugs")

    # Show sample of loaded data
    print("\n📋 Sample data (first 5 drugs):")
    for drug in drug_names[:5]:
        atc_codes = atc_map.get(drug, [])
        targets = target_map.get(drug, [])
        target_count = len(targets) if targets else 0
        print(f"   - {drug}: {len(atc_codes)} ATC codes, {target_count} targets")

    # Build ATC similarity matrix
    print("\n🔧 Building ATC similarity matrix...")
    S_atc = build_atc_similarity(drug_names, atc_map)

    # Build Target similarity matrix
    print("🔧 Building Target similarity matrix...")
    S_tgt = build_target_similarity(drug_names, target_map)

    # Build integrated similarity (max of ATC and Target)
    print("🔧 Building Integrated similarity matrix (max)...")
    S_int = build_integrated_similarity(np.zeros_like(S_atc), S_atc, S_tgt)

    print(f"\n✅ Similarity matrices built:")
    print(f"   S_atc shape: {S_atc.shape}")
    print(f"   S_tgt shape: {S_tgt.shape}")
    print(f"   S_int shape: {S_int.shape}")

    # Print sample similarities
    print("\n📊 Sample similarities (first 5 drugs):")
    for i in range(min(5, len(drug_names))):
        for j in range(i + 1, min(5, len(drug_names))):
            print(
                f"   {drug_names[i]:20s} vs {drug_names[j]:20s} | ATC: {S_atc[i,j]:.4f} | Target: {S_tgt[i,j]:.4f} | Int: {S_int[i,j]:.4f}"
            )

    # Save outputs if requested
    if output_prefix:
        # Save drug names
        with open(f"{output_prefix}_drug_names.json", "w") as f:
            json.dump(drug_names, f, indent=2)

        # Save similarity matrices
        np.save(f"{output_prefix}_S_atc.npy", S_atc)
        np.save(f"{output_prefix}_S_tgt.npy", S_tgt)
        np.save(f"{output_prefix}_S_int.npy", S_int)

        print(f"\n💾 Saved outputs with prefix: {output_prefix}")
        print(f"   - {output_prefix}_drug_names.json")
        print(f"   - {output_prefix}_S_atc.npy")
        print(f"   - {output_prefix}_S_tgt.npy")
        print(f"   - {output_prefix}_S_int.npy")

    return {
        "drug_names": drug_names,
        "S_atc": S_atc,
        "S_tgt": S_tgt,
        "S_int": S_int,
        "n_drugs": len(drug_names),
    }


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Path to cleaned JSON files
    breast_json = Path(
        "drug_info_output/strictly_cleaned_no_chembl/breast_3drug_drug_info_strictly_cleaned.json"
    )
    lung_json = Path(
        "drug_info_output/strictly_cleaned_no_chembl/lung_3drug_drug_info_strictly_cleaned.json"
    )

    # Output directory for similarity matrices
    output_dir = Path("similarity_matrices")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("SIMILARITY MATRIX BUILDER")
    print("=" * 70)

    # Process Breast Cancer data
    if breast_json.exists():
        print("\n" + "=" * 70)
        print("BREAST CANCER DATASET")
        print("=" * 70)

        breast_result = build_similarities_from_cleaned_json(
            breast_json, output_prefix=str(output_dir / "breast")
        )
    else:
        print(f"\n❌ Breast cancer file not found: {breast_json}")

    # Process Lung Cancer data
    if lung_json.exists():
        print("\n" + "=" * 70)
        print("LUNG CANCER DATASET")
        print("=" * 70)

        lung_result = build_similarities_from_cleaned_json(
            lung_json, output_prefix=str(output_dir / "lung")
        )
    else:
        print(f"\n❌ Lung cancer file not found: {lung_json}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if breast_json.exists():
        print(f"\n📁 Breast Cancer:")
        print(f"   Drugs: {breast_result['n_drugs']}")
        print(f"   ATC similarity matrix: {breast_result['S_atc'].shape}")
        print(f"   Target similarity matrix: {breast_result['S_tgt'].shape}")

    if lung_json.exists():
        print(f"\n📁 Lung Cancer:")
        print(f"   Drugs: {lung_result['n_drugs']}")
        print(f"   ATC similarity matrix: {lung_result['S_atc'].shape}")
        print(f"   Target similarity matrix: {lung_result['S_tgt'].shape}")

    print(f"\n✅ All outputs saved to: {output_dir}/")
