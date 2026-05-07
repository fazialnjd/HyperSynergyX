import requests

# this module is older i have a better one 


# ---------------------------
# 1. ChEMBL search
# ---------------------------
def search_chembl(drug_name):

    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name}&format=json"

    r = requests.get(url)

    if r.status_code != 200:
        return None

    data = r.json()

    mols = data.get("molecules", [])

    if not mols:
        return None

    return mols[0]


# ---------------------------
# 2. SMILES
# ---------------------------
def get_smiles(mol):

    return mol.get("molecule_structures", {}).get("canonical_smiles")


# ---------------------------
# 3. ATC
# ---------------------------
def get_atc(mol):

    return mol.get("atc_classifications", [])


# ---------------------------
# 4. RAW targets from ChEMBL
# ---------------------------
def get_raw_targets(chembl_id):

    url = (
        "https://www.ebi.ac.uk/chembl/api/data/target.json"
        f"?molecule_chembl_id={chembl_id}&limit=500"
    )

    r = requests.get(url)

    if r.status_code != 200:
        return []

    data = r.json()

    targets = []

    for t in data.get("targets", []):

        name = t.get("pref_name")
        organism = t.get("organism")
        target_type = t.get("target_type")

        if not name:
            continue

        targets.append({"name": name, "organism": organism, "type": target_type})

    return targets


# ---------------------------
# 5. CLEANING (IMPORTANT PART)
# ---------------------------
# TODO: it return too many targets
def clean_targets(raw_targets):

    keep_keywords = [
        "thymidylate synthase",
        "dihydropyrimidine dehydrogenase",
        "cyclooxygenase",
        "prostaglandin",
        "lipoxygenase",
        "reductase",
        "dehydrogenase",
        "kinase",
        "topoisomerase",
        "synthase",
    ]

    bad_keywords = [
        "homo",
        "mus",
        "rattus",
        "oryctolagus",
        "cavia",
        "trichomonas",
        "sp.",
        "unknown",
        "unchecked",
    ]

    clean = set()

    for t in raw_targets:

        name = t["name"]
        org = str(t["organism"]).lower() if t["organism"] else ""
        typ = str(t["type"]).lower() if t["type"] else ""

        low = name.lower()

        # must be human protein
        if org != "homo sapiens":
            continue

        # must be protein target
        if "protein" not in typ and typ != "single protein":
            continue

        # remove noise
        if any(b in low for b in bad_keywords):
            continue

        # must match biological relevance
        if any(k in low for k in keep_keywords):
            clean.add(name)

    return list(clean)


# ---------------------------
# 6. MAIN PIPELINE
# ---------------------------
def get_drug_info(drug_name):

    mol = search_chembl(drug_name)

    if not mol:
        return None

    chembl_id = mol["molecule_chembl_id"]

    raw_targets = get_raw_targets(chembl_id)

    return {
        "drug": drug_name,
        "chembl_id": chembl_id,
        "smiles": get_smiles(mol),
        "atc_codes": get_atc(mol),
        "targets": clean_targets(raw_targets),
    }


# ---------------------------
# TEST
# ---------------------------
if __name__ == "__main__":

    # print(get_drug_info("fluorouracil"))
    print(get_drug_info("aspirin"))
