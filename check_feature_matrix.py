import json
import numpy as np

with open("similarity_matrices/breast_drug_names.json") as f:
    drug_names = json.load(f)

print("Number of drugs:", len(drug_names))
print(drug_names)
