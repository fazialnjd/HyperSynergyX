# compute_synergy_scores.py
# the same approach of dbrwh.py

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append("/home/faezeh/uni/progect/HyperSynergyX")

from dbrwh import load_incidence, dbrwh_scores


def load_incidence_matrix(incidence_path):
    """
    Load incidence matrix from TSV file.

    Parameters:
    -----------
    incidence_path : str
        Path to TSV file with drugs as index and combinations as columns

    Returns:
    --------
    drug_names : list
        List of drug names
    H : np.ndarray
        Incidence matrix (n_drugs x n_combinations)
    """
    df = pd.read_csv(incidence_path, sep="\t", index_col=0)
    drug_names = df.index.tolist()
    H = df.values.astype(float)
    return drug_names, H


def load_similarity_matrix(sim_path):
    """
    Load similarity matrix from NPY file.

    Parameters:
    -----------
    sim_path : str
        Path to NPY file containing similarity matrix

    Returns:
    --------
    Sim : np.ndarray
        Similarity matrix (n_drugs x n_drugs)
    """
    if sim_path.endswith(".npy"):
        Sim = np.load(sim_path)
    else:
        # Try loading as TSV
        df = pd.read_csv(sim_path, sep="\t", index_col=0)
        Sim = df.values.astype(float)
    return Sim


def compute_synergy_for_dataset(
    incidence_matrix_path,
    output_path,
    alpha=0.8,
    sim_atc_path=None,
    sim_target_path=None,
    sim_integrated_path=None,
):
    """
    Compute synergy scores using DBRWH for a given incidence matrix.

    Parameters:
    -----------
    incidence_matrix_path : str
        Path to the TSV file containing the incidence matrix
    output_path : str
        Path to save the output synergy scores
    alpha : float
        Damping factor for the random walk (default: 0.8)
    sim_atc_path : str, optional
        Path to ATC similarity matrix NPY file
    sim_target_path : str, optional
        Path to Target similarity matrix NPY file
    sim_integrated_path : str, optional
        Path to Integrated similarity matrix NPY file

    Returns:
    --------
    synergy_scores : np.ndarray
        3D array of shape (n_drugs, n_drugs, n_drugs) with synergy scores
    drug_names : list
        Names of drugs used in the analysis
    """

    print(f"Loading incidence matrix from {incidence_matrix_path}...")
    drug_names, H = load_incidence_matrix(incidence_matrix_path)

    print(f"Loaded incidence matrix with shape: {H.shape}")
    print(f"Number of drugs: {len(drug_names)}")
    print(f"Number of hyperedges (combinations): {H.shape[1]}")

    # Load similarity matrix (prioritize integrated, then ATC, then Target)
    Sim = None
    if sim_integrated_path and os.path.exists(sim_integrated_path):
        print(f"Loading integrated similarity matrix from {sim_integrated_path}...")
        Sim = load_similarity_matrix(sim_integrated_path)
        print(f"Loaded integrated similarity matrix with shape: {Sim.shape}")
    elif sim_atc_path and os.path.exists(sim_atc_path):
        print(f"Loading ATC similarity matrix from {sim_atc_path}...")
        Sim = load_similarity_matrix(sim_atc_path)
        print(f"Loaded ATC similarity matrix with shape: {Sim.shape}")
    elif sim_target_path and os.path.exists(sim_target_path):
        print(f"Loading Target similarity matrix from {sim_target_path}...")
        Sim = load_similarity_matrix(sim_target_path)
        print(f"Loaded Target similarity matrix with shape: {Sim.shape}")
    else:
        print("No similarity matrix provided. Using uniform similarity (identity).")
        Sim = np.eye(len(drug_names))

    # Compute synergy scores using DBRWH
    print(f"\nComputing synergy scores with alpha={alpha}...")
    synergy_scores = dbrwh_scores(
        H, alpha=alpha, Sim=Sim, restart_bias=True, attraction_bias=True
    )

    print(f"Synergy scores computed with shape: {synergy_scores.shape}")
    print(f"Score range: [{synergy_scores.min():.4f}, {synergy_scores.max():.4f}]")

    # Save results
    save_synergy_results(synergy_scores, drug_names, output_path)

    return synergy_scores, drug_names


def save_synergy_results(synergy_scores, drug_names, output_dir):
    """
    Save synergy scores in multiple formats for analysis.

    Parameters:
    -----------
    synergy_scores : np.ndarray
        3D array of synergy scores (n_drugs, n_drugs, n_drugs)
    drug_names : list
        Names of drugs
    output_dir : str
        Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(drug_names)

    # 1. Save full 3D array as NPZ (binary format)
    np.savez(
        os.path.join(output_dir, "synergy_scores_3d.npz"),
        scores=synergy_scores,
        drug_names=np.array(drug_names),
    )
    print(f"Saved 3D synergy scores to {output_dir}/synergy_scores_3d.npz")

    # 2. Extract and save top drug combinations as CSV
    triplets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                score = synergy_scores[i, j, k]
                triplets.append(
                    {
                        "drug_1": drug_names[i],
                        "drug_2": drug_names[j],
                        "drug_3": drug_names[k],
                        "synergy_score": score,
                    }
                )

    triplets_df = pd.DataFrame(triplets)
    triplets_df = triplets_df.sort_values("synergy_score", ascending=False)
    triplets_df.to_csv(
        os.path.join(output_dir, "synergy_triplets_ranked.csv"), index=False
    )
    print(f"Saved ranked triplets to {output_dir}/synergy_triplets_ranked.csv")

    # 3. Top 20 combinations
    top_20 = triplets_df.head(20)
    print("\n" + "=" * 60)
    print("TOP 20 DRUG COMBINATIONS")
    print("=" * 60)
    print(top_20.to_string(index=False))
    top_20.to_csv(os.path.join(output_dir, "synergy_top_20.csv"), index=False)

    # 4. Save pairwise average synergy scores
    pairwise_scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                scores_with_ij = []
                for k in range(n):
                    if k != i and k != j:
                        scores_with_ij.append(synergy_scores[i, j, k])
                pairwise_scores[i, j] = np.mean(scores_with_ij) if scores_with_ij else 0

    pairwise_df = pd.DataFrame(pairwise_scores, index=drug_names, columns=drug_names)
    pairwise_df.to_csv(os.path.join(output_dir, "pairwise_synergy_scores.csv"))
    print(f"Saved pairwise synergy scores to {output_dir}/pairwise_synergy_scores.csv")

    # 5. Summary statistics
    print("\n" + "=" * 60)
    print("SYNERGY SCORES SUMMARY")
    print("=" * 60)
    print(f"Total triplets evaluated: {len(triplets_df)}")
    print(f"Mean synergy score: {triplets_df['synergy_score'].mean():.4f}")
    print(f"Std synergy score: {triplets_df['synergy_score'].std():.4f}")
    print(f"Max synergy score: {triplets_df['synergy_score'].max():.4f}")
    print(f"Min synergy score: {triplets_df['synergy_score'].min():.4f}")

    return triplets_df


def main():
    """
    Main function to compute synergy scores for both breast and lung datasets.
    """
    # Paths to incidence matrices (from hypergraph construction)
    base_dir = Path("/home/faezeh/uni/progect/HyperSynergyX")
    incidence_dir = base_dir / "hypergraph_output"
    similarity_dir = base_dir / "similarity_matrices"

    breast_incidence = incidence_dir / "breast_incidence_matrix.tsv"
    lung_incidence = incidence_dir / "lung_incidence_matrix.tsv"

    # Similarity matrices (from sim_builder.py)
    breast_sim_integrated = similarity_dir / "breast_S_int.npy"
    breast_sim_atc = similarity_dir / "breast_S_atc.npy"
    breast_sim_target = similarity_dir / "breast_S_tgt.npy"

    lung_sim_integrated = similarity_dir / "lung_S_int.npy"
    lung_sim_atc = similarity_dir / "lung_S_atc.npy"
    lung_sim_target = similarity_dir / "lung_S_tgt.npy"

    alpha = 0.8  # Damping factor for random walk

    # Create output directories
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # ========================================================================
    # BREAST CANCER DATASET
    # ========================================================================
    print("\n" + "=" * 70)
    print("BREAST CANCER DATASET")
    print("=" * 70)

    if breast_incidence.exists():
        print(f"\n✅ Found incidence matrix: {breast_incidence}")

        # Try integrated similarity first
        sim_path = None
        sim_type = "none"

        if breast_sim_integrated.exists():
            sim_path = breast_sim_integrated
            sim_type = "integrated (max of ATC and Target)"
        elif breast_sim_atc.exists():
            sim_path = breast_sim_atc
            sim_type = "ATC"
        elif breast_sim_target.exists():
            sim_path = breast_sim_target
            sim_type = "Target"

        print(f"Using similarity matrix: {sim_type}")

        breast_scores, breast_drugs = compute_synergy_for_dataset(
            str(breast_incidence),
            str(output_dir / "breast_synergy"),
            alpha=alpha,
            sim_integrated_path=(
                str(breast_sim_integrated) if breast_sim_integrated.exists() else None
            ),
            sim_atc_path=str(breast_sim_atc) if breast_sim_atc.exists() else None,
            sim_target_path=(
                str(breast_sim_target) if breast_sim_target.exists() else None
            ),
        )
    else:
        print(f"\n❌ Incidence matrix not found: {breast_incidence}")
        breast_drugs = []

    # ========================================================================
    # LUNG CANCER DATASET
    # ========================================================================
    print("\n" + "=" * 70)
    print("LUNG CANCER DATASET")
    print("=" * 70)

    if lung_incidence.exists():
        print(f"\n✅ Found incidence matrix: {lung_incidence}")

        # Try integrated similarity first
        if lung_sim_integrated.exists():
            sim_path = lung_sim_integrated
            sim_type = "integrated (max of ATC and Target)"
        elif lung_sim_atc.exists():
            sim_path = lung_sim_atc
            sim_type = "ATC"
        elif lung_sim_target.exists():
            sim_path = lung_sim_target
            sim_type = "Target"
        else:
            sim_path = None
            sim_type = "none"

        print(f"Using similarity matrix: {sim_type}")

        lung_scores, lung_drugs = compute_synergy_for_dataset(
            str(lung_incidence),
            str(output_dir / "lung_synergy"),
            alpha=alpha,
            sim_integrated_path=(
                str(lung_sim_integrated) if lung_sim_integrated.exists() else None
            ),
            sim_atc_path=str(lung_sim_atc) if lung_sim_atc.exists() else None,
            sim_target_path=str(lung_sim_target) if lung_sim_target.exists() else None,
        )
    else:
        print(f"\n❌ Incidence matrix not found: {lung_incidence}")
        lung_drugs = []

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n📊 Breast Cancer:")
    print(f"   Number of drugs: {len(breast_drugs)}")
    print(f"   Results saved to: {output_dir / 'breast_synergy'}/")

    print(f"\n📊 Lung Cancer:")
    print(f"   Number of drugs: {len(lung_drugs)}")
    print(f"   Results saved to: {output_dir / 'lung_synergy'}/")

    print("\n✅ All synergy scores computed successfully!")
    print("\nOutput files:")
    print("  - synergy_scores_3d.npz     : Full 3D synergy tensor")
    print("  - synergy_triplets_ranked.csv : All triplets ranked by score")
    print("  - synergy_top_20.csv         : Top 20 combinations")
    print("  - pairwise_synergy_scores.csv : Pairwise average synergy")


if __name__ == "__main__":
    main()
