# compute_synergy_scores.py
# the same approach of dbrwh.py

import numpy as np
import pandas as pd
import sys

sys.path.append(
    "/home/faezeh/uni/progect/HyperSynergyX"
)  # Add path to HyperSynergyX repo

from dbrwh import load_incidence, dbrwh_scores, combine_sims


def compute_synergy_for_dataset(
    incidence_matrix_path, output_path, alpha=0.8, sim_matrix_path=None
):
    """
    Compute synergy scores using DBRWH for a given incidence matrix.

    Parameters:
    -----------
    incidence_matrix_path : str
        Path to the TSV/CSV file containing the incidence matrix
    output_path : str
        Path to save the output synergy scores
    alpha : float
        Damping factor for the random walk (default: 0.8)
    sim_matrix_path : str, optional
        Path to drug similarity matrix (if not provided, uses uniform similarity)

    Returns:
    --------
    synergy_scores : np.ndarray
        3D array of shape (n_drugs, n_drugs, n_drugs) with synergy scores
    drug_names : list
        Names of drugs used in the analysis
    """

    print(f"Loading incidence matrix from {incidence_matrix_path}...")
    drug_names, H = load_incidence(incidence_matrix_path)

    print(f"Loaded incidence matrix with shape: {H.shape}")
    print(f"Number of drugs: {len(drug_names)}")
    print(f"Number of conditions/samples: {H.shape[1]}")

    # Load similarity matrix if provided
    Sim = None
    if sim_matrix_path:
        print(f"Loading similarity matrix from {sim_matrix_path}...")
        _, Sim = load_incidence(sim_matrix_path)
        print(f"Loaded similarity matrix with shape: {Sim.shape}")

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

    import os

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
    print("\nTop 20 Drug Combinations:")
    print(top_20.to_string())
    top_20.to_csv(os.path.join(output_dir, "synergy_top_20.csv"), index=False)

    # 4. Save pairwise average synergy scores
    pairwise_scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Average synergy when drug i and j are paired with all others
                scores_with_ij = []
                for k in range(n):
                    if k != i and k != j:
                        scores_with_ij.append(synergy_scores[i, j, k])
                pairwise_scores[i, j] = np.mean(scores_with_ij) if scores_with_ij else 0

    pairwise_df = pd.DataFrame(pairwise_scores, index=drug_names, columns=drug_names)
    pairwise_df.to_csv(os.path.join(output_dir, "pairwise_synergy_scores.csv"))
    print(f"Saved pairwise synergy scores to {output_dir}/pairwise_synergy_scores.csv")

    return triplets_df


def main():
    """
    Main function to compute synergy scores for both breast and lung datasets.
    """

    # Configuration
    breast_incidence = "/home/faezeh/uni/progect/HyperSynergyX/hypergraph_output/breast_incidence_matrix.tsv"
    lung_incidence = "/home/faezeh/uni/progect/HyperSynergyX/hypergraph_output/lung_incidence_matrix.tsv"

    # Optional: similarity matrices (uncomment if you have them)
    # breast_sim = 'breast_similarity_matrix.tsv'
    # lung_sim = 'lung_similarity_matrix.tsv'

    alpha = 0.8  # Damping factor for random walk

    # Compute synergy scores for breast cancer
    print("=" * 60)
    print("BREAST CANCER DATASET")
    print("=" * 60)
    breast_scores, breast_drugs = compute_synergy_for_dataset(
        breast_incidence,
        "output/breast_synergy",
        alpha=alpha,
        sim_matrix_path=None,  # Set to 'breast_similarity_matrix.tsv' if available
    )

    print("\n" + "=" * 60)
    print("LUNG CANCER DATASET")
    print("=" * 60)
    # Compute synergy scores for lung cancer
    lung_scores, lung_drugs = compute_synergy_for_dataset(
        lung_incidence,
        "output/lung_synergy",
        alpha=alpha,
        sim_matrix_path=None,  # Set to 'lung_similarity_matrix.tsv' if available
    )

    # Optional: Compare datasets
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Breast - Computed synergy for {len(breast_drugs)} drugs")
    print(f"Lung   - Computed synergy for {len(lung_drugs)} drugs")
    print("\nResults saved to:")
    print("  - output/breast_synergy/")
    print("  - output/lung_synergy/")


if __name__ == "__main__":
    main()
