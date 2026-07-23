"""
analyze_similarities.py
========================
تحلیل و ویژوالیزیشن ماتریس‌های شباهت داروها
Breast Cancer (27 drugs) + Lung Cancer (9 drugs)

Outputs (saved to similarity_analysis/):
  01_heatmaps_breast.png
  02_heatmaps_lung.png
  03_clustermaps_breast.png
  04_clustermaps_lung.png
  05_distributions.png
  06_network_breast.png
  07_network_lung.png
  08_similarity_comparison.png

Run:
    python analyze_similarities.py
"""

import numpy as np
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Suppress RDKit deprecation noise if rdkit is imported elsewhere ──────────
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("similarity_matrices")
OUTPUT_DIR = Path("similarity_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "chem":   "#0279EE",
    "atc":    "#FF9400",
    "target": "#75A025",
    "int":    "#E9ED4C",
}
CMAP_HEAT = "YlOrRd"
SIM_LABELS = {
    "S_chem": "Chemical (Tanimoto)",
    "S_atc":  "ATC (Jaccard)",
    "S_tgt":  "Target (Jaccard)",
    "S_int":  "Integrated (max)",
}
CANCER_COLORS = {"breast": "#FD9BED", "lung": "#0279EE"}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi":   150,
})


# ── Load data ────────────────────────────────────────────────────────────────
def load_dataset(cancer: str) -> dict:
    d = {}
    d["drug_names"] = json.load(open(BASE_DIR / f"{cancer}_drug_names.json"))
    for key in ["S_chem", "S_atc", "S_tgt", "S_int"]:
        d[key] = np.load(BASE_DIR / f"{cancer}_{key}.npy")
    return d


print("Loading data...")
datasets = {c: load_dataset(c) for c in ["breast", "lung"]}
for c, d in datasets.items():
    n = len(d["drug_names"])
    print(f"  {c}: {n} drugs loaded")


# ── Helper: upper-triangle values (no diagonal) ──────────────────────────────
def upper_tri(mat):
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


# ============================================================================
# STEP 1 — HEATMAPS
# ============================================================================
def plot_heatmaps(cancer: str, data: dict, out_path: Path):
    drug_names = data["drug_names"]
    n = len(drug_names)
    annotate = n <= 12   # annotate cell values only for small matrices

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Drug Similarity Matrices — {cancer.capitalize()} Cancer ({n} drugs)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ax, (key, label) in zip(axes.flat, SIM_LABELS.items()):
        mat = data[key]
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=drug_names,
            yticklabels=drug_names,
            cmap=CMAP_HEAT,
            vmin=0, vmax=1,
            annot=annotate,
            fmt=".2f" if annotate else "",
            linewidths=0.3 if annotate else 0,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Similarity"},
        )
        ax.set_title(label, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7 if n > 15 else 9)
        ax.tick_params(axis="y", rotation=0,  labelsize=7 if n > 15 else 9)

        # stats annotation
        vals = upper_tri(mat)
        ax.text(
            0.02, 0.98,
            f"mean={vals.mean():.2f}  max={vals.max():.2f}",
            transform=ax.transAxes,
            fontsize=8, va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("\n[1/5] Heatmaps...")
for cancer, data in datasets.items():
    plot_heatmaps(cancer, data, OUTPUT_DIR / f"0{'1' if cancer=='breast' else '2'}_heatmaps_{cancer}.png")


# ============================================================================
# STEP 2 — CLUSTERMAPS
# ============================================================================
def plot_clustermaps(cancer: str, data: dict, out_path: Path):
    drug_names = data["drug_names"]
    n = len(drug_names)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(
        f"Drug Similarity Clustermaps — {cancer.capitalize()} Cancer ({n} drugs)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ax, (key, label) in zip(axes.flat, SIM_LABELS.items()):
        mat = data[key]

        # Convert similarity to distance for clustering
        dist = 1.0 - mat
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)

        # Hierarchical clustering (average linkage)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")

        # Get leaf order from dendrogram
        dend = dendrogram(Z, no_plot=True)
        order = dend["leaves"]

        # Reorder matrix
        mat_ord = mat[np.ix_(order, order)]
        names_ord = [drug_names[i] for i in order]

        annotate = n <= 12
        sns.heatmap(
            mat_ord,
            ax=ax,
            xticklabels=names_ord,
            yticklabels=names_ord,
            cmap=CMAP_HEAT,
            vmin=0, vmax=1,
            annot=annotate,
            fmt=".2f" if annotate else "",
            linewidths=0.3 if annotate else 0,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Similarity"},
        )
        ax.set_title(f"{label}\n(hierarchical clustering, average linkage)",
                     fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7 if n > 15 else 9)
        ax.tick_params(axis="y", rotation=0,  labelsize=7 if n > 15 else 9)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("\n[2/5] Clustermaps...")
for cancer, data in datasets.items():
    plot_clustermaps(cancer, data, OUTPUT_DIR / f"0{'3' if cancer=='breast' else '4'}_clustermaps_{cancer}.png")


# ============================================================================
# STEP 3 — DISTRIBUTION PLOTS
# ============================================================================
def plot_distributions(datasets: dict, out_path: Path):
    sim_keys = list(SIM_LABELS.keys())
    n_sim = len(sim_keys)

    fig, axes = plt.subplots(2, n_sim, figsize=(18, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle(
        "Distribution of Pairwise Similarity Values\n(upper triangle, diagonal excluded)",
        fontsize=13, fontweight="bold"
    )

    for col, (key, label) in enumerate(SIM_LABELS.items()):
        # Row 0: KDE overlay for both cancers
        ax_kde = axes[0, col]
        for cancer, data in datasets.items():
            vals = upper_tri(data[key])
            color = CANCER_COLORS[cancer]
            ax_kde.hist(vals, bins=20, alpha=0.35, color=color, density=True)
            vals_smooth = np.linspace(0, 1, 300)
            from scipy.stats import gaussian_kde
            if vals.std() > 0:
                kde = gaussian_kde(vals, bw_method=0.3)
                ax_kde.plot(vals_smooth, kde(vals_smooth), color=color,
                            lw=2, label=f"{cancer} (μ={vals.mean():.2f})")
            else:
                ax_kde.axvline(vals.mean(), color=color, lw=2,
                               label=f"{cancer} (μ={vals.mean():.2f})")
        ax_kde.set_title(label, fontweight="bold")
        ax_kde.set_xlabel("Similarity")
        ax_kde.set_ylabel("Density")
        ax_kde.set_xlim(-0.05, 1.05)
        ax_kde.legend(fontsize=8)
        ax_kde.grid(alpha=0.3)

        # Row 1: Box plots side by side
        ax_box = axes[1, col]
        box_data, box_labels, box_colors = [], [], []
        for cancer, data in datasets.items():
            vals = upper_tri(data[key])
            box_data.append(vals)
            box_labels.append(cancer)
            box_colors.append(CANCER_COLORS[cancer])

        bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_xticks([1, 2])
        ax_box.set_xticklabels(box_labels)
        ax_box.set_ylabel("Similarity")
        ax_box.set_ylim(-0.05, 1.05)
        ax_box.grid(alpha=0.3, axis="y")

        # Add mean ± std text
        for i, (cancer, data) in enumerate(datasets.items(), 1):
            vals = upper_tri(data[key])
            ax_box.text(i, 1.02, f"μ={vals.mean():.2f}\nσ={vals.std():.2f}",
                        ha="center", va="bottom", fontsize=7.5)

    # Legend
    patches = [mpatches.Patch(color=CANCER_COLORS[c], label=c.capitalize(), alpha=0.7)
               for c in datasets]
    fig.legend(handles=patches, loc="upper right", fontsize=10,
               bbox_to_anchor=(1.0, 1.0))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("\n[3/5] Distributions...")
plot_distributions(datasets, OUTPUT_DIR / "05_distributions.png")


# ============================================================================
# STEP 4 — NETWORK GRAPHS
# ============================================================================
NETWORK_THRESHOLDS = {"breast": 0.3, "lung": 0.5}

def plot_network(cancer: str, data: dict, out_path: Path,
                 threshold: float = 0.3):
    drug_names = data["drug_names"]
    mat = data["S_int"]
    n = len(drug_names)

    G = nx.Graph()
    G.add_nodes_from(drug_names)

    for i in range(n):
        for j in range(i + 1, n):
            w = mat[i, j]
            if w >= threshold:
                G.add_edge(drug_names[i], drug_names[j], weight=w)

    n_edges = G.number_of_edges()
    degrees = dict(G.degree())

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#FAF9F3")
    ax.set_facecolor("#FAF9F3")

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(n))

    # Node sizes proportional to degree (min size for isolated nodes)
    node_sizes = [max(300, degrees[node] * 400) for node in G.nodes()]

    # Node colors by degree
    degree_vals = [degrees[node] for node in G.nodes()]
    vmax_deg = max(degree_vals) if max(degree_vals) > 0 else 1

    # Draw edges with width/alpha proportional to weight
    edges = G.edges(data=True)
    edge_weights = [d["weight"] for _, _, d in edges]
    if edge_weights:
        edge_widths = [1 + 5 * w for w in edge_weights]
        edge_alphas = [0.3 + 0.6 * w for w in edge_weights]
        # Draw edges one by one for per-edge alpha
        for (u, v, d), lw, alpha in zip(G.edges(data=True), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                width=lw, alpha=alpha, edge_color="#555555"
            )

    # Draw nodes
    nc = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=degree_vals,
        cmap=plt.cm.YlOrRd,
        vmin=0, vmax=vmax_deg,
        alpha=0.9,
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8 if n > 15 else 9,
        font_weight="bold",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                norm=plt.Normalize(vmin=0, vmax=vmax_deg))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Node degree (# connections)", fontsize=9)

    ax.set_title(
        f"Drug Similarity Network — {cancer.capitalize()} Cancer\n"
        f"Integrated similarity (S_int ≥ {threshold}) | "
        f"{n} drugs, {n_edges} edges",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    # Legend for edge weight
    for sim_val, label in [(threshold, f"threshold ({threshold})"),
                            (0.6, "sim=0.6"), (0.9, "sim=0.9")]:
        if sim_val >= threshold:
            ax.plot([], [], color="#555555",
                    lw=1 + 5 * sim_val, alpha=0.3 + 0.6 * sim_val,
                    label=label)
    ax.legend(title="Edge weight", loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("\n[4/5] Network graphs...")
for cancer, data in datasets.items():
    thr = NETWORK_THRESHOLDS[cancer]
    plot_network(cancer, data,
                 OUTPUT_DIR / f"0{'6' if cancer=='breast' else '7'}_network_{cancer}.png",
                 threshold=thr)


# ============================================================================
# STEP 5 — SIMILARITY TYPE COMPARISON (scatter + correlation)
# ============================================================================
def plot_similarity_comparison(datasets: dict, out_path: Path):
    pairs = [
        ("S_chem", "S_atc",  "Chemical vs ATC"),
        ("S_chem", "S_tgt",  "Chemical vs Target"),
        ("S_atc",  "S_tgt",  "ATC vs Target"),
    ]

    n_cancers = len(datasets)
    n_pairs   = len(pairs)

    fig, axes = plt.subplots(n_cancers, n_pairs,
                             figsize=(15, 9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle(
        "Pairwise Similarity Type Comparison\n"
        "(each point = one drug pair, upper triangle)",
        fontsize=13, fontweight="bold"
    )

    for row, (cancer, data) in enumerate(datasets.items()):
        n = len(data["drug_names"])
        idx = np.triu_indices(n, k=1)

        for col, (key_x, key_y, title) in enumerate(pairs):
            ax = axes[row, col]
            x = data[key_x][idx]
            y = data[key_y][idx]

            color = CANCER_COLORS[cancer]

            # Scatter with density coloring
            ax.scatter(x, y, alpha=0.5, s=25, color=color, edgecolors="none")

            # Pearson r
            if x.std() > 0 and y.std() > 0:
                r, pval = pearsonr(x, y)
                r_str = f"r = {r:.3f}"
                p_str = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
            else:
                r_str = "r = N/A"
                p_str = ""

            # Regression line
            if x.std() > 0:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(0, 1, 100)
                ax.plot(xline, m * xline + b, color="black", lw=1.5, ls="--", alpha=0.7)

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(SIM_LABELS[key_x], fontsize=8)
            ax.set_ylabel(SIM_LABELS[key_y], fontsize=8)
            ax.set_title(
                f"{title}\n{cancer.capitalize()} ({n} drugs)",
                fontweight="bold", fontsize=9
            )
            ax.text(0.97, 0.05, f"{r_str}\n{p_str}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.grid(alpha=0.3)
            ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


print("\n[5/5] Similarity type comparison...")
plot_similarity_comparison(datasets, OUTPUT_DIR / "08_similarity_comparison.png")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("DONE — all figures saved to:", OUTPUT_DIR)
print("=" * 60)
output_files = sorted(OUTPUT_DIR.glob("*.png"))
for f in output_files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:45s}  {size_kb:6.1f} KB")
