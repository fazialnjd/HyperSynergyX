"""
build_hypergraph.py
====================
Constructs a drug-combination hypergraph HG(V, E) from 3-drug TSV files.

  V = individual drugs (vertices)
  E = 3-drug combinations (hyperedges)

Outputs saved to HyperSynergyX/hypergraph_output/:
  breast_incidence_matrix.tsv
  lung_incidence_matrix.tsv
  breast_hypergraph.png
  lung_hypergraph.png
  hypergraph_combined.png

Usage:
  cd /home/faezeh/uni/progect/HyperSynergyX
  python build_hypergraph.py
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import networkx as nx
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "filtered"
OUTDIR = Path(__file__).parent / "hypergraph_output"
OUTDIR.mkdir(exist_ok=True)


# ── 1. Load ────────────────────────────────────────────────────────────────────
def load(fname):
    path = BASE / fname
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for col in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({".": pd.NA, "nan": pd.NA, "": pd.NA})
    df = df.dropna(subset=["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"])
    print(
        f"  Loaded {fname}: {len(df)} rows, "
        f"{df[['Drug_Name_1','Drug_Name_2','Drug_Name_3']].stack().nunique()} unique drugs"
    )
    return df


print("Loading files...")
breast_df = load("breast_3drug.tsv")
lung_df = load("lung_3drug.tsv")


# ── 2. Incidence matrix ────────────────────────────────────────────────────────
def build_incidence_matrix(df):
    """
    Binary DataFrame H of shape (|V| x |E|).
    H[drug, combo] = 1 if drug participates in that combination.
    """
    drugs = sorted(
        set(
            df["Drug_Name_1"].tolist()
            + df["Drug_Name_2"].tolist()
            + df["Drug_Name_3"].tolist()
        )
    )
    combos = df["DrugCom_ID"].tolist()
    drug_idx = {d: i for i, d in enumerate(drugs)}
    combo_idx = {c: j for j, c in enumerate(combos)}

    H = np.zeros((len(drugs), len(combos)), dtype=int)
    for _, row in df.iterrows():
        j = combo_idx[row["DrugCom_ID"]]
        for col in ["Drug_Name_1", "Drug_Name_2", "Drug_Name_3"]:
            d = str(row[col]).strip()
            if d not in (".", "nan", "", "<NA>"):
                H[drug_idx[d], j] = 1

    return pd.DataFrame(H, index=drugs, columns=combos)


print("\nBuilding incidence matrices...")
H_breast = build_incidence_matrix(breast_df)
H_lung = build_incidence_matrix(lung_df)

H_breast.to_csv(OUTDIR / "breast_incidence_matrix.tsv", sep="\t")
H_lung.to_csv(OUTDIR / "lung_incidence_matrix.tsv", sep="\t")
print(f"  Breast H: {H_breast.shape}  -> breast_incidence_matrix.tsv")
print(f"  Lung   H: {H_lung.shape}    -> lung_incidence_matrix.tsv")


# ── 3. Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(H_df, label):
    H = H_df.values
    drugs = H_df.index.tolist()
    combos = H_df.columns.tolist()
    degree = H.sum(axis=1)
    cooccur = H @ H.T
    np.fill_diagonal(cooccur, 0)

    deg_df = pd.DataFrame(
        {
            "drug": drugs,
            "degree": degree,
            "pct_combos": (degree / len(combos) * 100).round(1),
        }
    ).sort_values("degree", ascending=False)

    print(f"\n=== {label} — Node Degrees ===")
    print(deg_df.to_string(index=False))
    print(
        f"  Hyperedge size range: "
        f"{H.sum(axis=0).min()} – {H.sum(axis=0).max()} (should all be 3)"
    )
    return deg_df, cooccur


deg_breast, cooc_breast = compute_metrics(H_breast, "BREAST")
deg_lung, cooc_lung = compute_metrics(H_lung, "LUNG")


# ── 4. Visualisation ───────────────────────────────────────────────────────────
def spring_layout(drugs, cooccur_matrix):
    G = nx.Graph()
    G.add_nodes_from(drugs)
    n = len(drugs)
    for i in range(n):
        for j in range(i + 1, n):
            w = cooccur_matrix[i, j]
            if w > 0:
                G.add_edge(drugs[i], drugs[j], weight=float(w))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.8 / np.sqrt(max(n, 1)))
    return G, pos


def draw_panel(
    H_df, degree_df, cooccur, cancer_label, ax, global_vmin, global_vmax, max_edges=60
):
    drugs = H_df.index.tolist()
    combos = H_df.columns.tolist()
    H = H_df.values
    G, pos = spring_layout(drugs, cooccur)

    # co-occurrence backbone edges
    for u, v in G.edges():
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color="#cccccc",
            lw=0.6,
            zorder=1,
        )

    # hyperedge triangles (sample if > max_edges for visual clarity)
    rng = np.random.default_rng(0)
    edge_idx = list(range(len(combos)))
    if len(edge_idx) > max_edges:
        edge_idx = rng.choice(edge_idx, size=max_edges, replace=False).tolist()

    patches, colors = [], []
    for k, j in enumerate(edge_idx):
        members = [drugs[i] for i in range(len(drugs)) if H[i, j] == 1]
        if len(members) < 3:
            continue
        pts = np.array([pos[d] for d in members])
        patches.append(Polygon(pts, closed=True))
        colors.append(k)

    pc = PatchCollection(patches, alpha=0.13, cmap="tab20", edgecolors="none", zorder=2)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)

    # nodes — unified color scale across both panels
    deg_map = dict(zip(degree_df["drug"], degree_df["degree"]))
    node_x = [pos[d][0] for d in drugs]
    node_y = [pos[d][1] for d in drugs]
    node_s = [
        150
        + 550 * (deg_map.get(d, 1) - global_vmin) / max(global_vmax - global_vmin, 1)
        for d in drugs
    ]
    node_c = [deg_map.get(d, 1) for d in drugs]

    sc = ax.scatter(
        node_x,
        node_y,
        s=node_s,
        c=node_c,
        cmap="YlOrRd",
        zorder=4,
        edgecolors="#333333",
        linewidths=0.8,
        vmin=global_vmin,
        vmax=global_vmax,
    )

    # labels with boundary padding
    xs = [pos[d][0] for d in drugs]
    ys = [pos[d][1] for d in drugs]
    xpad = (max(xs) - min(xs)) * 0.10 + 0.25
    ypad = (max(ys) - min(ys)) * 0.10 + 0.25
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    for d in drugs:
        x, y = pos[d]
        ax.text(
            x,
            y + 0.09,
            d,
            ha="center",
            va="bottom",
            fontsize=6.8,
            zorder=5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.50, pad=0.02)
    cbar.set_label("Node degree  d(v)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        f"{cancer_label} cancer\n"
        f"|V| = {len(drugs)} drugs   |E| = {len(combos)} hyperedges",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    ax.axis("off")


# unified degree scale across both panels
global_vmin = min(deg_breast["degree"].min(), deg_lung["degree"].min())
global_vmax = max(deg_breast["degree"].max(), deg_lung["degree"].max())

# ── individual figures ─────────────────────────────────────────────────────────
for df_h, deg, cooc, label, fname in [
    (H_breast, deg_breast, cooc_breast, "Breast", "breast_hypergraph.png"),
    (H_lung, deg_lung, cooc_lung, "Lung", "lung_hypergraph.png"),
]:
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#fafafa")
    draw_panel(df_h, deg, cooc, label, ax, global_vmin, global_vmax)
    tri = mpatches.Patch(
        facecolor="#aaaaff",
        alpha=0.4,
        label="Hyperedge = 3-drug combination (shaded triangle)",
    )
    fig.legend(
        handles=[tri],
        loc="lower center",
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f"Drug Combination Hypergraph  HG(V, E) — {label} cancer\n"
        "Node size & color ∝ degree",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(OUTDIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

# ── combined figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor("#fafafa")
draw_panel(
    H_breast, deg_breast, cooc_breast, "Breast", axes[0], global_vmin, global_vmax
)
draw_panel(H_lung, deg_lung, cooc_lung, "Lung", axes[1], global_vmin, global_vmax)
tri = mpatches.Patch(
    facecolor="#aaaaff",
    alpha=0.4,
    label="Hyperedge = 3-drug combination (shaded triangle)",
)
fig.legend(
    handles=[tri],
    loc="lower center",
    fontsize=9.5,
    frameon=True,
    bbox_to_anchor=(0.5, 0.005),
)
fig.suptitle(
    "Drug Combination Hypergraph   HG(V, E)\n"
    "Vertices = individual drugs  ·  Hyperedges = known 3-drug combinations  ·  "
    "Node size & color ∝ degree  ·  Colorbar scale unified across panels",
    fontsize=11,
    fontweight="bold",
    y=0.99,
)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(OUTDIR / "hypergraph_combined.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nAll outputs saved to: {OUTDIR}/")
print("  breast_incidence_matrix.tsv")
print("  lung_incidence_matrix.tsv")
print("  breast_hypergraph.png")
print("  lung_hypergraph.png")
print("  hypergraph_combined.png")
