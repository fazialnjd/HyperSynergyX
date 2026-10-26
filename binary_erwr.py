"""Leakage-safe binary ERWR for cell-line-specific drug pairs.

The original ERWR implementation returns a pairwise random-walk score matrix.
This module keeps that behavior and adds data/split/graph utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"drug_row", "drug_col", "cell_line_name", "result"}


@dataclass
class GraphStats:
    cell_line: str
    n_nodes: int
    n_positive_edges: int
    possible_edges: int
    density: float
    sparsity: float
    isolated_nodes: int
    connected_components: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def load_interactions(path: str) -> pd.DataFrame:
    """Load extract.xlsx or CSV and validate required binary-label columns."""
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    for col in ("drug_row", "drug_col", "cell_line_name"):
        out[col] = out[col].astype(str).str.strip()
    out["result"] = pd.to_numeric(out["result"], errors="raise").astype(int)
    if not out["result"].isin([0, 1]).all():
        raise ValueError("result must contain only binary values 0 and 1")

    # Sort text drug identifiers so A+B and B+A represent one unordered pair.
    pairs = out[["drug_row", "drug_col"]].apply(
        lambda row: sorted([row["drug_row"], row["drug_col"]]),
        axis=1,
        result_type="expand",
    )
    out["drug_a"] = pairs[0].astype(str)
    out["drug_b"] = pairs[1].astype(str)
    if (out["drug_a"] == out["drug_b"]).any():
        raise ValueError("Self-pairs are not supported")
    out = out.drop_duplicates(["drug_a", "drug_b", "cell_line_name"], keep="first")
    return out.reset_index(drop=True)


def rwrdc(adj_matrix: np.ndarray, c: float = 0.05) -> np.ndarray:
    """Return the original ERWR pairwise score matrix P + P.T."""
    A = np.asarray(adj_matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adj_matrix must be square")
    if not 0 <= c < 1:
        raise ValueError("c must satisfy 0 <= c < 1")
    A = np.maximum((A + A.T) / 2.0, 0.0)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    degree = A.sum(axis=1)
    W = np.divide(A, degree[:, None], out=np.zeros_like(A), where=degree[:, None] > 0)
    M = np.linalg.pinv(np.eye(n) - c * W.T)
    P = np.zeros((n, n), dtype=float)
    for k in range(n):
        r0 = np.zeros(n, dtype=float)
        r0[k] = 1.0
        P[:, k] = (1.0 - c) * (M @ r0)
    S = P + P.T
    np.fill_diagonal(S, 0.0)
    return normalize_scores(S)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize off-diagonal scores for binary ranking."""
    out = np.asarray(scores, dtype=float).copy()
    np.fill_diagonal(out, 0.0)
    if out.size == 0:
        return out
    off = ~np.eye(out.shape[0], dtype=bool)
    lo, hi = np.nanmin(out[off]), np.nanmax(out[off])
    if hi > lo:
        out = (out - lo) / (hi - lo)
    else:
        out.fill(0.0)
    np.fill_diagonal(out, 0.0)
    return out


def build_graph(train_df: pd.DataFrame, all_drugs: Sequence[str], cell_line: str | None = None):
    """Build a binary undirected graph from positive TRAIN pairs only."""
    drugs = list(all_drugs)
    idx = {drug: i for i, drug in enumerate(drugs)}
    A = np.zeros((len(drugs), len(drugs)), dtype=float)
    part = train_df if cell_line is None else train_df[train_df["cell_line_name"] == cell_line]
    positives = part[part["result"] == 1]
    for row in positives.itertuples(index=False):
        i, j = idx[row.drug_a], idx[row.drug_b]
        if i != j:
            A[i, j] = A[j, i] = 1.0
    n = len(drugs)
    possible = n * (n - 1) // 2
    edges = int(np.count_nonzero(np.triu(A, 1)))
    G = nx.from_numpy_array(A)
    density = edges / possible if possible else 0.0
    stats = GraphStats(
        cell_line=cell_line or "GLOBAL",
        n_nodes=n,
        n_positive_edges=edges,
        possible_edges=possible,
        density=float(density),
        sparsity=float(1.0 - density),
        isolated_nodes=sum(1 for _, degree in G.degree() if degree == 0),
        connected_components=nx.number_connected_components(G) if n else 0,
    )
    return A, idx, stats


def make_splits(df: pd.DataFrame, split: str, folds: int, seed: int):
    """Create sample, drug_pair, or cell_line group folds."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if split == "sample":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(df, df["result"].values))
    if split == "drug_pair":
        groups = (df["drug_a"] + "||" + df["drug_b"]).values
    elif split == "cell_line":
        groups = df["cell_line_name"].values
    else:
        raise ValueError("split must be one of: sample, drug_pair, cell_line")
    return list(GroupKFold(n_splits=folds).split(df, df["result"].values, groups))


def score_pairs(score_matrix: np.ndarray, index: Mapping[str, int], rows: pd.DataFrame) -> np.ndarray:
    scores = []
    for row in rows.itertuples(index=False):
        i, j = index.get(row.drug_a), index.get(row.drug_b)
        scores.append(float(score_matrix[i, j]) if i is not None and j is not None else 0.0)
    return np.asarray(scores, dtype=float)
