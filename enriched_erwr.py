"""Top-k similarity-enriched weighted ERWR graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from binary_erwr import rwrdc, score_pairs


@dataclass
class EnrichedGraphStats:
    cell_line: str
    mode: str
    n_nodes: int
    n_positive_edges: int
    n_similarity_edges: int
    possible_edges: int
    density: float
    sparsity: float
    isolated_nodes: int
    connected_components: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _top_k_similarity_edges(similarity: np.ndarray, top_k: int) -> set[tuple[int, int]]:
    """Return the symmetrized union of each node's top-k non-self neighbors."""
    if top_k < 1:
        return set()

    n = similarity.shape[0]
    k = min(int(top_k), max(0, n - 1))
    edges: set[tuple[int, int]] = set()

    for i in range(n):
        candidates = np.delete(np.arange(n), i)
        order = np.argsort(similarity[i, candidates])[::-1][:k]
        for position in order:
            j = int(candidates[position])
            if similarity[i, j] > 0:
                edges.add(tuple(sorted((i, j))))

    return edges


def build_enriched_graph(
    train_df: pd.DataFrame,
    all_drugs: Sequence[str],
    similarity: np.ndarray,
    drug_index: Mapping[str, int],
    cell_line: str | None,
    mode: str = "enriched",
    beta: float = 0.10,
    similarity_top_k: int = 5,
):
    """Build a positive-pair graph plus a sparse top-k similarity prior."""
    if mode not in {"baseline", "enriched", "similarity_only"}:
        raise ValueError("mode must be baseline, enriched, or similarity_only")
    if similarity_top_k < 1:
        raise ValueError("similarity_top_k must be at least 1")

    n = len(all_drugs)
    A = np.zeros((n, n), dtype=float)
    positive_edges: set[tuple[int, int]] = set()

    if mode != "similarity_only":
        part = train_df if cell_line is None else train_df[
            train_df["cell_line_name"] == cell_line
        ]
        for row in part[part["result"] == 1].itertuples(index=False):
            i, j = drug_index[row.drug_a], drug_index[row.drug_b]
            if i != j:
                A[i, j] = A[j, i] = 1.0
                positive_edges.add(tuple(sorted((i, j))))

    similarity_edges: set[tuple[int, int]] = set()
    if mode in {"enriched", "similarity_only"}:
        for i, j in _top_k_similarity_edges(similarity, similarity_top_k):
            value = float(similarity[i, j])
            if value <= 0:
                continue
            A[i, j] += beta * value
            A[j, i] += beta * value
            if (i, j) not in positive_edges:
                similarity_edges.add((i, j))

    graph = nx.from_numpy_array(A)
    possible = n * (n - 1) // 2
    weighted_edges = int(np.count_nonzero(np.triu(A, 1)))
    density = weighted_edges / possible if possible else 0.0
    stats = EnrichedGraphStats(
        cell_line=cell_line or "GLOBAL",
        mode=mode,
        n_nodes=n,
        n_positive_edges=len(positive_edges),
        n_similarity_edges=len(similarity_edges),
        possible_edges=possible,
        density=float(density),
        sparsity=float(1.0 - density),
        isolated_nodes=sum(1 for _, degree in graph.degree() if degree == 0),
        connected_components=nx.number_connected_components(graph) if n else 0,
    )
    return A, stats


def score_with_graph(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    all_drugs: Sequence[str],
    similarity: np.ndarray,
    drug_index: Mapping[str, int],
    split: str,
    mode: str,
    beta: float,
    similarity_top_k: int,
):
    """Score query rows using cell-specific or global weighted graphs."""
    scores = np.zeros(len(query_df), dtype=float)
    stats = []
    cache = {}

    for cell, group in query_df.groupby("cell_line_name", sort=False):
        graph_cell = None if split == "cell_line" else cell
        key = graph_cell or "GLOBAL"
        if key not in cache:
            A, graph_stats = build_enriched_graph(
                train_df, all_drugs, similarity, drug_index, graph_cell,
                mode=mode, beta=beta, similarity_top_k=similarity_top_k,
            )
            cache[key] = (rwrdc(A), graph_stats)
            stats.append(graph_stats.as_dict())

        matrix, _ = cache[key]
        positions = query_df.index.get_indexer(group.index)
        scores[positions] = score_pairs(matrix, drug_index, group)

    return scores, stats


def score_cell_line_transfer(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_similarity: np.ndarray,
    drug_index: Mapping[str, int],
    cell_names: list[str],
    cell_similarity: np.ndarray,
    cell_index: Mapping[str, int],
    mode: str,
    beta: float,
    similarity_top_k: int,
):
    """Score unseen cell lines by similarity-weighted train-cell graphs."""
    scores = np.zeros(len(query_df), dtype=float)
    stats = []
    graph_cache = {}

    for cell, group in query_df.groupby("cell_line_name", sort=False):
        if cell not in cell_index:
            raise ValueError(f"Cell line {cell!r} is missing from cell features")

        target_idx = cell_index[cell]
        weights = []
        matrices = []
        for train_cell in sorted(train_df["cell_line_name"].unique()):
            if train_cell not in cell_index:
                continue
            source_idx = cell_index[train_cell]
            weight = float(cell_similarity[target_idx, source_idx])
            if weight <= 0:
                continue
            if train_cell not in graph_cache:
                A, graph_stats = build_enriched_graph(
                    train_df, all_drugs, drug_similarity, drug_index, train_cell,
                    mode=mode, beta=beta, similarity_top_k=similarity_top_k,
                )
                graph_cache[train_cell] = (rwrdc(A), graph_stats)
                stats.append(graph_stats.as_dict())
            matrices.append(graph_cache[train_cell][0])
            weights.append(weight)

        if not matrices:
            A, graph_stats = build_enriched_graph(
                train_df, all_drugs, drug_similarity, drug_index, None,
                mode=mode, beta=beta, similarity_top_k=similarity_top_k,
            )
            matrices = [rwrdc(A)]
            weights = [1.0]
            stats.append(graph_stats.as_dict())

        combined = np.average(np.stack(matrices), axis=0, weights=np.asarray(weights))
        positions = query_df.index.get_indexer(group.index)
        scores[positions] = score_pairs(combined, drug_index, group)

    return scores, stats
