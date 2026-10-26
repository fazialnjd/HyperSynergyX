"""Weighted ERWR utilities using continuous synergy values and sparse repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from binary_erwr import rwrdc, score_pairs


@dataclass
class WeightedGraphStats:
    cell_line: str
    score_column: str
    n_nodes: int
    n_weighted_edges: int
    total_edge_weight: float
    mean_edge_weight: float
    density: float
    isolated_nodes_before_similarity: int
    n_similarity_edges: int
    isolated_nodes: int
    connected_components: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        return np.ones_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _rank_scale(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    ranks = series.rank(method="average", pct=True).to_numpy(dtype=float)
    return np.clip(ranks, 0.0, 1.0)


def _top_k_similarity_edges(
    similarity: np.ndarray,
    nodes: Sequence[int],
    top_k: int,
) -> set[tuple[int, int]]:
    if similarity is None or top_k < 1:
        return set()
    similarity = np.asarray(similarity, dtype=float)
    n = similarity.shape[0]
    edges: set[tuple[int, int]] = set()
    k = min(int(top_k), max(0, n - 1))
    for i in nodes:
        candidates = np.delete(np.arange(n), int(i))
        order = np.argsort(similarity[int(i), candidates])[::-1][:k]
        for position in order:
            j = int(candidates[position])
            value = float(similarity[int(i), j])
            if value > 0:
                edges.add(tuple(sorted((int(i), j))))
    return edges


def build_weighted_graph(
    train_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_index: Mapping[str, int],
    score_column: str,
    cell_line: str | None = None,
    weight_mode: str = "rank",
    positive_label_column: str = "result",
    similarity: np.ndarray | None = None,
    similarity_top_k: int = 0,
    similarity_beta: float = 0.05,
) -> tuple[np.ndarray, WeightedGraphStats]:
    """Build weighted positive graph and optionally repair isolated nodes only."""
    if score_column not in train_df.columns:
        raise ValueError(f"Missing score column: {score_column}")
    if positive_label_column not in train_df.columns:
        raise ValueError(f"Missing label column: {positive_label_column}")
    if weight_mode not in {"rank", "minmax", "raw"}:
        raise ValueError("weight_mode must be rank, minmax, or raw")
    if similarity_beta < 0:
        raise ValueError("similarity_beta must be non-negative")
    if similarity is not None and np.asarray(similarity).shape != (len(all_drugs), len(all_drugs)):
        raise ValueError("similarity must have shape (n_drugs, n_drugs)")

    part = train_df if cell_line is None else train_df[
        train_df["cell_line_name"] == cell_line
    ]
    part = part[part[positive_label_column].astype(int) == 1].copy()
    part[score_column] = pd.to_numeric(part[score_column], errors="coerce")
    part = part.dropna(subset=[score_column])
    part = part.groupby(["drug_a", "drug_b"], as_index=False)[score_column].max()

    values = part[score_column].to_numpy(dtype=float)
    if weight_mode == "rank":
        weights = _rank_scale(values)
    elif weight_mode == "minmax":
        weights = _minmax(values)
    else:
        weights = np.maximum(values, 0.0)
        maximum = float(weights.max()) if len(weights) else 0.0
        if maximum > 0:
            weights = weights / maximum

    n = len(all_drugs)
    adjacency = np.zeros((n, n), dtype=float)
    for row, weight in zip(part.itertuples(index=False), weights):
        i = drug_index[row.drug_a]
        j = drug_index[row.drug_b]
        if i == j or weight <= 0:
            continue
        adjacency[i, j] = max(adjacency[i, j], float(weight))
        adjacency[j, i] = adjacency[i, j]

    isolated_before = np.flatnonzero(np.count_nonzero(adjacency, axis=1) == 0).tolist()
    similarity_edges = _top_k_similarity_edges(
        similarity,
        isolated_before,
        similarity_top_k,
    )
    for i, j in similarity_edges:
        value = float(similarity[i, j])
        adjacency[i, j] += similarity_beta * value
        adjacency[j, i] = adjacency[i, j]

    graph = nx.from_numpy_array(adjacency)
    possible = n * (n - 1) // 2
    edge_values = adjacency[np.triu_indices(n, k=1)]
    edge_values = edge_values[edge_values > 0]
    stats = WeightedGraphStats(
        cell_line=cell_line or "GLOBAL",
        score_column=score_column,
        n_nodes=n,
        n_weighted_edges=int(len(edge_values)),
        total_edge_weight=float(edge_values.sum()) if len(edge_values) else 0.0,
        mean_edge_weight=float(edge_values.mean()) if len(edge_values) else 0.0,
        density=(len(edge_values) / possible if possible else 0.0),
        isolated_nodes_before_similarity=len(isolated_before),
        n_similarity_edges=len(similarity_edges),
        isolated_nodes=sum(1 for _, degree in graph.degree() if degree == 0),
        connected_components=(nx.number_connected_components(graph) if n else 0),
    )
    return adjacency, stats


def score_weighted_graph(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_index: Mapping[str, int],
    split: str,
    score_column: str,
    weight_mode: str,
    positive_label_column: str = "result",
    similarity: np.ndarray | None = None,
    similarity_top_k: int = 0,
    similarity_beta: float = 0.05,
):
    scores = np.zeros(len(query_df), dtype=float)
    stats = []
    cache = {}

    for cell, group in query_df.groupby("cell_line_name", sort=False):
        graph_cell = None if split == "cell_line" else cell
        key = graph_cell or "GLOBAL"
        if key not in cache:
            adjacency, graph_stats = build_weighted_graph(
                train_df,
                all_drugs,
                drug_index,
                score_column,
                graph_cell,
                weight_mode,
                positive_label_column,
                similarity,
                similarity_top_k,
                similarity_beta,
            )
            cache[key] = (rwrdc(adjacency), graph_stats)
            stats.append(graph_stats.as_dict())
        matrix, _ = cache[key]
        positions = query_df.index.get_indexer(group.index)
        scores[positions] = score_pairs(matrix, drug_index, group)

    return scores, stats
