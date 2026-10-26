"""Leakage-safe coincidence and transition matrices for ERWR enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from binary_erwr import rwrdc, score_pairs


@dataclass
class CoincidenceGraphStats:
    cell_line: str
    mode: str
    n_nodes: int
    n_positive_edges: int
    n_hyperedges: int
    n_coincidence_edges: int
    density: float
    sparsity: float
    isolated_nodes_before: int
    isolated_nodes_after: int
    connected_components: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix),
        where=row_sum > 0,
    )


def build_positive_adjacency(
    train_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_index: Mapping[str, int],
    cell_line: str | None,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Build positive TRAIN adjacency without using validation/test labels."""
    n = len(all_drugs)
    adjacency = np.zeros((n, n), dtype=float)
    edges: set[tuple[int, int]] = set()
    part = train_df if cell_line is None else train_df[
        train_df["cell_line_name"] == cell_line
    ]

    for row in part[part["result"] == 1].itertuples(index=False):
        i = drug_index[row.drug_a]
        j = drug_index[row.drug_b]
        if i == j:
            continue
        adjacency[i, j] = adjacency[j, i] = 1.0
        edges.add(tuple(sorted((i, j))))

    return adjacency, edges


def build_pair_incidence(
    train_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_index: Mapping[str, int],
    cell_line: str | None,
) -> np.ndarray:
    """Create H where each positive TRAIN pair is a two-drug hyperedge."""
    part = train_df if cell_line is None else train_df[
        train_df["cell_line_name"] == cell_line
    ]
    positives = part[part["result"] == 1]
    H = np.zeros((len(all_drugs), len(positives)), dtype=float)

    for edge_index, row in enumerate(positives.itertuples(index=False)):
        i = drug_index[row.drug_a]
        j = drug_index[row.drug_b]
        if i != j:
            H[i, edge_index] = 1.0
            H[j, edge_index] = 1.0

    return H


def coincidence_from_incidence(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return C=H De^-1 H.T and its row-normalized transition matrix."""
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be a 2-D incidence matrix")
    if H.shape[1] == 0:
        C = np.zeros((H.shape[0], H.shape[0]), dtype=float)
        return C, C.copy()

    edge_degree = H.sum(axis=0)
    valid = edge_degree > 0
    H_valid = H[:, valid]
    edge_degree = edge_degree[valid]
    C = (H_valid / edge_degree[None, :]) @ H_valid.T
    np.fill_diagonal(C, 0.0)
    T = _row_normalize(C)
    return C, T


def normalize_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize off-diagonal values to [0, 1]."""
    out = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(out, 0.0)
    if out.shape[0] < 2:
        return out
    values = out[~np.eye(out.shape[0], dtype=bool)]
    lo, hi = float(values.min()), float(values.max())
    if hi > lo:
        out = (out - lo) / (hi - lo)
    else:
        out.fill(0.0)
    np.fill_diagonal(out, 0.0)
    return np.clip(out, 0.0, 1.0)


def propagate_coincidence(
    coincidence: np.ndarray,
    transition: np.ndarray,
    propagation_lambda: float = 0.10,
) -> np.ndarray:
    """Blend direct coincidence with T C T.T structural propagation."""
    if not 0 <= propagation_lambda <= 1:
        raise ValueError("propagation_lambda must be in [0, 1]")
    propagated = transition @ coincidence @ transition.T
    result = (
        (1.0 - propagation_lambda) * coincidence
        + propagation_lambda * propagated
    )
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)
    return result


def align_feature_incidence(
    entity_names: Sequence[str],
    feature_names: Sequence[str],
    features: np.ndarray,
) -> np.ndarray:
    """Align an external entity-by-feature table as a nonnegative incidence H."""
    lookup = {str(name): i for i, name in enumerate(feature_names)}
    X = np.asarray(features, dtype=float)
    H = np.zeros((len(entity_names), X.shape[1]), dtype=float)
    for i, name in enumerate(entity_names):
        source = lookup.get(str(name))
        if source is not None:
            H[i] = np.maximum(X[source], 0.0)
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    nonempty = H.sum(axis=0) > 0
    return H[:, nonempty]


def build_coincidence_enriched_graph(
    train_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_index: Mapping[str, int],
    cell_line: str | None,
    beta: float = 0.05,
    propagation_lambda: float = 0.10,
    kg_incidence: np.ndarray | None = None,
    kg_alpha: float = 0.50,
) -> tuple[np.ndarray, CoincidenceGraphStats]:
    """Reweight positive edges using train coincidence and optional KG coincidence.

    No new drug edges are created. This prevents the dense-graph failure seen
    with direct embedding similarity enrichment.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if not 0 <= kg_alpha <= 1:
        raise ValueError("kg_alpha must be in [0, 1]")

    adjacency, positive_edges = build_positive_adjacency(
        train_df, all_drugs, drug_index, cell_line
    )
    isolated_before = int(np.sum(np.count_nonzero(adjacency, axis=1) == 0))

    H_pair = build_pair_incidence(
        train_df, all_drugs, drug_index, cell_line
    )
    C_pair, T_pair = coincidence_from_incidence(H_pair)
    C_final = propagate_coincidence(
        C_pair,
        T_pair,
        propagation_lambda,
    )

    if kg_incidence is not None:
        C_kg, _ = coincidence_from_incidence(kg_incidence)
        C_kg = normalize_off_diagonal(C_kg)
        C_final = (
            (1.0 - kg_alpha) * normalize_off_diagonal(C_final)
            + kg_alpha * C_kg
        )

    C_norm = normalize_off_diagonal(C_final)
    enriched = adjacency * (1.0 + beta * C_norm)
    graph = nx.from_numpy_array(enriched)
    final_edges = int(np.count_nonzero(np.triu(enriched, 1)))
    possible = len(all_drugs) * (len(all_drugs) - 1) // 2

    stats = CoincidenceGraphStats(
        cell_line=cell_line or "GLOBAL",
        mode="coincidence_reweighted",
        n_nodes=len(all_drugs),
        n_positive_edges=len(positive_edges),
        n_hyperedges=int(H_pair.shape[1]),
        n_coincidence_edges=final_edges,
        density=final_edges / possible if possible else 0.0,
        sparsity=1.0 - (final_edges / possible if possible else 0.0),
        isolated_nodes_before=isolated_before,
        isolated_nodes_after=sum(
            1 for _, degree in graph.degree() if degree == 0
        ),
        connected_components=(
            nx.number_connected_components(graph)
            if len(all_drugs) else 0
        ),
    )
    return enriched, stats


def score_with_coincidence_graph(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    all_drugs: Sequence[str],
    drug_index: Mapping[str, int],
    split: str,
    beta: float,
    propagation_lambda: float,
    kg_incidence: np.ndarray | None = None,
    kg_alpha: float = 0.50,
):
    scores = np.zeros(len(query_df), dtype=float)
    stats = []
    cache = {}

    for cell, group in query_df.groupby("cell_line_name", sort=False):
        graph_cell = None if split == "cell_line" else cell
        key = graph_cell or "GLOBAL"
        if key not in cache:
            A, graph_stats = build_coincidence_enriched_graph(
                train_df,
                all_drugs,
                drug_index,
                graph_cell,
                beta=beta,
                propagation_lambda=propagation_lambda,
                kg_incidence=kg_incidence,
                kg_alpha=kg_alpha,
            )
            cache[key] = (rwrdc(A), graph_stats)
            stats.append(graph_stats.as_dict())

        matrix, _ = cache[key]
        positions = query_df.index.get_indexer(group.index)
        scores[positions] = score_pairs(matrix, drug_index, group)

    return scores, stats
