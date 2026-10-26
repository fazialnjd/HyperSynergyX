"""Feature loading and cosine-similarity utilities for enriched ERWR."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_feature_table(path: str | Path, id_col: str | None = None) -> Tuple[list[str], np.ndarray]:
    """Load an ID + numeric feature table from CSV or Excel."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"Feature file is empty: {path}")

    if id_col is None:
        candidates = [
            col for col in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[col])
        ]
        id_col = candidates[0] if candidates else frame.columns[0]

    if id_col not in frame.columns:
        raise ValueError(f"Identifier column {id_col!r} not found in {path}")

    ids = frame[id_col].astype(str).str.strip()
    feature_frame = frame.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.dropna(axis=1, how="all").fillna(0.0)

    if feature_frame.shape[1] == 0:
        raise ValueError(f"No numeric feature columns found in {path}")

    feature_frame.index = ids
    feature_frame = feature_frame.groupby(level=0, sort=False).mean()
    return feature_frame.index.tolist(), feature_frame.to_numpy(dtype=float)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Return non-negative cosine similarity in [0, 1].

    Negative cosine values are clipped to zero instead of being shifted by
    (cosine + 1) / 2. Shifting would turn unrelated vectors with cosine near
    zero into artificial similarity near 0.5 and make the graph dense.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be a 2-D matrix")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    cosine = normalized @ normalized.T
    similarity = np.clip(cosine, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def align_similarity(
    entity_names: list[str],
    feature_names: list[str],
    similarity: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Align a feature similarity matrix to the requested entity order."""
    lookup = {name: i for i, name in enumerate(feature_names)}
    n = len(entity_names)
    aligned = np.zeros((n, n), dtype=float)
    present = [lookup.get(name) for name in entity_names]

    for i, source_i in enumerate(present):
        if source_i is None:
            continue
        for j, source_j in enumerate(present):
            if source_j is not None:
                aligned[i, j] = similarity[source_i, source_j]

    for i, source_i in enumerate(present):
        if source_i is not None:
            aligned[i, i] = 1.0

    return aligned, sum(source is not None for source in present)
