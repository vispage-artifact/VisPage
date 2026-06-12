"""Embedding cosine retriever."""

from __future__ import annotations

import math
from typing import Any

from visual_memory_system.retrieval.base import Retriever
from visual_memory_system.schema import MemoryUnit, QueryRecord

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in experiment envs.
    np = None


class EmbeddingRetriever(Retriever):
    name = "embedding_retrieve"

    def __init__(
        self,
        memory_embeddings: dict[str, list[float]],
        query_embeddings: dict[str, list[float]],
        *,
        require_full_topk: bool = True,
    ) -> None:
        self.memory_embeddings = memory_embeddings
        self.query_embeddings = query_embeddings
        self.require_full_topk = require_full_topk
        self._memory_ids: tuple[str, ...] | None = None
        self._memory_matrix: Any = None
        self._query_vectors: dict[str, Any] | None = None
        if np is not None:
            self._query_vectors = {
                query_id: _normalized_array(vector, f"query {query_id}")
                for query_id, vector in query_embeddings.items()
            }

    def retrieve(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        if query.query_id not in self.query_embeddings:
            raise KeyError(f"missing embedding for query {query.query_id}")
        if np is not None:
            return self._retrieve_numpy(query, memory_units, topk)
        query_vec = self.query_embeddings[query.query_id]
        scored: list[tuple[float, str]] = []
        for unit in memory_units:
            if unit.unit_id not in self.memory_embeddings:
                raise KeyError(f"missing embedding for memory unit {unit.unit_id}")
            score = cosine(query_vec, self.memory_embeddings[unit.unit_id])
            scored.append((score, unit.unit_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [unit_id for _, unit_id in scored[:topk]]
        if self.require_full_topk and len(selected) < topk:
            raise ValueError(f"retrieved {len(selected)} units, expected topk={topk}")
        return selected

    def _retrieve_numpy(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        if self._query_vectors is None:
            raise RuntimeError("query vectors were not initialized")
        if query.query_id not in self._query_vectors:
            raise KeyError(f"missing embedding for query {query.query_id}")
        memory_ids, memory_matrix = self._memory_index(memory_units)
        if topk > len(memory_ids):
            if self.require_full_topk:
                raise ValueError(f"retrieved {len(memory_ids)} units, expected topk={topk}")
            topk = len(memory_ids)
        scores = memory_matrix @ self._query_vectors[query.query_id]
        candidate_indices = np.argpartition(-scores, topk - 1)[:topk]
        ordered_indices = sorted(
            candidate_indices.tolist(),
            key=lambda index: (-float(scores[index]), memory_ids[index]),
        )
        return [memory_ids[index] for index in ordered_indices]

    def _memory_index(self, memory_units: list[MemoryUnit]):
        memory_ids = tuple(unit.unit_id for unit in memory_units)
        if self._memory_ids == memory_ids and self._memory_matrix is not None:
            return self._memory_ids, self._memory_matrix
        vectors = []
        for unit_id in memory_ids:
            if unit_id not in self.memory_embeddings:
                raise KeyError(f"missing embedding for memory unit {unit_id}")
            vectors.append(_normalized_array(self.memory_embeddings[unit_id], f"memory unit {unit_id}"))
        self._memory_ids = memory_ids
        self._memory_matrix = np.stack(vectors, axis=0)
        return self._memory_ids, self._memory_matrix


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimensions differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(y * y for y in b))
    if an == 0 or bn == 0:
        raise ValueError("zero-norm embedding")
    return dot / (an * bn)


def _normalized_array(vector: list[float], name: str):
    if np is None:
        raise RuntimeError("numpy is required for vectorized embeddings")
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError(f"zero-norm embedding for {name}")
    return array / norm
