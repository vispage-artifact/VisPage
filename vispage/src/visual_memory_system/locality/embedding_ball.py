"""Embedding-ball candidate construction."""

from __future__ import annotations

from typing import Any

from visual_memory_system.locality.base import LocalityEstimator
from visual_memory_system.retrieval.embedding import cosine
from visual_memory_system.schema import CandidatePage, MemoryUnit, QueryRecord, RegisteredPage

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in experiment envs.
    np = None


class EmbeddingBallEstimator(LocalityEstimator):
    name = "embedding_ball"

    def __init__(
        self,
        memory_embeddings: dict[str, list[float]],
        *,
        radius_percentile: float,
        max_units: int,
    ) -> None:
        if not 0 < radius_percentile <= 100:
            raise ValueError("radius_percentile must be in (0, 100]")
        if max_units <= 0:
            raise ValueError("max_units must be positive")
        self.memory_embeddings = memory_embeddings
        self.radius_percentile = radius_percentile
        self.max_units = max_units
        self._memory_ids: tuple[str, ...] | None = None
        self._memory_matrix: Any = None
        self._memory_index_by_id: dict[str, int] = {}

    def propose_pages(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        memory_units: list[MemoryUnit],
        registered_pages: dict[str, RegisteredPage],
    ) -> list[CandidatePage]:
        del registered_pages
        unit_by_id = {unit.unit_id: unit for unit in memory_units}
        if np is not None:
            return self._propose_pages_numpy(
                query=query,
                retrieved_unit_ids=retrieved_unit_ids,
                memory_units=memory_units,
                unit_by_id=unit_by_id,
            )
        candidates: list[CandidatePage] = []
        for anchor_id in retrieved_unit_ids:
            if anchor_id not in unit_by_id:
                raise KeyError(f"retrieved unit {anchor_id} is not in memory units")
            anchor_vec = self.memory_embeddings[anchor_id]
            scored: list[tuple[float, str]] = []
            for unit in memory_units:
                vec = self.memory_embeddings[unit.unit_id]
                distance = 1.0 - cosine(anchor_vec, vec)
                scored.append((distance, unit.unit_id))
            scored.sort(key=lambda item: (item[0], item[1]))
            keep_n = max(1, int(round(len(scored) * self.radius_percentile / 100.0)))
            selected = tuple(unit_id for _, unit_id in scored[: min(keep_n, self.max_units)])
            cov = _coverage(retrieved_unit_ids, selected)
            candidate_id = f"{self.name}:{query.query_id}:{anchor_id}:{self.radius_percentile:g}"
            candidates.append(
                CandidatePage(
                    candidate_id=candidate_id,
                    unit_ids=selected,
                    anchor_query_id=query.query_id,
                    anchor_unit_ids=(anchor_id,),
                    predicted_coverage=cov,
                    score=cov,
                    metadata={
                        "anchor_id": anchor_id,
                        "radius_percentile": self.radius_percentile,
                    },
                )
            )
        candidates.sort(key=lambda page: (-page.score, len(page.unit_ids), page.candidate_id))
        return candidates

    def _propose_pages_numpy(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        memory_units: list[MemoryUnit],
        unit_by_id: dict[str, MemoryUnit],
    ) -> list[CandidatePage]:
        memory_ids, memory_matrix = self._memory_index(memory_units)
        keep_n = max(1, int(round(len(memory_ids) * self.radius_percentile / 100.0)))
        keep_n = min(keep_n, self.max_units)
        candidates: list[CandidatePage] = []
        for anchor_id in retrieved_unit_ids:
            if anchor_id not in unit_by_id:
                raise KeyError(f"retrieved unit {anchor_id} is not in memory units")
            anchor_index = self._memory_index_by_id[anchor_id]
            scores = memory_matrix @ memory_matrix[anchor_index]
            distances = 1.0 - scores
            candidate_indices = np.argpartition(distances, keep_n - 1)[:keep_n]
            ordered_indices = sorted(
                candidate_indices.tolist(),
                key=lambda index: (float(distances[index]), memory_ids[index]),
            )
            selected = tuple(memory_ids[index] for index in ordered_indices)
            cov = _coverage(retrieved_unit_ids, selected)
            candidate_id = f"{self.name}:{query.query_id}:{anchor_id}:{self.radius_percentile:g}"
            candidates.append(
                CandidatePage(
                    candidate_id=candidate_id,
                    unit_ids=selected,
                    anchor_query_id=query.query_id,
                    anchor_unit_ids=(anchor_id,),
                    predicted_coverage=cov,
                    score=cov,
                    metadata={
                        "anchor_id": anchor_id,
                        "radius_percentile": self.radius_percentile,
                    },
                )
            )
        candidates.sort(key=lambda page: (-page.score, len(page.unit_ids), page.candidate_id))
        return candidates

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
        self._memory_index_by_id = {
            unit_id: index for index, unit_id in enumerate(memory_ids)
        }
        return self._memory_ids, self._memory_matrix


def _coverage(required: list[str], available: tuple[str, ...]) -> float:
    if not required:
        return 0.0
    return len(set(required) & set(available)) / len(set(required))


def _normalized_array(vector: list[float], name: str):
    if np is None:
        raise RuntimeError("numpy is required for vectorized embeddings")
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError(f"zero-norm embedding for {name}")
    return array / norm
