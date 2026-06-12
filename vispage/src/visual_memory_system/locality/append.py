"""Append-style foreground page construction."""

from __future__ import annotations

from visual_memory_system.locality.base import LocalityEstimator
from visual_memory_system.schema import CandidatePage, MemoryUnit, QueryRecord, RegisteredPage


class AppendEstimator(LocalityEstimator):
    """Marker estimator for foreground-only append construction.

    Append mode does not propose background prewarm candidates. The runner
    registers pages that were actually submitted in foreground.
    """

    name = "append"

    def propose_pages(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        memory_units: list[MemoryUnit],
        registered_pages: dict[str, RegisteredPage],
    ) -> list[CandidatePage]:
        del query, retrieved_unit_ids, memory_units, registered_pages
        return []
