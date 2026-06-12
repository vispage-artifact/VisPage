"""Baseline foreground-only execution."""

from __future__ import annotations

from visual_memory_system.locality.base import LocalityEstimator
from visual_memory_system.schema import CandidatePage, MemoryUnit, QueryRecord, RegisteredPage


class BaselineEstimator(LocalityEstimator):
    """Marker estimator for foreground-only baseline runs."""

    name = "baseline"

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

