"""Locality estimator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visual_memory_system.schema import CandidatePage, MemoryUnit, QueryRecord, RegisteredPage


class LocalityEstimator(ABC):
    name: str

    @abstractmethod
    def propose_pages(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        memory_units: list[MemoryUnit],
        registered_pages: dict[str, RegisteredPage],
    ) -> list[CandidatePage]:
        """Return temporary candidates.

        This method should be called only after foreground baseline-cold execution. The
        estimator does not register pages and does not submit prewarm requests.
        """

        raise NotImplementedError
