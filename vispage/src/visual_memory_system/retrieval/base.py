"""Retrieval interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visual_memory_system.schema import MemoryUnit, QueryRecord


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        raise NotImplementedError

