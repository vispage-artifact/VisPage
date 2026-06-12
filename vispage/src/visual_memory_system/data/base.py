"""Dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visual_memory_system.schema import MemoryUnit, QueryRecord


class DataAdapter(ABC):
    @abstractmethod
    def load_memory_units(self) -> list[MemoryUnit]:
        raise NotImplementedError

    @abstractmethod
    def load_queries(self) -> list[QueryRecord]:
        raise NotImplementedError

