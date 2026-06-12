"""Runtime client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visual_memory_system.schema import CacheSelection, ForegroundPlan, RuntimeResponse, VisualPage


class RuntimeClient(ABC):
    @abstractmethod
    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        raise NotImplementedError

    @abstractmethod
    def send_foreground(self, plan: ForegroundPlan) -> RuntimeResponse:
        raise NotImplementedError

    @abstractmethod
    def send_background(self, page: VisualPage) -> RuntimeResponse:
        raise NotImplementedError

