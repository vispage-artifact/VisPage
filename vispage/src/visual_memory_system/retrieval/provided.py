"""Retriever that uses QueryRecord.access_units explicitly."""

from __future__ import annotations

from visual_memory_system.retrieval.base import Retriever
from visual_memory_system.schema import MemoryUnit, QueryRecord


class ProvidedRetriever(Retriever):
    name = "provided_access_units"

    def retrieve(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        del memory_units
        if not query.access_units:
            raise ValueError(f"query {query.query_id} has no provided access_units")
        return list(query.access_units[:topk])

