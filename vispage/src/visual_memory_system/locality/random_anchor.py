"""Random anchor locality ablation."""

from __future__ import annotations

import random

from visual_memory_system.locality.base import LocalityEstimator
from visual_memory_system.schema import CandidatePage, MemoryUnit, QueryRecord, RegisteredPage


class RandomAnchorEstimator(LocalityEstimator):
    name = "random_anchor"

    def __init__(self, *, page_units: int, seed: int = 0) -> None:
        if page_units <= 0:
            raise ValueError("page_units must be positive")
        self.page_units = page_units
        self.rng = random.Random(seed)

    def propose_pages(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        memory_units: list[MemoryUnit],
        registered_pages: dict[str, RegisteredPage],
    ) -> list[CandidatePage]:
        del registered_pages
        unit_ids = [unit.unit_id for unit in memory_units]
        retrieved = tuple(dict.fromkeys(retrieved_unit_ids))
        if len(retrieved) > self.page_units:
            selected = retrieved[: self.page_units]
        else:
            remaining = [unit_id for unit_id in unit_ids if unit_id not in set(retrieved)]
            fill_count = min(self.page_units - len(retrieved), len(remaining))
            selected = (*retrieved, *self.rng.sample(remaining, fill_count))
        cov = len(set(retrieved_unit_ids) & set(selected)) / max(1, len(set(retrieved_unit_ids)))
        return [
            CandidatePage(
                candidate_id=f"{self.name}:{query.query_id}:{self.rng.randrange(10**12)}",
                unit_ids=selected,
                anchor_query_id=query.query_id,
                anchor_unit_ids=tuple(retrieved_unit_ids),
                predicted_coverage=cov,
                score=cov,
            )
        ]
