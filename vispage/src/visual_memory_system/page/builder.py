"""Build stable visual pages from admitted candidates."""

from __future__ import annotations

from visual_memory_system.schema import CandidatePage, VisualPage, stable_page_key


class PageBuilder:
    def __init__(self, *, key_prefix: str = "vmpage", max_units: int | None = None) -> None:
        self.key_prefix = key_prefix
        self.max_units = max_units

    def build(self, candidate: CandidatePage) -> VisualPage:
        return self._build(candidate, enforce_max_units=True)

    def build_unbounded(self, candidate: CandidatePage) -> VisualPage:
        return self._build(candidate, enforce_max_units=False)

    def _build(self, candidate: CandidatePage, *, enforce_max_units: bool) -> VisualPage:
        if len(candidate.unit_ids) != len(set(candidate.unit_ids)):
            raise ValueError(f"candidate {candidate.candidate_id} contains duplicate unit ids")
        if enforce_max_units and self.max_units is not None and len(candidate.unit_ids) > self.max_units:
            raise ValueError(
                f"candidate {candidate.candidate_id} has {len(candidate.unit_ids)} units, "
                f"exceeds max_units={self.max_units}"
            )
        unit_ids = tuple(sorted(candidate.unit_ids))
        if not unit_ids:
            raise ValueError(f"candidate {candidate.candidate_id} has no units")
        page_key = stable_page_key(self.key_prefix, unit_ids)
        return VisualPage(
            page_key=page_key,
            unit_ids=unit_ids,
            metadata={
                **candidate.metadata,
                "candidate_id": candidate.candidate_id,
                "anchor_query_id": candidate.anchor_query_id,
                "predicted_coverage": candidate.predicted_coverage,
            },
        )

    def build_ephemeral(
        self,
        *,
        key_prefix: str,
        query_id: str,
        unit_ids: tuple[str, ...],
    ) -> VisualPage:
        if not unit_ids:
            raise ValueError(f"ephemeral page for query {query_id} has no units")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError(f"ephemeral page for query {query_id} contains duplicate unit ids")
        canonical_ids = tuple(sorted(unit_ids))
        page_key = stable_page_key(f"{self.key_prefix}:{key_prefix}:{query_id}", canonical_ids)
        return VisualPage(
            page_key=page_key,
            unit_ids=canonical_ids,
            metadata={"query_id": query_id, "ephemeral": True, "kind": key_prefix},
        )
