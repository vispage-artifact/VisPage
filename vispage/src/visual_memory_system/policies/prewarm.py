"""Background prewarm admission policy."""

from __future__ import annotations

from visual_memory_system.schema import CandidatePage, RegisteredPage


class PrewarmPolicy:
    def __init__(
        self,
        *,
        min_candidate_coverage: float,
        max_new_in_old: float | None = None,
        max_inflight_pages: int | None = None,
    ) -> None:
        if not 0 <= min_candidate_coverage <= 1:
            raise ValueError("min_candidate_coverage must be in [0, 1]")
        if max_new_in_old is not None and not 0 <= max_new_in_old <= 1:
            raise ValueError("max_new_in_old must be in [0, 1]")
        self.min_candidate_coverage = min_candidate_coverage
        self.max_new_in_old = max_new_in_old
        self.max_inflight_pages = max_inflight_pages

    def admit(
        self,
        *,
        candidates: list[CandidatePage],
        registered_pages: dict[str, RegisteredPage],
        inflight_count: int = 0,
    ) -> list[CandidatePage]:
        if self.max_inflight_pages is not None and inflight_count >= self.max_inflight_pages:
            return []

        admitted: list[CandidatePage] = []
        registered_unit_sets = [set(reg.page.unit_ids) for reg in registered_pages.values()]
        for candidate in candidates:
            if candidate.predicted_coverage < self.min_candidate_coverage:
                continue
            if self.max_new_in_old is not None and registered_unit_sets:
                new_set = set(candidate.unit_ids)
                max_overlap = max(len(new_set & old_set) / max(1, len(new_set)) for old_set in registered_unit_sets)
                if max_overlap > self.max_new_in_old:
                    continue
            admitted.append(candidate)
            if self.max_inflight_pages is not None and inflight_count + len(admitted) >= self.max_inflight_pages:
                break
        return admitted

