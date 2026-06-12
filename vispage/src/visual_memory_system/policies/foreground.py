"""Foreground planning policy."""

from __future__ import annotations

from visual_memory_system.schema import (
    CacheSelection,
    ForegroundPlan,
    QueryRecord,
    RegisteredPage,
    VisualPage,
    coverage,
)


class ForegroundPolicy:
    def __init__(self, *, min_coverage: float) -> None:
        if not 0 <= min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        self.min_coverage = min_coverage

    def eligible_registered_pages(
        self,
        *,
        retrieved_unit_ids: list[str],
        registered_pages: dict[str, RegisteredPage],
    ) -> list[RegisteredPage]:
        eligible = [
            reg
            for reg in registered_pages.values()
            if coverage(retrieved_unit_ids, reg.page.unit_ids) >= self.min_coverage
        ]
        eligible.sort(
            key=lambda reg: (
                -coverage(retrieved_unit_ids, reg.page.unit_ids),
                len(reg.page.unit_ids),
                reg.page.page_key,
            )
        )
        return eligible

    def plan(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        selection: CacheSelection,
        registered_pages: dict[str, RegisteredPage],
    ) -> ForegroundPlan:
        if selection.has_ready_page:
            if selection.selected_page_key not in registered_pages:
                raise KeyError(f"runtime selected unregistered page {selection.selected_page_key}")
            page = registered_pages[selection.selected_page_key].page
            cov = coverage(retrieved_unit_ids, page.unit_ids)
            if cov < self.min_coverage:
                raise ValueError(
                    f"runtime selected page coverage {cov:.3f} below threshold {self.min_coverage:.3f}"
                )
            page_unit_ids = set(page.unit_ids)
            residual = tuple(unit_id for unit_id in retrieved_unit_ids if unit_id not in page_unit_ids)
            return ForegroundPlan(
                mode="warm_page",
                query=query,
                retrieved_unit_ids=tuple(retrieved_unit_ids),
                selected_page=page,
                residual_unit_ids=residual,
                lease_id=selection.lease_id,
                coverage=cov,
            )

        return ForegroundPlan(
            mode="baseline_cold",
            query=query,
            retrieved_unit_ids=tuple(retrieved_unit_ids),
            selected_page=None,
            residual_unit_ids=tuple(retrieved_unit_ids),
            lease_id=None,
            coverage=0.0,
        )


def selected_page_or_none(plan: ForegroundPlan) -> VisualPage | None:
    return plan.selected_page
