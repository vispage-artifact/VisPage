"""Strict serial visual memory runtime runner."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Callable

from visual_memory_system.locality.base import LocalityEstimator
from visual_memory_system.page.builder import PageBuilder
from visual_memory_system.page.renderer import PillowPageRenderer
from visual_memory_system.policies.foreground import ForegroundPolicy
from visual_memory_system.policies.prewarm import PrewarmPolicy
from visual_memory_system.retrieval.base import Retriever
from visual_memory_system.runtime.client import RuntimeClient
from visual_memory_system.schema import (
    CacheState,
    CandidatePage,
    ExecutionPath,
    ForegroundPlan,
    MemoryUnit,
    QueryRecord,
    RegisteredPage,
    RuntimeTraceRow,
    VisualPage,
    coverage,
)


ProgressCallback = Callable[[int, int, RuntimeTraceRow], None]


@dataclass
class RunResult:
    rows: list[RuntimeTraceRow] = field(default_factory=list)
    registered_pages: dict[str, RegisteredPage] = field(default_factory=dict)
    background_errors: list[str] = field(default_factory=list)


class SystemRunner:
    """Run demand queries strictly serially.

    Lifecycle invariant:
    - future foreground selection sees only registered pages;
    - page proposal happens only after baseline_cold;
    - only submitted pages are registered.
    """

    def __init__(
        self,
        *,
        memory_units: list[MemoryUnit],
        retriever: Retriever,
        locality_estimator: LocalityEstimator,
        page_builder: PageBuilder,
        renderer: PillowPageRenderer,
        foreground_policy: ForegroundPolicy,
        prewarm_policy: PrewarmPolicy,
        runtime_client: RuntimeClient,
        topk: int,
        max_background_workers: int = 1,
        append_foreground_pages: bool = False,
        disable_page_registration: bool = False,
        register_foreground_residual_pages: bool = True,
        max_foreground_update_amplification: float | None = None,
        max_visual_tokens: int | None = None,
        max_cache_visual_tokens: int | None = None,
        temporal_memory_mask: bool = False,
    ) -> None:
        if topk <= 0:
            raise ValueError("topk must be positive")
        if max_background_workers <= 0:
            raise ValueError("max_background_workers must be positive")
        if max_visual_tokens is not None and max_visual_tokens <= 0:
            raise ValueError("max_visual_tokens must be positive")
        if max_cache_visual_tokens is not None and max_cache_visual_tokens <= 0:
            raise ValueError("max_cache_visual_tokens must be positive")
        if (
            max_foreground_update_amplification is not None
            and max_foreground_update_amplification <= 0
        ):
            raise ValueError("max_foreground_update_amplification must be positive")
        if not memory_units:
            raise ValueError("memory_units must not be empty")
        if topk > len(memory_units):
            raise ValueError(f"topk={topk} exceeds memory unit count={len(memory_units)}")
        unit_ids = [unit.unit_id for unit in memory_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("memory_units contain duplicate unit_id values")
        self.memory_units = memory_units
        self.memory_unit_ids = set(unit_ids)
        self.retriever = retriever
        self.locality_estimator = locality_estimator
        self.page_builder = page_builder
        self.renderer = renderer
        self.foreground_policy = foreground_policy
        self.prewarm_policy = prewarm_policy
        self.runtime_client = runtime_client
        self.topk = topk
        self.append_foreground_pages = append_foreground_pages
        self.disable_page_registration = disable_page_registration
        self.register_foreground_residual_pages = register_foreground_residual_pages
        self.max_foreground_update_amplification = max_foreground_update_amplification
        self.max_visual_tokens = max_visual_tokens
        self.max_cache_visual_tokens = max_cache_visual_tokens or max_visual_tokens
        self.temporal_memory_mask = temporal_memory_mask
        self.background_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_background_workers
        )
        self.background_futures: list[tuple[str, concurrent.futures.Future]] = []
        self.background_errors: list[str] = []
        self.registered_pages: dict[str, RegisteredPage] = {}
        self.foreground_input_pages_by_key: dict[str, tuple[VisualPage, ...]] = {}
        self.memory_by_id = {unit.unit_id: unit for unit in memory_units}

    def run(
        self,
        queries: list[QueryRecord],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> RunResult:
        self._validate_query_order(queries)
        rows: list[RuntimeTraceRow] = []
        total_queries = len(queries)
        try:
            for index, query in enumerate(queries, start=1):
                row = self.run_one(query)
                rows.append(row)
                if progress_callback is not None:
                    progress_callback(index, total_queries, row)
        finally:
            self._drain_background()
        return RunResult(
            rows=rows,
            registered_pages=dict(self.registered_pages),
            background_errors=list(self.background_errors),
        )

    def run_one(self, query: QueryRecord) -> RuntimeTraceRow:
        self._collect_background_completions()
        available_memory_units = self._available_memory_units(query)
        effective_topk = min(self.topk, len(available_memory_units))
        if effective_topk <= 0:
            raise ValueError(f"query {query.query_id} has no available memory units")
        retrieved = self.retriever.retrieve(query, available_memory_units, effective_topk)
        self._validate_retrieved_units(query, retrieved, expected_topk=effective_topk)
        temporal_masked_count = len(self.memory_units) - len(available_memory_units)

        eligible = (
            []
            if self.disable_page_registration
            else self._causally_visible_registered_pages(
                retrieved_unit_ids=retrieved,
                registered_pages=self.registered_pages,
                available_unit_ids={unit.unit_id for unit in available_memory_units},
            )
        )
        selection = self.runtime_client.select_ready_page([reg.page.page_key for reg in eligible])
        plan = self.foreground_policy.plan(
            query=query,
            retrieved_unit_ids=retrieved,
            selection=selection,
            registered_pages=self.registered_pages,
        )
        plan = self._materialize_foreground_plan(plan)
        budget_fallback = False
        original_plan_visual_tokens = self._plan_visual_tokens(plan)
        if self._exceeds_visual_budget(plan):
            if plan.mode != "warm_page":
                raise RuntimeError(
                    f"baseline foreground for {query.query_id} estimates "
                    f"{original_plan_visual_tokens} visual tokens, exceeding "
                    f"page.max_visual_tokens={self.max_visual_tokens}"
                )
            budget_fallback = True
            plan = ForegroundPlan(
                mode="baseline_cold",
                query=query,
                retrieved_unit_ids=tuple(retrieved),
                selected_page=None,
                residual_unit_ids=tuple(retrieved),
                lease_id=None,
                coverage=0.0,
            )
            plan = self._materialize_foreground_plan(plan)
            if self._exceeds_visual_budget(plan):
                raise RuntimeError(
                    f"budget fallback for {query.query_id} estimates "
                    f"{self._plan_visual_tokens(plan)} visual tokens, exceeding "
                    f"page.max_visual_tokens={self.max_visual_tokens}"
                )
        response = self.runtime_client.send_foreground(plan)

        submitted_page_keys: list[str] = []
        foreground_update_registration_skipped_reason: str | None = None
        foreground_update_registration_amplification: float | None = None
        foreground_update_registration_limit = self.max_foreground_update_amplification
        if plan.mode == "baseline_cold":
            if self.disable_page_registration:
                pass
            elif self.append_foreground_pages:
                if plan.cache_page is None:
                    raise ValueError(f"append baseline plan for {query.query_id} lacks cache_page")
                registered_key = self._register_ready_foreground_page(
                    page=plan.cache_page,
                    query=query,
                    source="append_fallback",
                    metadata={"retrieved_unit_ids": tuple(retrieved)},
                    foreground_input_pages=plan.input_pages,
                )
                if registered_key is not None:
                    submitted_page_keys.append(registered_key)
            else:
                candidates = self.locality_estimator.propose_pages(
                    query=query,
                    retrieved_unit_ids=retrieved,
                    memory_units=available_memory_units,
                    registered_pages=self.registered_pages,
                )
                if not candidates:
                    raise ValueError(
                        f"locality estimator {self.locality_estimator.name} returned no candidates "
                        f"after baseline_cold for query {query.query_id}"
                    )
                candidates = self._fit_candidates_to_visual_budget(
                    candidates=candidates,
                    retrieved_unit_ids=retrieved,
                )
                admitted = self.prewarm_policy.admit(
                    candidates=candidates,
                    registered_pages=self.registered_pages,
                    inflight_count=self._inflight_background_count(),
                )
                for candidate in admitted:
                    page = self.page_builder.build(candidate)
                    page = self.renderer.render_page(page, self.memory_by_id)
                    if self._page_exceeds_visual_budget(page):
                        continue
                    if page.page_key in self.registered_pages:
                        raise ValueError(
                            f"candidate {candidate.candidate_id} built duplicate registered page "
                            f"{page.page_key}"
                        )
                    self._submit_background(page)
                    self.registered_pages[page.page_key] = RegisteredPage(
                        page=page,
                        state=CacheState.WARMING,
                        submitted_at_query_index=query.query_index,
                        metadata={"candidate_id": candidate.candidate_id},
                    )
                    submitted_page_keys.append(page.page_key)
        elif plan.selected_page is not None and not self.disable_page_registration:
            self.registered_pages[
                plan.selected_page.page_key
            ].last_selected_query_index = query.query_index
            if (
                self.register_foreground_residual_pages
                and plan.cache_page is not None
                and plan.cache_page.page_key not in self.registered_pages
            ):
                foreground_update_registration_amplification = (
                    len(plan.cache_page.unit_ids) / len(plan.retrieved_unit_ids)
                    if plan.retrieved_unit_ids
                    else 0.0
                )
                if self._foreground_update_exceeds_amplification_limit(plan.cache_page, plan):
                    foreground_update_registration_skipped_reason = "amplification_limit"
                else:
                    registered_key = self._register_ready_foreground_page(
                        page=plan.cache_page,
                        query=query,
                        source="foreground_residual",
                        metadata={
                            "selected_page_key": plan.selected_page.page_key,
                            "residual_unit_ids": plan.residual_unit_ids,
                            "foreground_update_amplification": (
                                foreground_update_registration_amplification
                            ),
                            "foreground_update_amplification_limit": (
                                foreground_update_registration_limit
                            ),
                        },
                        foreground_input_pages=plan.input_pages,
                    )
                    if registered_key is not None:
                        submitted_page_keys.append(registered_key)

        carried_unit_ids = self._carried_unit_ids(plan)
        execution_path = self._execution_path(plan)
        background_inflight = self._inflight_background_count()
        return RuntimeTraceRow(
            query_id=query.query_id,
            query_index=query.query_index,
            mode=plan.mode,
            retrieved_unit_count=len(retrieved),
            selected_page_key=plan.selected_page.page_key if plan.selected_page else None,
            coverage=plan.coverage,
            residual_count=len(plan.residual_unit_ids),
            submitted_page_keys=tuple(submitted_page_keys),
            engine_ttft_ms=response.engine_ttft_ms,
            client_ttft_ms=response.client_ttft_ms,
            prompt_tokens=response.prompt_tokens,
            server_ttft_ms=response.server_ttft_ms,
            ttft_ms=(
                response.server_ttft_ms
                if response.server_ttft_ms is not None
                else response.engine_ttft_ms
            ),
            pre_scheduler_delay_ms=response.pre_scheduler_delay_ms,
            output_text=response.output_text,
            execution_path=execution_path,
            evidence_coverage=coverage(retrieved, carried_unit_ids),
            read_amplification=len(carried_unit_ids) / len(retrieved),
            carried_unit_count=len(carried_unit_ids),
            selected_page_unit_count=(
                len(plan.selected_page.unit_ids) if plan.selected_page else 0
            ),
            registered_page_count=len(self.registered_pages),
            background_inflight_count=background_inflight,
            metadata={
                "eligible_registered_pages": len(eligible),
                "background_inflight": background_inflight,
                "selected_runtime_state": selection.kv_state.value if selection.kv_state else None,
                "cache_select_reason": selection.metadata.get("reason"),
                "cache_select_inspected": selection.metadata.get("inspected", []),
                "selected_page_source": (
                    self._registered_page_source(plan.selected_page.page_key)
                    if plan.selected_page
                    else None
                ),
                "selected_page_root_source": (
                    self._registered_page_root_source(plan.selected_page.page_key)
                    if plan.selected_page
                    else None
                ),
                "selected_page_submitted_at": (
                    self.registered_pages[plan.selected_page.page_key].submitted_at_query_index
                    if plan.selected_page
                    else None
                ),
                "selected_page_unit_count": (
                    len(plan.selected_page.unit_ids) if plan.selected_page else 0
                ),
                "visual_tokens_estimate": self._plan_visual_tokens(plan),
                "visual_token_budget": self.max_visual_tokens,
                "cache_visual_token_budget": self.max_cache_visual_tokens,
                "visual_budget_fallback": budget_fallback,
                "foreground_update_registration_skipped": (
                    foreground_update_registration_skipped_reason is not None
                ),
                "foreground_update_registration_skipped_reason": (
                    foreground_update_registration_skipped_reason
                ),
                "foreground_update_amplification": foreground_update_registration_amplification,
                "foreground_update_amplification_limit": foreground_update_registration_limit,
                "visual_tokens_estimate_before_budget_fallback": (
                    original_plan_visual_tokens if budget_fallback else None
                ),
                "retrieved_coverage_best_registered": max(
                    [
                        coverage(retrieved, reg.page.unit_ids)
                        for reg in eligible
                    ],
                    default=0.0,
                ),
                "temporal_memory_mask": self.temporal_memory_mask,
                "temporal_masked_memory_units": temporal_masked_count,
                "available_memory_units": len(available_memory_units),
                "configured_topk": self.topk,
                "effective_topk": effective_topk,
                "runtime_metric": "server_request_to_first_token_ms"
                if response.server_ttft_ms is not None
                else "engine_queue_to_first_token_ms",
                "vllm_internal_metrics": response.metadata.get("vllm_internal_metrics", {}),
            },
        )

    def _foreground_update_exceeds_amplification_limit(
        self,
        page: VisualPage,
        plan: ForegroundPlan,
    ) -> bool:
        if self.max_foreground_update_amplification is None:
            return False
        if not plan.retrieved_unit_ids:
            return False
        amplification = len(page.unit_ids) / len(plan.retrieved_unit_ids)
        return amplification > self.max_foreground_update_amplification

    def _materialize_foreground_plan(self, plan):
        if plan.mode == "warm_page":
            if plan.selected_page is None:
                raise ValueError(f"warm_page plan for {plan.query.query_id} lacks selected_page")
            input_pages = list(self._registered_foreground_input_pages(plan.selected_page))
            cache_page = None
            if plan.residual_unit_ids:
                residual_page = self.page_builder.build_ephemeral(
                    key_prefix="residual",
                    query_id=plan.query.query_id,
                    unit_ids=plan.residual_unit_ids,
                )
                input_pages.append(self.renderer.render_page(residual_page, self.memory_by_id))
                cache_page = self._build_foreground_residual_page(plan)
            return ForegroundPlan(
                mode=plan.mode,
                query=plan.query,
                retrieved_unit_ids=plan.retrieved_unit_ids,
                input_pages=tuple(input_pages),
                selected_page=plan.selected_page,
                cache_page=cache_page,
                residual_unit_ids=plan.residual_unit_ids,
                lease_id=plan.lease_id,
                coverage=plan.coverage,
            )

        if self.append_foreground_pages:
            baseline_page = self._build_append_fallback_page(plan)
        else:
            baseline_page = self.page_builder.build_ephemeral(
                key_prefix="baseline",
                query_id=plan.query.query_id,
                unit_ids=plan.retrieved_unit_ids,
            )
        baseline_page = self.renderer.render_page(baseline_page, self.memory_by_id)
        return ForegroundPlan(
            mode=plan.mode,
            query=plan.query,
            retrieved_unit_ids=plan.retrieved_unit_ids,
            input_pages=(baseline_page,),
            selected_page=None,
            cache_page=baseline_page if self.append_foreground_pages else None,
            residual_unit_ids=plan.residual_unit_ids,
            lease_id=None,
            coverage=plan.coverage,
        )

    def _build_append_fallback_page(self, plan: ForegroundPlan) -> VisualPage:
        candidate = CandidatePage(
            candidate_id=f"append_fallback:{plan.query.query_id}",
            unit_ids=plan.retrieved_unit_ids,
            anchor_query_id=plan.query.query_id,
            anchor_unit_ids=plan.retrieved_unit_ids,
            predicted_coverage=1.0,
            score=1.0,
            metadata={"source": "append_fallback"},
        )
        return self.page_builder.build(candidate)

    def _build_foreground_residual_page(self, plan: ForegroundPlan) -> VisualPage:
        if plan.selected_page is None:
            raise ValueError(f"warm_page plan for {plan.query.query_id} lacks selected_page")
        combined_unit_ids = tuple(
            dict.fromkeys((*plan.selected_page.unit_ids, *plan.residual_unit_ids))
        )
        candidate = CandidatePage(
            candidate_id=f"foreground_residual:{plan.query.query_id}:{plan.selected_page.page_key}",
            unit_ids=combined_unit_ids,
            anchor_query_id=plan.query.query_id,
            anchor_unit_ids=plan.retrieved_unit_ids,
            predicted_coverage=coverage(plan.retrieved_unit_ids, combined_unit_ids),
            score=coverage(plan.retrieved_unit_ids, combined_unit_ids),
            metadata={
                "source": "foreground_residual",
                "selected_page_key": plan.selected_page.page_key,
                "selected_page_unit_count": len(plan.selected_page.unit_ids),
                "residual_unit_count": len(plan.residual_unit_ids),
                "combined_unit_count": len(combined_unit_ids),
                "foreground_read_amplification": (
                    len(combined_unit_ids) / len(plan.retrieved_unit_ids)
                    if plan.retrieved_unit_ids
                    else 0.0
                ),
            },
        )
        page = self.page_builder.build_unbounded(candidate)
        return self.renderer.render_page(page, self.memory_by_id)

    def _register_ready_foreground_page(
        self,
        *,
        page: VisualPage,
        query: QueryRecord,
        source: str,
        metadata: dict,
        foreground_input_pages: tuple[VisualPage, ...],
    ) -> str | None:
        existing = self.registered_pages.get(page.page_key)
        if existing is not None:
            existing.last_selected_query_index = query.query_index
            return None
        foreground_input_pages = tuple(foreground_input_pages)
        self.foreground_input_pages_by_key[page.page_key] = foreground_input_pages
        self.registered_pages[page.page_key] = RegisteredPage(
            page=page,
            state=CacheState.READY,
            submitted_at_query_index=query.query_index,
            last_selected_query_index=query.query_index,
            metadata={
                "source": source,
                "foreground_input_page_keys": tuple(
                    input_page.page_key for input_page in foreground_input_pages
                ),
                **metadata,
            },
        )
        return page.page_key

    def _registered_foreground_input_pages(self, page: VisualPage) -> tuple[VisualPage, ...]:
        return self.foreground_input_pages_by_key.get(page.page_key, (page,))

    def _registered_page_source(self, page_key: str) -> str | None:
        reg = self.registered_pages.get(page_key)
        if reg is None:
            return None
        return _page_source(reg)

    def _registered_page_root_source(self, page_key: str) -> str | None:
        visited: set[str] = set()
        current_key: str | None = page_key
        root_source: str | None = None
        while current_key is not None and current_key not in visited:
            visited.add(current_key)
            reg = self.registered_pages.get(current_key)
            if reg is None:
                break
            source = _page_source(reg)
            if source is not None:
                root_source = source
            if source != "foreground_residual":
                break
            selected_key = reg.metadata.get("selected_page_key")
            current_key = str(selected_key) if selected_key else None
        return root_source

    def _exceeds_visual_budget(self, plan: ForegroundPlan) -> bool:
        if self.max_visual_tokens is None:
            return False
        return self._plan_visual_tokens(plan) > self.max_visual_tokens

    def _page_exceeds_visual_budget(self, page: VisualPage) -> bool:
        if self.max_cache_visual_tokens is None:
            return False
        return _page_visual_tokens(page) > self.max_cache_visual_tokens

    def _fit_candidates_to_visual_budget(
        self,
        *,
        candidates: list[CandidatePage],
        retrieved_unit_ids: list[str],
    ) -> list[CandidatePage]:
        if self.max_cache_visual_tokens is None:
            return candidates
        fitted: list[CandidatePage] = []
        for candidate in candidates:
            fitted_candidate = self._fit_candidate_to_visual_budget(
                candidate=candidate,
                retrieved_unit_ids=retrieved_unit_ids,
            )
            if fitted_candidate is not None:
                fitted.append(fitted_candidate)
        fitted.sort(key=lambda page: (-page.score, len(page.unit_ids), page.candidate_id))
        return fitted

    def _fit_candidate_to_visual_budget(
        self,
        *,
        candidate: CandidatePage,
        retrieved_unit_ids: list[str],
    ) -> CandidatePage | None:
        if self.max_cache_visual_tokens is None:
            return candidate
        original_units = tuple(dict.fromkeys(candidate.unit_ids))
        original_tokens = self.renderer.estimate_page_tokens(original_units, self.memory_by_id)
        if original_tokens <= self.max_cache_visual_tokens:
            return candidate

        candidate_set = set(original_units)
        selected: list[str] = []
        selected_set: set[str] = set()
        priority_units = [
            unit_id
            for unit_id in retrieved_unit_ids
            if unit_id in candidate_set
        ]
        ordered_units = [
            *priority_units,
            *(unit_id for unit_id in original_units if unit_id not in set(priority_units)),
        ]
        for unit_id in ordered_units:
            if unit_id in selected_set:
                continue
            trial = tuple((*selected, unit_id))
            if self.renderer.estimate_page_tokens(trial, self.memory_by_id) <= self.max_cache_visual_tokens:
                selected.append(unit_id)
                selected_set.add(unit_id)
        if not selected:
            return None

        selected_units = tuple(selected)
        fitted_coverage = coverage(retrieved_unit_ids, selected_units)
        estimated_tokens = self.renderer.estimate_page_tokens(selected_units, self.memory_by_id)
        return CandidatePage(
            candidate_id=f"{candidate.candidate_id}:budget{self.max_cache_visual_tokens}",
            unit_ids=selected_units,
            anchor_query_id=candidate.anchor_query_id,
            anchor_unit_ids=candidate.anchor_unit_ids,
            predicted_coverage=fitted_coverage,
            score=fitted_coverage,
            metadata={
                **candidate.metadata,
                "budget_trimmed": True,
                "budget_visual_tokens": self.max_cache_visual_tokens,
                "foreground_visual_tokens": self.max_visual_tokens,
                "original_unit_count": len(original_units),
                "budgeted_unit_count": len(selected_units),
                "original_visual_tokens_estimate": original_tokens,
                "budgeted_visual_tokens_estimate": estimated_tokens,
            },
        )

    @staticmethod
    def _plan_visual_tokens(plan: ForegroundPlan) -> int:
        return sum(_page_visual_tokens(page) for page in plan.input_pages)

    @staticmethod
    def _validate_query_order(queries: list[QueryRecord]) -> None:
        seen_query_ids: set[str] = set()
        last_index_by_task: dict[str, int] = {}
        for position, query in enumerate(queries):
            if query.query_id in seen_query_ids:
                raise ValueError(f"duplicate query_id {query.query_id!r} at input position {position}")
            seen_query_ids.add(query.query_id)
            last_index = last_index_by_task.get(query.task_id)
            if last_index is not None and query.query_index <= last_index:
                raise ValueError(
                    f"queries for task {query.task_id!r} are not in strict input trace order: "
                    f"query {query.query_id!r} has index {query.query_index} after {last_index}"
                )
            last_index_by_task[query.task_id] = query.query_index

    def _validate_retrieved_units(
        self,
        query: QueryRecord,
        retrieved: list[str],
        *,
        expected_topk: int | None = None,
    ) -> None:
        expected = self.topk if expected_topk is None else expected_topk
        if len(retrieved) != expected:
            raise ValueError(
                f"query {query.query_id} retrieved {len(retrieved)} units, expected exactly topk={expected}"
            )
        if len(retrieved) != len(set(retrieved)):
            raise ValueError(f"query {query.query_id} retrieved duplicate unit ids")
        missing = [unit_id for unit_id in retrieved if unit_id not in self.memory_unit_ids]
        if missing:
            raise ValueError(f"query {query.query_id} retrieved unknown unit ids: {missing}")
        if self.temporal_memory_mask:
            unavailable = [
                unit_id
                for unit_id in retrieved
                if unit_id not in {unit.unit_id for unit in self._available_memory_units(query)}
            ]
            if unavailable:
                raise ValueError(
                    f"query {query.query_id} retrieved temporally unavailable unit ids: {unavailable}"
                )

    def _available_memory_units(self, query: QueryRecord) -> list[MemoryUnit]:
        if not self.temporal_memory_mask:
            return self.memory_units
        query_date = _query_date(query)
        if query_date is None:
            return self.memory_units
        return [
            unit
            for unit in self.memory_units
            if (unit_date := _memory_unit_date(unit)) is None or unit_date <= query_date
        ]

    def _causally_visible_registered_pages(
        self,
        *,
        retrieved_unit_ids: list[str],
        registered_pages: dict[str, RegisteredPage],
        available_unit_ids: set[str],
    ) -> list[RegisteredPage]:
        if not self.temporal_memory_mask:
            return self.foreground_policy.eligible_registered_pages(
                retrieved_unit_ids=retrieved_unit_ids,
                registered_pages=registered_pages,
            )
        visible_pages = {
            page_key: reg
            for page_key, reg in registered_pages.items()
            if set(reg.page.unit_ids).issubset(available_unit_ids)
        }
        return self.foreground_policy.eligible_registered_pages(
            retrieved_unit_ids=retrieved_unit_ids,
            registered_pages=visible_pages,
        )

    def _submit_background(self, page: VisualPage) -> None:
        future = self.background_executor.submit(self.runtime_client.send_background, page)
        self.background_futures.append((page.page_key, future))

    def _collect_background_completions(self) -> None:
        still_running: list[tuple[str, concurrent.futures.Future]] = []
        for page_key, future in self.background_futures:
            if not future.done():
                still_running.append((page_key, future))
                continue
            self._record_background_result(page_key, future)
        self.background_futures = still_running

    def _drain_background(self) -> None:
        for page_key, future in self.background_futures:
            self._record_background_result(page_key, future)
        self.background_futures = []
        self.background_executor.shutdown(wait=True)

    def _record_background_result(
        self,
        page_key: str,
        future: concurrent.futures.Future,
    ) -> None:
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - exercised by integration/runtime failures.
            self.background_errors.append(f"{page_key}: {exc}")

    def _inflight_background_count(self) -> int:
        return sum(1 for _, future in self.background_futures if not future.done())

    @staticmethod
    def _execution_path(plan: ForegroundPlan) -> ExecutionPath:
        if plan.mode == "baseline_cold":
            return "fallback"
        if not plan.residual_unit_ids:
            return "full_hit"
        return "partial_hit"

    @staticmethod
    def _carried_unit_ids(plan: ForegroundPlan) -> tuple[str, ...]:
        if plan.mode == "baseline_cold":
            return plan.retrieved_unit_ids
        if plan.selected_page is None:
            raise ValueError(f"warm_page plan for {plan.query.query_id} lacks selected_page")
        return (*plan.selected_page.unit_ids, *plan.residual_unit_ids)


def _page_source(reg: RegisteredPage) -> str | None:
    source = reg.metadata.get("source") or reg.page.metadata.get("source")
    if source is not None:
        return str(source)
    candidate_id = reg.metadata.get("candidate_id") or reg.page.metadata.get("candidate_id")
    if candidate_id is None:
        return None
    return str(candidate_id).split(":", 1)[0]


def _page_visual_tokens(page: VisualPage) -> int:
    return int(page.prompt_tokens_estimate or 0)


def _query_date(query: QueryRecord) -> dt.date | None:
    for key in ("question_date", "task_date", "date", "timestamp"):
        parsed = _parse_date(query.metadata.get(key))
        if parsed is not None:
            return parsed
    return None


def _memory_unit_date(unit: MemoryUnit) -> dt.date | None:
    parsed = _parse_date(unit.metadata.get("date"))
    if parsed is not None:
        return parsed
    parsed = _parse_date(unit.metadata.get("timestamp"))
    if parsed is not None:
        return parsed
    if unit.timestamp is not None:
        return dt.datetime.fromtimestamp(unit.timestamp).date()
    return None


def _parse_date(value) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(value))
    if not match:
        return None
    try:
        return dt.date.fromisoformat(match.group(1))
    except ValueError:
        return None
