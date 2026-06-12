import threading

from visual_memory_system.data.generic_jsonl import GenericJsonlAdapter
from visual_memory_system.locality.append import AppendEstimator
from visual_memory_system.locality.base import LocalityEstimator
from visual_memory_system.page.builder import PageBuilder
from visual_memory_system.policies.foreground import ForegroundPolicy
from visual_memory_system.policies.prewarm import PrewarmPolicy
from visual_memory_system.retrieval.base import Retriever
from visual_memory_system.runtime.client import RuntimeClient
from visual_memory_system.runtime.vllm import VllmRuntimeClient
from visual_memory_system.runner.system_runner import SystemRunner
from visual_memory_system.schema import (
    CacheSelection,
    CacheState,
    CandidatePage,
    MemoryUnit,
    QueryRecord,
    RegisteredPage,
    RuntimeResponse,
    VisualPage,
)


def test_runner_rejects_query_reordering() -> None:
    queries = [
        QueryRecord(query_id="q2", task_id="t", query_index=2, query_text="q2", access_units=("u1",)),
        QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q1", access_units=("u1",)),
    ]

    try:
        SystemRunner._validate_query_order(queries)
    except ValueError as exc:
        assert "strict input trace order" in str(exc)
    else:
        raise AssertionError("expected runner to reject out-of-order query trace")


def test_loader_requires_access_fields(tmp_path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    query_path = tmp_path / "queries.jsonl"
    memory_path.write_text('{"unit_id":"u1","text":"hello"}\n', encoding="utf-8")
    query_path.write_text(
        '{"query_id":"q1","task_id":"t","query_index":1,"query_text":"q"}\n',
        encoding="utf-8",
    )

    adapter = GenericJsonlAdapter(memory_path, query_path)
    try:
        adapter.load_queries()
    except ValueError as exc:
        assert "missing required fields" in str(exc)
        assert "access_units" in str(exc)
        assert "access_type" in str(exc)
    else:
        raise AssertionError("expected loader to reject implicit access fields")


def test_page_builder_rejects_duplicate_candidate_units() -> None:
    candidate = CandidatePage(
        candidate_id="c",
        unit_ids=("u1", "u1"),
        anchor_query_id="q1",
        anchor_unit_ids=("u1",),
        predicted_coverage=1.0,
        score=1.0,
    )
    try:
        PageBuilder().build(candidate)
    except ValueError as exc:
        assert "duplicate unit ids" in str(exc)
    else:
        raise AssertionError("expected duplicate candidate units to be rejected")


def test_foreground_policy_falls_back_without_ready_page() -> None:
    policy = ForegroundPolicy(min_coverage=0.5)
    registered_pages = {
        "p1": RegisteredPage(page=VisualPage(page_key="p1", unit_ids=("u1", "u2")))
    }
    plan = policy.plan(
        query=QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"),
        retrieved_unit_ids=["u1", "u2"],
        selection=CacheSelection(selected_page_key=None, kv_state=CacheState.MISSING),
        registered_pages=registered_pages,
    )
    assert plan.mode == "baseline_cold"
    assert plan.selected_page is None
    assert plan.residual_unit_ids == ("u1", "u2")


def test_foreground_policy_uses_only_ready_selected_page() -> None:
    policy = ForegroundPolicy(min_coverage=0.5)
    registered_pages = {
        "p1": RegisteredPage(page=VisualPage(page_key="p1", unit_ids=("u1", "u2")))
    }
    plan = policy.plan(
        query=QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"),
        retrieved_unit_ids=["u1", "u2", "u3"],
        selection=CacheSelection(
            selected_page_key="p1",
            lease_id="lease",
            kv_state=CacheState.READY,
        ),
        registered_pages=registered_pages,
    )
    assert plan.mode == "warm_page"
    assert plan.selected_page == registered_pages["p1"].page
    assert plan.residual_unit_ids == ("u3",)
    assert plan.coverage == 2 / 3


def test_foreground_policy_prefers_higher_coverage_page_over_smaller_page() -> None:
    policy = ForegroundPolicy(min_coverage=0.3)
    registered_pages = {
        "large": RegisteredPage(
            page=VisualPage(
                page_key="large",
                unit_ids=("u1", "u2", "u3", "x1", "x2", "x3", "x4", "x5"),
            )
        ),
        "small": RegisteredPage(page=VisualPage(page_key="small", unit_ids=("u1", "u2"))),
    }

    eligible = policy.eligible_registered_pages(
        retrieved_unit_ids=["u1", "u2", "u3", "u4", "u5"],
        registered_pages=registered_pages,
    )

    assert [reg.page.page_key for reg in eligible] == ["large", "small"]


def test_foreground_policy_uses_smaller_page_as_coverage_tiebreaker() -> None:
    policy = ForegroundPolicy(min_coverage=0.3)
    registered_pages = {
        "large": RegisteredPage(
            page=VisualPage(page_key="large", unit_ids=("u1", "u2", "x1", "x2"))
        ),
        "small": RegisteredPage(page=VisualPage(page_key="small", unit_ids=("u1", "u2"))),
    }

    eligible = policy.eligible_registered_pages(
        retrieved_unit_ids=["u1", "u2", "u3"],
        registered_pages=registered_pages,
    )

    assert [reg.page.page_key for reg in eligible] == ["small", "large"]


def test_runner_does_not_wait_for_background_prewarm() -> None:
    runtime = BlockingBackgroundRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1"]),
        locality_estimator=FixedLocalityEstimator(("u1", "u2")),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=1,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))
    assert row.mode == "baseline_cold"
    assert len(row.submitted_page_keys) == 1
    assert runtime.background_started.wait(timeout=1.0)

    runtime.release_background.set()
    runner._drain_background()


def test_runner_registers_foreground_residual_page_for_future_selection() -> None:
    runtime = WarmResidualRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=FixedLocalityEstimator(("u1", "u2")),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
    )
    runner.registered_pages["p1"] = RegisteredPage(
        page=VisualPage(page_key="p1", unit_ids=("u1",), image_path="/tmp/fake-page.png"),
        state=CacheState.READY,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))

    assert row.mode == "warm_page"
    assert row.residual_count == 1
    assert len(row.submitted_page_keys) == 1
    registered_key = row.submitted_page_keys[0]
    assert registered_key in runner.registered_pages
    registered = runner.registered_pages[registered_key]
    assert registered.state == CacheState.READY
    assert registered.page.unit_ids == ("u1", "u2")
    assert registered.metadata["source"] == "foreground_residual"
    assert runtime.last_plan is not None
    assert runtime.last_plan.cache_page is not None
    assert runtime.last_plan.cache_page.page_key == registered_key


def test_runner_can_skip_foreground_residual_registration() -> None:
    runtime = WarmResidualRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=FixedLocalityEstimator(("u1", "u2")),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
        register_foreground_residual_pages=False,
    )
    runner.registered_pages["p1"] = RegisteredPage(
        page=VisualPage(page_key="p1", unit_ids=("u1",), image_path="/tmp/fake-page.png"),
        state=CacheState.READY,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))

    assert row.mode == "warm_page"
    assert row.residual_count == 1
    assert row.submitted_page_keys == ()
    assert set(runner.registered_pages) == {"p1"}
    assert runtime.last_plan is not None
    assert runtime.last_plan.cache_page is not None


def test_foreground_residual_registration_respects_update_amplification_limit() -> None:
    runtime = WarmResidualRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=FixedLocalityEstimator(("u1", "u2")),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
        max_foreground_update_amplification=0.75,
    )
    runner.registered_pages["p1"] = RegisteredPage(
        page=VisualPage(page_key="p1", unit_ids=("u1",), image_path="/tmp/fake-page.png"),
        state=CacheState.READY,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))

    assert row.mode == "warm_page"
    assert row.residual_count == 1
    assert row.submitted_page_keys == ()
    assert set(runner.registered_pages) == {"p1"}
    assert row.metadata["foreground_update_registration_skipped"] is True
    assert row.metadata["foreground_update_registration_skipped_reason"] == "amplification_limit"
    assert row.metadata["foreground_update_amplification"] == 1.0
    assert row.metadata["foreground_update_amplification_limit"] == 0.75
    assert runtime.last_plan is not None
    assert runtime.last_plan.cache_page is not None


def test_foreground_residual_registration_ignores_page_max_units() -> None:
    runtime = WarmResidualRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=FixedLocalityEstimator(("u1",)),
        page_builder=PageBuilder(key_prefix="test", max_units=1),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
    )
    runner.registered_pages["p1"] = RegisteredPage(
        page=VisualPage(page_key="p1", unit_ids=("u1",), image_path="/tmp/fake-page.png"),
        state=CacheState.READY,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))

    assert row.mode == "warm_page"
    assert row.read_amplification == 1.0
    assert len(row.submitted_page_keys) == 1
    registered = runner.registered_pages[row.submitted_page_keys[0]]
    assert registered.page.unit_ids == ("u1", "u2")
    assert registered.page.metadata["combined_unit_count"] == 2
    assert registered.page.metadata["foreground_read_amplification"] == 1.0


def test_append_mode_registers_fallback_foreground_page_without_background() -> None:
    runtime = AppendRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=AppendEstimator(),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
        append_foreground_pages=True,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))

    assert row.mode == "baseline_cold"
    assert row.coverage == 0.0
    assert row.background_inflight_count == 0
    assert len(row.submitted_page_keys) == 1
    registered_key = row.submitted_page_keys[0]
    assert registered_key in runner.registered_pages
    registered = runner.registered_pages[registered_key]
    assert registered.state == CacheState.READY
    assert registered.page.unit_ids == ("u1", "u2")
    assert registered.metadata["source"] == "append_fallback"
    assert runtime.background_pages == []
    assert runtime.last_plan is not None
    assert runtime.last_plan.cache_page is not None
    assert runtime.last_plan.cache_page.page_key == registered_key


def test_baseline_mode_never_registers_or_submits_pages() -> None:
    runtime = AppendRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=FixedLocalityEstimator(("u1", "u2")),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
        disable_page_registration=True,
    )

    first = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q1"))
    second = runner.run_one(QueryRecord(query_id="q2", task_id="t", query_index=2, query_text="q2"))

    assert first.mode == "baseline_cold"
    assert second.mode == "baseline_cold"
    assert first.submitted_page_keys == ()
    assert second.submitted_page_keys == ()
    assert runner.registered_pages == {}
    assert runtime.background_pages == []
    assert runtime.last_plan is not None
    assert runtime.last_plan.cache_page is None


def test_visual_budget_falls_back_from_oversized_warm_page() -> None:
    runtime = WarmResidualRuntime()
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
        ],
        retriever=FixedRetriever(["u1", "u2"]),
        locality_estimator=FixedLocalityEstimator(("u1", "u2")),
        page_builder=PageBuilder(key_prefix="test", max_units=2),
        renderer=BudgetRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
        max_visual_tokens=100,
    )
    runner.registered_pages["p1"] = RegisteredPage(
        page=VisualPage(
            page_key="p1",
            unit_ids=("u1",),
            image_path="/tmp/fake-large-page.png",
            prompt_tokens_estimate=120,
        ),
        state=CacheState.READY,
    )

    row = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q"))

    assert row.mode == "baseline_cold"
    assert row.selected_page_key is None
    assert row.metadata["visual_budget_fallback"] is True
    assert row.metadata["visual_tokens_estimate_before_budget_fallback"] == 170
    assert row.metadata["visual_tokens_estimate"] == 20
    assert runtime.last_plan is not None
    assert [page.page_key for page in runtime.last_plan.input_pages] != ["p1"]


def test_registered_foreground_residual_page_reuses_original_input_sequence() -> None:
    runtime = CompositeSelectingRuntime(seed_page_key="p1")
    runner = SystemRunner(
        memory_units=[
            MemoryUnit(unit_id="u1", text="one"),
            MemoryUnit(unit_id="u2", text="two"),
            MemoryUnit(unit_id="u3", text="three"),
        ],
        retriever=SequenceRetriever([["u1", "u2"], ["u1", "u2", "u3"]]),
        locality_estimator=AppendEstimator(),
        page_builder=PageBuilder(key_prefix="test", max_units=3),
        renderer=PassthroughRenderer(),
        foreground_policy=ForegroundPolicy(min_coverage=0.5),
        prewarm_policy=PrewarmPolicy(min_candidate_coverage=0.0, max_inflight_pages=1),
        runtime_client=runtime,
        topk=2,
        append_foreground_pages=True,
    )
    runner.registered_pages["p1"] = RegisteredPage(
        page=VisualPage(page_key="p1", unit_ids=("u1",), image_path="/tmp/fake-page.png"),
        state=CacheState.READY,
    )

    first = runner.run_one(QueryRecord(query_id="q1", task_id="t", query_index=1, query_text="q1"))
    composite_key = first.submitted_page_keys[0]
    second = runner.run_one(QueryRecord(query_id="q2", task_id="t", query_index=2, query_text="q2"))

    assert composite_key == second.selected_page_key
    first_input_keys = [page.page_key for page in runtime.foreground_plans[0].input_pages]
    second_input_keys = [page.page_key for page in runtime.foreground_plans[1].input_pages]
    assert first_input_keys == second_input_keys
    assert first_input_keys != [composite_key]


def test_vllm_cache_select_uses_candidates_payload() -> None:
    captured = {}
    client = VllmRuntimeClient(base_url="http://127.0.0.1:1", model="m")

    def fake_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"selected": None, "kv_state": "missing"}

    client._post_json = fake_post

    selection = client.select_ready_page(["p1", "p2"])

    assert captured == {
        "path": "/v1/am/cache/select",
        "payload": {"candidates": [{"cache_key": "p1"}, {"cache_key": "p2"}]},
    }
    assert selection.selected_page_key is None
    assert selection.kv_state == CacheState.MISSING


class FixedRetriever(Retriever):
    name = "fixed"

    def __init__(self, unit_ids: list[str]) -> None:
        self.unit_ids = unit_ids

    def retrieve(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        del query, memory_units
        return self.unit_ids[:topk]


class SequenceRetriever(Retriever):
    name = "sequence"

    def __init__(self, unit_ids_by_call: list[list[str]]) -> None:
        self.unit_ids_by_call = unit_ids_by_call
        self.index = 0

    def retrieve(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        del query, memory_units
        value = self.unit_ids_by_call[self.index]
        self.index += 1
        return value[:topk]


class FixedLocalityEstimator(LocalityEstimator):
    name = "fixed"

    def __init__(self, unit_ids: tuple[str, ...]) -> None:
        self.unit_ids = unit_ids

    def propose_pages(
        self,
        *,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        memory_units: list[MemoryUnit],
        registered_pages: dict[str, RegisteredPage],
    ) -> list[CandidatePage]:
        del retrieved_unit_ids, memory_units, registered_pages
        return [
            CandidatePage(
                candidate_id=f"candidate:{query.query_id}",
                unit_ids=self.unit_ids,
                anchor_query_id=query.query_id,
                anchor_unit_ids=(self.unit_ids[0],),
                predicted_coverage=1.0,
                score=1.0,
            )
        ]


class PassthroughRenderer:
    def render_page(
        self,
        page: VisualPage,
        memory_by_id: dict[str, MemoryUnit],
    ) -> VisualPage:
        del memory_by_id
        return VisualPage(
            page_key=page.page_key,
            unit_ids=page.unit_ids,
            image_path="/tmp/fake-page.png",
            prompt_tokens_estimate=page.prompt_tokens_estimate,
            metadata=page.metadata,
        )


class BudgetRenderer:
    def render_page(
        self,
        page: VisualPage,
        memory_by_id: dict[str, MemoryUnit],
    ) -> VisualPage:
        del memory_by_id
        if page.metadata.get("kind") == "residual":
            tokens = 50
        elif page.metadata.get("kind") == "baseline":
            tokens = 20
        else:
            tokens = page.prompt_tokens_estimate or 20
        return VisualPage(
            page_key=page.page_key,
            unit_ids=page.unit_ids,
            image_path="/tmp/fake-page.png",
            prompt_tokens_estimate=tokens,
            metadata=page.metadata,
        )


class BlockingBackgroundRuntime(RuntimeClient):
    def __init__(self) -> None:
        self.background_started = threading.Event()
        self.release_background = threading.Event()

    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        del page_keys
        return CacheSelection(selected_page_key=None, kv_state=CacheState.MISSING)

    def send_foreground(self, plan) -> RuntimeResponse:
        return RuntimeResponse(
            request_id=f"fg:{plan.query.query_id}",
            engine_ttft_ms=1.0,
            client_ttft_ms=1.0,
            prompt_tokens=1,
        )

    def send_background(self, page: VisualPage) -> RuntimeResponse:
        self.background_started.set()
        assert self.release_background.wait(timeout=5.0)
        return RuntimeResponse(
            request_id=f"bg:{page.page_key}",
            engine_ttft_ms=1.0,
            client_ttft_ms=1.0,
            prompt_tokens=1,
        )


class WarmResidualRuntime(RuntimeClient):
    def __init__(self) -> None:
        self.last_plan = None

    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        assert page_keys == ["p1"]
        return CacheSelection(
            selected_page_key="p1",
            lease_id="lease",
            kv_state=CacheState.READY,
        )

    def send_foreground(self, plan) -> RuntimeResponse:
        self.last_plan = plan
        return RuntimeResponse(
            request_id=f"fg:{plan.query.query_id}",
            engine_ttft_ms=1.0,
            client_ttft_ms=1.0,
            prompt_tokens=1,
        )

    def send_background(self, page: VisualPage) -> RuntimeResponse:
        raise AssertionError(f"unexpected background submission for {page.page_key}")


class AppendRuntime(RuntimeClient):
    def __init__(self) -> None:
        self.last_plan = None
        self.background_pages: list[VisualPage] = []

    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        del page_keys
        return CacheSelection(selected_page_key=None, kv_state=CacheState.MISSING)

    def send_foreground(self, plan) -> RuntimeResponse:
        self.last_plan = plan
        return RuntimeResponse(
            request_id=f"fg:{plan.query.query_id}",
            engine_ttft_ms=1.0,
            client_ttft_ms=1.0,
            prompt_tokens=1,
        )

    def send_background(self, page: VisualPage) -> RuntimeResponse:
        self.background_pages.append(page)
        raise AssertionError(f"unexpected background submission for {page.page_key}")


class SequenceRuntime(RuntimeClient):
    def __init__(self, selected_page_keys: list[str | None]) -> None:
        self.selected_page_keys = selected_page_keys
        self.select_index = 0
        self.foreground_plans = []

    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        selected = self.selected_page_keys[self.select_index]
        self.select_index += 1
        if selected is None:
            return CacheSelection(selected_page_key=None, kv_state=CacheState.MISSING)
        assert selected in page_keys
        return CacheSelection(
            selected_page_key=selected,
            lease_id=f"lease:{selected}",
            kv_state=CacheState.READY,
        )

    def send_foreground(self, plan) -> RuntimeResponse:
        self.foreground_plans.append(plan)
        return RuntimeResponse(
            request_id=f"fg:{plan.query.query_id}",
            engine_ttft_ms=1.0,
            client_ttft_ms=1.0,
            prompt_tokens=1,
        )

    def send_background(self, page: VisualPage) -> RuntimeResponse:
        raise AssertionError(f"unexpected background submission for {page.page_key}")


class CompositeSelectingRuntime(RuntimeClient):
    def __init__(self, *, seed_page_key: str) -> None:
        self.seed_page_key = seed_page_key
        self.select_index = 0
        self.foreground_plans = []

    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        self.select_index += 1
        if self.select_index == 1:
            assert self.seed_page_key in page_keys
            selected = self.seed_page_key
        else:
            selected = next(page_key for page_key in page_keys if page_key != self.seed_page_key)
        return CacheSelection(
            selected_page_key=selected,
            lease_id=f"lease:{selected}",
            kv_state=CacheState.READY,
        )

    def send_foreground(self, plan) -> RuntimeResponse:
        self.foreground_plans.append(plan)
        return RuntimeResponse(
            request_id=f"fg:{plan.query.query_id}",
            engine_ttft_ms=1.0,
            client_ttft_ms=1.0,
            prompt_tokens=1,
        )

    def send_background(self, page: VisualPage) -> RuntimeResponse:
        raise AssertionError(f"unexpected background submission for {page.page_key}")
