# Target Architecture

## High-level Pipeline

```text
Dataset Adapter
  -> Retriever
  -> Locality Estimator
  -> Page Constructor
  -> Renderer / Page Store
  -> Runtime Scheduler / vLLM Client
  -> Trace Logger
  -> Evaluator
```

The pipeline should be configured by experiment files, but the modules
themselves should not know which experiment is calling them.

## Page Lifecycle Semantics

The runtime system should use a strict page lifecycle. This avoids mixing
temporary locality candidates with pages that the runtime can actually reuse.

```text
Baseline-cold-triggered proposal
  -> CandidatePage
  -> Admission decision
  -> SubmittedPage
  -> RegisteredPage
  -> Runtime-selected ReadyPage
```

### CandidatePage

A `CandidatePage` is a temporary page proposal generated after the current
foreground query takes the regular `baseline_cold` demand path.

Important semantics:

- Candidate pages are generated only after `baseline_cold`.
- A warm hit should not generate a new page by default.
- Candidate pages are not visible to future foreground selection.
- Candidate pages do not enter the runtime page registry.
- Candidate pages can be logged for analysis, but they are not reusable pages.

This keeps the system from creating speculative pages for every query and from
polluting future selection with pages that were never materialized.

### SubmittedPage

A `SubmittedPage` is a candidate that passed admission control and has been sent
to vLLM as a background prewarm request.

Admission may reject a candidate because:

- it has too little expected coverage;
- it is too similar to existing registered pages;
- it exceeds page cost or amplification limits;
- there are already too many inflight background requests;
- it is unlikely to be useful for future queries.

### RegisteredPage

A `RegisteredPage` is a submitted page that is known to the agent-side page
registry. Only registered pages can be considered by later foreground queries.

The registry may contain pages in different runtime states:

```text
warming
ready
partial
evicted
missing
```

However, the registry only tracks pages that were actually submitted. A local
candidate that was never submitted must not appear as a registered page.

### ReadyPage

A `ReadyPage` is not decided by the agent alone. At query time, the agent sends
registered candidate keys whose predicted coverage is high enough to vLLM's
cache selection endpoint. vLLM then selects a page that is actually ready under
the runtime cache state.

```text
registered pages with coverage > threshold
  -> /v1/am/cache/select
  -> selected ready page or no selection
```

Foreground execution may use a selected ready page. If vLLM does not select a
ready page, the query falls back to the baseline retrieval path.

### Policy Summary

The intended runtime policy is:

```text
1. For each query, retrieve top-k evidence.
2. Check existing RegisteredPages for sufficient coverage.
3. Ask vLLM to select one truly ready page from those registered pages.
4. If a ready page is selected, use it plus optional residual evidence.
5. If no ready page is selected, use the regular `baseline_cold` path.
6. Only after `baseline_cold`, generate new CandidatePages from the current evidence.
7. Submit only candidates that pass admission control.
8. Register only submitted pages.
```

This means the runtime-visible page set is exactly the registered page set. The
system must never assume an unsubmitted candidate can be reused later.

## Module Layout

Planned source tree:

```text
src/visual_memory_system/
  schema.py
  data/
    base.py
    generic_jsonl.py
    locomo.py
  retrieval/
    base.py
    embedding.py
    provided.py
  locality/
    base.py
    embedding_ball.py
    random_anchor.py
    oracle_future.py
  page/
    builder.py
    renderer.py
    store.py
  runtime/
    client.py
    prompt.py
    cache_registry.py
  policies/
    foreground.py
    prewarm.py
    admission.py
  runner/
    system_runner.py
    trace_logger.py
  evaluation/
    metrics.py
    quality.py

experiments/
  exp1_end_to_end.py
  exp2_locality.py
  exp3_cache_lifecycle.py
  exp4_scheduler_materialization.py

configs/
  locomo_exp1.yaml
  generic_jsonl_template.yaml

scripts/
  run_exp1.sh
```

## Core Interfaces

### DataAdapter

```python
class DataAdapter:
    def load_memory_units(self) -> list[MemoryUnit]:
        ...

    def load_queries(self) -> list[QueryRecord]:
        ...
```

Responsibilities:

- Convert dataset-specific files into normalized schemas.
- Validate required fields.
- Attach metadata without making policy decisions.

It should not:

- Perform retrieval.
- Construct pages.
- Call vLLM.

### Retriever

```python
class Retriever:
    def retrieve(
        self,
        query: QueryRecord,
        memory_units: list[MemoryUnit],
        topk: int,
    ) -> list[str]:
        ...
```

Responsibilities:

- Produce the demand evidence set for the current query.
- Record retrieval metadata and scores.

Allowed implementations:

- `ProvidedRetriever`
- `EmbeddingRetriever`
- `OracleFutureRetriever` only for explicitly marked oracle experiments.

### LocalityEstimator

```python
class LocalityEstimator:
    def propose_pages(
        self,
        query: QueryRecord,
        retrieved_unit_ids: list[str],
        history: RuntimeHistory,
    ) -> list[PageCandidate]:
        ...
```

Responsibilities:

- Convert anchors into candidate memory regions/pages.
- Estimate coverage and cost.
- Produce stable candidate IDs before rendering.
- Produce proposals only when the foreground policy asks for baseline-cold-triggered
  page construction.

Examples:

- embedding ball around retrieved anchors;
- random anchor ablation;
- future oracle region for upper-bound analysis.

### PageConstructor

```python
class PageConstructor:
    def build(self, candidate: PageCandidate) -> VisualPage:
        ...
```

Responsibilities:

- Enforce page budget and amplification constraints.
- Order memory units canonically.
- Create stable page keys.

### Renderer

```python
class Renderer:
    def render_unit(self, unit: MemoryUnit) -> RenderedUnit:
        ...

    def render_page(self, page: VisualPage) -> RenderedPage:
        ...
```

Responsibilities:

- Render each memory unit with stable dimensions.
- Compose pages from unit renderings.
- Avoid hidden compression differences between baseline, page, and residual.

### RuntimeClient

```python
class RuntimeClient:
    def select_ready_page(self, candidates: list[str]) -> CacheSelection:
        ...

    def send_foreground(self, request: ForegroundRequest) -> RuntimeResponse:
        ...

    def send_background(self, request: BackgroundPrewarmRequest) -> RuntimeResponse:
        ...
```

Responsibilities:

- Hide HTTP details.
- Parse engine TTFT and client TTFT.
- Pass AM role, page key, and lease metadata to vLLM.

### ForegroundPolicy

```python
class ForegroundPolicy:
    def plan(
        self,
        retrieved_unit_ids: list[str],
        ready_pages: list[VisualPage],
    ) -> ForegroundPlan:
        ...
```

Responsibilities:

- Choose warm page if coverage is sufficient.
- Add residual units if configured.
- Use the regular baseline-cold path when no page is ready or coverage is too low.

Important constraint:

- Current vLLM prefix cache works best when the foreground uses one warm page as
  a contiguous prefix. Multiple independently warmed pages should not be assumed
  to compose into a full prefix hit.

### PrewarmPolicy

```python
class PrewarmPolicy:
    def admit(
        self,
        candidates: list[PageCandidate],
        cache_state: CacheState,
        history: RuntimeHistory,
    ) -> list[BackgroundPrewarmRequest]:
        ...
```

Responsibilities:

- Decide whether to submit candidate pages.
- Avoid redundant pages with high overlap.
- Limit background pressure.
- Register a page only after it is submitted to runtime prewarm.

### SystemRunner

```python
class SystemRunner:
    def run(self, queries: list[QueryRecord]) -> RunResult:
        ...
```

Responsibilities:

- Enforce strict serial demand query execution.
- Launch background prewarm opportunistically.
- Write trace rows and summaries.

## Experiment Boundaries

Experiments should only configure:

- dataset adapter;
- retrieval mode;
- locality estimator;
- page budget;
- runtime endpoint;
- foreground/prewarm policies;
- metrics.

They should not contain custom rendering, retrieval, or vLLM request logic.

## Required Run Artifacts

Each run should write:

```text
run_config.json
summary.json
trace.jsonl
pages.jsonl
requests.jsonl
quality/answers.jsonl optional
quality/judge_results.jsonl optional
```

The config must include:

- dataset name and split;
- retrieval mode and top-k;
- embedding cache metadata;
- renderer scale and layout parameters;
- locality estimator parameters;
- amplification/page constraints;
- runtime endpoint and model name;
- vLLM AM feature flags if known.

## Failure Policy

The refactored system should fail explicitly when:

- retrieval mode is missing;
- embedding cache is missing for embedding retrieval;
- top-k retrieval returns fewer than expected units without a documented reason;
- renderer cannot produce a stable page key;
- runtime response lacks required engine TTFT metrics in latency experiments;
- oracle methods are used outside oracle-labeled runs.
