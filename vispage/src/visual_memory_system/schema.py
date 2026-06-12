"""Shared schemas for the visual memory system.

The schema layer is intentionally dataset-neutral. Dataset adapters may add
metadata, but core modules should only depend on these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


AccessType = Literal["gold", "actual", "profile", "retrieved", "provided"]
ExecutionPath = Literal["full_hit", "partial_hit", "fallback"]


class CacheState(str, Enum):
    MISSING = "missing"
    WARMING = "warming"
    READY = "ready"
    PARTIAL = "partial"
    EVICTED = "evicted"


@dataclass(frozen=True)
class MemoryUnit:
    unit_id: str
    text: str
    task_id: str | None = None
    source_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    step_id: int | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    task_id: str
    query_index: int
    query_text: str
    access_units: tuple[str, ...] = ()
    access_type: AccessType = "provided"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePage:
    """Temporary page proposal.

    Candidate pages are not runtime-visible. They may be submitted only after a
    foreground baseline-cold execution and only if admission accepts them.
    """

    candidate_id: str
    unit_ids: tuple[str, ...]
    anchor_query_id: str
    anchor_unit_ids: tuple[str, ...]
    predicted_coverage: float
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualPage:
    page_key: str
    unit_ids: tuple[str, ...]
    image_path: str | None = None
    prompt_tokens_estimate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegisteredPage:
    page: VisualPage
    state: CacheState = CacheState.WARMING
    submitted_at_query_index: int | None = None
    last_selected_query_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CacheSelection:
    selected_page_key: str | None
    lease_id: str | None = None
    kv_state: CacheState | None = None
    encoder_state: str | None = None
    inspected: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_ready_page(self) -> bool:
        return self.selected_page_key is not None and self.kv_state == CacheState.READY


@dataclass(frozen=True)
class ForegroundPlan:
    mode: Literal["warm_page", "baseline_cold"]
    query: QueryRecord
    retrieved_unit_ids: tuple[str, ...]
    input_pages: tuple[VisualPage, ...] = ()
    selected_page: VisualPage | None = None
    cache_page: VisualPage | None = None
    residual_unit_ids: tuple[str, ...] = ()
    lease_id: str | None = None
    coverage: float = 0.0


@dataclass(frozen=True)
class RuntimeResponse:
    request_id: str
    engine_ttft_ms: float
    client_ttft_ms: float | None = None
    server_ttft_ms: float | None = None
    pre_scheduler_delay_ms: float | None = None
    prompt_tokens: int | None = None
    output_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTraceRow:
    query_id: str
    query_index: int
    mode: Literal["warm_page", "baseline_cold"]
    retrieved_unit_count: int
    selected_page_key: str | None
    coverage: float
    residual_count: int
    submitted_page_keys: tuple[str, ...]
    engine_ttft_ms: float
    client_ttft_ms: float | None
    prompt_tokens: int | None
    server_ttft_ms: float | None = None
    ttft_ms: float | None = None
    pre_scheduler_delay_ms: float | None = None
    output_text: str | None = None
    execution_path: ExecutionPath = "fallback"
    evidence_coverage: float = 0.0
    read_amplification: float = 0.0
    carried_unit_count: int = 0
    selected_page_unit_count: int = 0
    registered_page_count: int = 0
    background_inflight_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def stable_page_key(prefix: str, unit_ids: tuple[str, ...], *, version: str = "v1") -> str:
    """Create a stable page key from canonical unit order."""

    import hashlib

    payload = "\n".join(unit_ids).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{prefix}:{version}:{digest}"


def coverage(required: tuple[str, ...] | list[str], available: tuple[str, ...] | list[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 0.0
    return len(required_set & set(available)) / len(required_set)
