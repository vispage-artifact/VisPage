"""Configuration dataclasses for reproducible runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RetrievalMode = Literal["provided_access_units", "embedding_retrieve"]
LocalityMode = Literal["baseline", "append", "embedding_ball", "random_anchor", "oracle_future"]
RuntimeMode = Literal["vllm"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    memory_path: str
    query_path: str
    temporal_memory_mask: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode
    topk: int
    embedding_cache_path: str | None = None


@dataclass(frozen=True)
class LocalityConfig:
    mode: LocalityMode
    radius_percentile: float | None = None
    max_units: int | None = None
    random_seed: int = 0


@dataclass(frozen=True)
class PageConfig:
    key_prefix: str = "vmpage"
    max_units: int | None = None
    max_amplification: float | None = None
    max_foreground_update_amplification: float | None = None
    max_visual_tokens: int | None = None
    max_cache_visual_tokens: int | None = None


@dataclass(frozen=True)
class RendererConfig:
    output_dir: str
    unit_width: int = 1024
    width_scale: float = 1.0
    post_render_scale: float = 1.0
    font_size: int = 24
    line_height: int = 34
    padding: int = 24
    chars_per_line: int = 72
    tile_gap: int = 0
    run_tag: str | None = None


@dataclass(frozen=True)
class ForegroundConfig:
    min_coverage: float


@dataclass(frozen=True)
class PrewarmConfig:
    min_candidate_coverage: float
    max_new_in_old: float | None = None
    max_inflight_pages: int | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    mode: RuntimeMode
    base_url: str
    model: str
    api_key: str | None = None
    foreground_max_tokens: int = 64
    background_max_tokens: int = 1
    background_chunk_tokens: int = 256


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    locality: LocalityConfig
    renderer: RendererConfig
    runtime: RuntimeConfig
    page: PageConfig = field(default_factory=PageConfig)
    foreground: ForegroundConfig = field(default_factory=lambda: ForegroundConfig(min_coverage=0.5))
    prewarm: PrewarmConfig = field(
        default_factory=lambda: PrewarmConfig(min_candidate_coverage=0.5)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
