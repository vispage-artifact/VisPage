"""Build system components from RunConfig."""

from __future__ import annotations

from visual_memory_system.config import RunConfig
from visual_memory_system.data.embeddings import load_embedding_cache
from visual_memory_system.data.generic_jsonl import GenericJsonlAdapter
from visual_memory_system.locality import (
    AppendEstimator,
    BaselineEstimator,
    EmbeddingBallEstimator,
    RandomAnchorEstimator,
)
from visual_memory_system.page.builder import PageBuilder
from visual_memory_system.page.renderer import PillowPageRenderer
from visual_memory_system.policies.foreground import ForegroundPolicy
from visual_memory_system.policies.prewarm import PrewarmPolicy
from visual_memory_system.retrieval.provided import ProvidedRetriever
from visual_memory_system.retrieval.embedding import EmbeddingRetriever
from visual_memory_system.runner.system_runner import SystemRunner
from visual_memory_system.runtime.vllm import VllmRuntimeClient


def build_runner_from_config(config: RunConfig) -> SystemRunner:
    adapter = GenericJsonlAdapter(config.dataset.memory_path, config.dataset.query_path)
    memory_units = adapter.load_memory_units()
    page_max_units = _effective_page_max_units(config)
    memory_embeddings = query_embeddings = None
    if config.retrieval.mode == "embedding_retrieve" or config.locality.mode == "embedding_ball":
        if not config.retrieval.embedding_cache_path:
            raise ValueError("embedding cache path is required for embedding retrieval/locality")
        memory_embeddings, query_embeddings = load_embedding_cache(config.retrieval.embedding_cache_path)

    if config.retrieval.mode == "provided_access_units":
        retriever = ProvidedRetriever()
    elif config.retrieval.mode == "embedding_retrieve":
        if memory_embeddings is None or query_embeddings is None:
            raise RuntimeError("embedding cache was not loaded")
        retriever = EmbeddingRetriever(memory_embeddings, query_embeddings)
    else:
        raise ValueError(f"unsupported retrieval mode {config.retrieval.mode!r}")

    if config.locality.mode == "baseline":
        locality_estimator = BaselineEstimator()
    elif config.locality.mode == "append":
        locality_estimator = AppendEstimator()
    elif config.locality.mode == "embedding_ball":
        if memory_embeddings is None:
            raise RuntimeError("memory embeddings were not loaded")
        if config.locality.radius_percentile is None:
            raise ValueError("embedding_ball requires locality.radius_percentile")
        if config.locality.max_units is None:
            raise ValueError("embedding_ball requires locality.max_units")
        locality_estimator = EmbeddingBallEstimator(
            memory_embeddings,
            radius_percentile=config.locality.radius_percentile,
            max_units=_bounded_locality_units(config.locality.max_units, page_max_units),
        )
    elif config.locality.mode == "random_anchor":
        if config.locality.max_units is None:
            raise ValueError("random_anchor requires locality.max_units")
        locality_estimator = RandomAnchorEstimator(
            page_units=_bounded_locality_units(config.locality.max_units, page_max_units),
            seed=config.locality.random_seed,
        )
    else:
        raise NotImplementedError(f"locality mode {config.locality.mode} is not implemented")

    if config.runtime.mode != "vllm":
        raise ValueError(f"unsupported runtime mode {config.runtime.mode!r}")

    return SystemRunner(
        memory_units=memory_units,
        retriever=retriever,
        locality_estimator=locality_estimator,
        page_builder=PageBuilder(key_prefix=config.page.key_prefix, max_units=page_max_units),
        renderer=PillowPageRenderer(
            output_dir=config.renderer.output_dir,
            unit_width=config.renderer.unit_width,
            width_scale=config.renderer.width_scale,
            post_render_scale=config.renderer.post_render_scale,
            font_size=config.renderer.font_size,
            line_height=config.renderer.line_height,
            padding=config.renderer.padding,
            chars_per_line=config.renderer.chars_per_line,
            tile_gap=config.renderer.tile_gap,
            run_tag=config.renderer.run_tag,
        ),
        foreground_policy=ForegroundPolicy(min_coverage=config.foreground.min_coverage),
        prewarm_policy=PrewarmPolicy(
            min_candidate_coverage=config.prewarm.min_candidate_coverage,
            max_new_in_old=config.prewarm.max_new_in_old,
            max_inflight_pages=config.prewarm.max_inflight_pages,
        ),
        runtime_client=VllmRuntimeClient(
            base_url=config.runtime.base_url,
            model=config.runtime.model,
            api_key=config.runtime.api_key,
            foreground_max_tokens=config.runtime.foreground_max_tokens,
            background_max_tokens=config.runtime.background_max_tokens,
            background_chunk_tokens=config.runtime.background_chunk_tokens,
        ),
        topk=config.retrieval.topk,
        append_foreground_pages=config.locality.mode == "append",
        disable_page_registration=config.locality.mode == "baseline",
        register_foreground_residual_pages=config.locality.mode != "random_anchor",
        max_foreground_update_amplification=config.page.max_foreground_update_amplification,
        max_visual_tokens=config.page.max_visual_tokens,
        max_cache_visual_tokens=config.page.max_cache_visual_tokens,
        temporal_memory_mask=config.dataset.temporal_memory_mask,
    )


def load_queries_from_config(config: RunConfig):
    adapter = GenericJsonlAdapter(config.dataset.memory_path, config.dataset.query_path)
    return adapter.load_queries()


def _effective_page_max_units(config: RunConfig) -> int | None:
    limits: list[int] = []
    if config.page.max_units is not None:
        if config.page.max_units <= 0:
            raise ValueError("page.max_units must be positive")
        limits.append(config.page.max_units)
    if config.page.max_amplification is not None:
        if config.page.max_amplification <= 0:
            raise ValueError("page.max_amplification must be positive")
        amplification_limit = int(config.retrieval.topk * config.page.max_amplification)
        if amplification_limit < 1:
            raise ValueError(
                "page.max_amplification is too small for retrieval.topk; "
                "effective page max units would be below 1"
            )
        limits.append(amplification_limit)
    if (
        config.page.max_foreground_update_amplification is not None
        and config.page.max_foreground_update_amplification <= 0
    ):
        raise ValueError("page.max_foreground_update_amplification must be positive")
    return min(limits) if limits else None


def _bounded_locality_units(locality_max_units: int, page_max_units: int | None) -> int:
    if locality_max_units <= 0:
        raise ValueError("locality.max_units must be positive")
    if page_max_units is None:
        return locality_max_units
    return min(locality_max_units, page_max_units)
