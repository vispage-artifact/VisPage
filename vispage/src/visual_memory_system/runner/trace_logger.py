"""Run artifact writer."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from visual_memory_system.config import RunConfig
from visual_memory_system.runner.system_runner import RunResult


class TraceLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_run(
        self,
        *,
        config: RunConfig,
        result: RunResult,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.write_json("run_config.json", config.to_dict())
        if manifest is not None:
            self.write_json("manifest.json", manifest)
        self.write_jsonl("trace.jsonl", result.rows)
        self.write_jsonl("pages.jsonl", result.registered_pages.values())
        summary = summarize_run(result)
        self.write_json("summary.json", summary)
        return summary

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    def write_jsonl(self, name: str, rows: Any) -> None:
        path = self.output_dir / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True))
                f.write("\n")


def summarize_run(result: RunResult) -> dict[str, Any]:
    rows = result.rows
    if not rows:
        return {
            "queries": 0,
            "registered_pages": len(result.registered_pages),
            "background_error_count": len(result.background_errors),
        }
    baseline_cold = sum(1 for row in rows if row.mode == "baseline_cold")
    warm = sum(1 for row in rows if row.mode == "warm_page")
    full_hit = sum(1 for row in rows if row.execution_path == "full_hit")
    partial_hit = sum(1 for row in rows if row.execution_path == "partial_hit")
    fallback = sum(1 for row in rows if row.execution_path == "fallback")
    engine = [row.engine_ttft_ms for row in rows]
    server = [row.server_ttft_ms for row in rows if row.server_ttft_ms is not None]
    primary_ttft = [
        row.server_ttft_ms if row.server_ttft_ms is not None else row.engine_ttft_ms
        for row in rows
    ]
    residual = [row.residual_count for row in rows]
    evidence_coverage = [row.evidence_coverage for row in rows]
    read_amplification = [row.read_amplification for row in rows]
    submitted_page_keys = {
        page_key for row in rows for page_key in row.submitted_page_keys
    }
    used_page_keys = {
        row.selected_page_key for row in rows if row.selected_page_key is not None
    }
    wasted_page_count = len(submitted_page_keys - used_page_keys)
    summary = {
        "queries": len(rows),
        "baseline_cold_queries": baseline_cold,
        "warm_page_queries": warm,
        "full_hit_queries": full_hit,
        "partial_hit_queries": partial_hit,
        "fallback_queries": fallback,
        "registered_pages": len(result.registered_pages),
        "ttft_metric": (
            "server_request_to_first_token_ms"
            if len(server) == len(rows)
            else "mixed_server_request_fallback_engine_queue_to_first_token_ms"
            if server
            else "engine_queue_to_first_token_ms"
        ),
        "ttft_ms_mean": sum(primary_ttft) / len(primary_ttft),
        "ttft_ms_min": min(primary_ttft),
        "ttft_ms_max": max(primary_ttft),
        "engine_ttft_ms_mean": sum(engine) / len(engine),
        "engine_ttft_ms_min": min(engine),
        "engine_ttft_ms_max": max(engine),
        "residual_count_mean": sum(residual) / len(residual),
        "evidence_coverage_mean": sum(evidence_coverage) / len(evidence_coverage),
        "missing_evidence_queries": sum(1 for value in evidence_coverage if value < 1.0),
        "read_amplification_mean": sum(read_amplification) / len(read_amplification),
        "submitted_pages": sum(len(row.submitted_page_keys) for row in rows),
        "unique_submitted_pages": len(submitted_page_keys),
        "used_pages": len(used_page_keys),
        "warm_events": warm,
        "wasted_submitted_pages": wasted_page_count,
        "wasted_submitted_page_rate": (
            wasted_page_count / len(submitted_page_keys) if submitted_page_keys else None
        ),
        "background_error_count": len(result.background_errors),
    }
    if server:
        summary.update(
            {
                "server_ttft_ms_mean": sum(server) / len(server),
                "server_ttft_ms_min": min(server),
                "server_ttft_ms_max": max(server),
            }
        )
    return summary


def build_run_output_dir(output_root: str | Path, config: RunConfig, timestamp: str) -> Path:
    root = Path(output_root)
    return (
        root
        / _slug(config.run_name)
        / _dataset_segment(config)
        / _retrieval_segment(config)
        / _locality_segment(config)
        / _page_policy_segment(config)
        / _runtime_segment(config)
        / timestamp
    )


def build_run_manifest(
    *,
    config: RunConfig,
    config_path: str | Path,
    output_root: str | Path,
    output_dir: str | Path,
    timestamp: str,
    query_limit: int | None,
    git_commit: str | None,
) -> dict[str, Any]:
    output_root_path = Path(output_root)
    output_dir_path = Path(output_dir)
    run_id = "__".join(
        [
            _slug(config.run_name),
            _dataset_segment(config),
            _retrieval_segment(config),
            _locality_segment(config),
            _page_policy_segment(config),
            _runtime_segment(config),
            timestamp,
        ]
    )
    segments = {
        "run_name": _slug(config.run_name),
        "dataset": _dataset_segment(config),
        "retrieval": _retrieval_segment(config),
        "locality": _locality_segment(config),
        "page_policy": _page_policy_segment(config),
        "runtime": _runtime_segment(config),
        "timestamp": timestamp,
    }
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "config_path": str(config_path),
        "output_root": str(output_root_path),
        "output_dir": str(output_dir_path),
        "query_limit": query_limit,
        "segments": segments,
        "dataset": config.dataset,
        "retrieval": config.retrieval,
        "locality": config.locality,
        "page": config.page,
        "foreground": config.foreground,
        "prewarm": config.prewarm,
        "runtime": config.runtime,
        "artifacts": {
            "run_config": "run_config.json",
            "manifest": "manifest.json",
            "trace": "trace.jsonl",
            "pages": "pages.jsonl",
            "summary": "summary.json",
        },
    }


def append_run_index(
    output_root: str | Path,
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(str(manifest["output_dir"]))
    try:
        output_dir_rel = str(output_dir.relative_to(root))
    except ValueError:
        output_dir_rel = str(output_dir)
    row = {
        "run_id": manifest["run_id"],
        "timestamp": manifest["timestamp"],
        "git_commit": manifest["git_commit"],
        "config_path": manifest["config_path"],
        "output_dir": str(output_dir),
        "output_dir_rel": output_dir_rel,
        "segments": manifest["segments"],
        "query_limit": manifest["query_limit"],
        "dataset": _to_jsonable(manifest["dataset"]),
        "retrieval": _to_jsonable(manifest["retrieval"]),
        "locality": _to_jsonable(manifest["locality"]),
        "page": _to_jsonable(manifest["page"]),
        "foreground": _to_jsonable(manifest["foreground"]),
        "prewarm": _to_jsonable(manifest["prewarm"]),
        "runtime": _to_jsonable(manifest["runtime"]),
        "summary": summary,
        "artifacts": {
            name: str(output_dir / filename)
            for name, filename in manifest["artifacts"].items()
        },
    }
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _dataset_segment(config: RunConfig) -> str:
    if config.dataset.name not in {"standard_jsonl_dataset", "generic_jsonl_dataset"}:
        return _slug(config.dataset.name)
    inferred = _infer_dataset_name(config.dataset.memory_path)
    if inferred is not None:
        return _slug(inferred)
    return _slug(config.dataset.name)


def _infer_dataset_name(memory_path: str) -> str | None:
    parts = Path(memory_path).parts
    if "dataset" not in parts:
        return None
    dataset_index = parts.index("dataset")
    if dataset_index + 1 >= len(parts):
        return None
    candidate = parts[dataset_index + 1]
    if candidate in {"processed", "candidate"}:
        return None
    return candidate


def _retrieval_segment(config: RunConfig) -> str:
    topk = f"topk{config.retrieval.topk}"
    if config.retrieval.mode == "embedding_retrieve":
        embedding_name = "embedding"
        if config.retrieval.embedding_cache_path:
            embedding_name = Path(config.retrieval.embedding_cache_path).stem
        return f"embed_{_slug(embedding_name)}_{topk}"
    return f"{_slug(config.retrieval.mode)}_{topk}"


def _locality_segment(config: RunConfig) -> str:
    if config.locality.mode == "embedding_ball":
        if config.locality.radius_percentile == 100.0:
            max_units = _value(config.locality.max_units)
            return f"nearest_amp_max{max_units}"
        percentile = _value(config.locality.radius_percentile)
        max_units = _value(config.locality.max_units)
        return f"ball_p{percentile}_max{max_units}"
    if config.locality.mode == "random_anchor":
        max_units = _value(config.locality.max_units)
        return f"random_max{max_units}_seed{config.locality.random_seed}"
    return _slug(config.locality.mode)


def _page_policy_segment(config: RunConfig) -> str:
    return (
        f"page{_value(config.page.max_units)}"
        f"_amp{_value(config.page.max_amplification)}"
        f"_fgcov{_value(config.foreground.min_coverage)}"
        f"_precov{_value(config.prewarm.min_candidate_coverage)}"
        f"_overlap{_value(config.prewarm.max_new_in_old)}"
        f"_inflight{_value(config.prewarm.max_inflight_pages)}"
        f"_wscale{_value(config.renderer.width_scale)}"
        f"_pscale{_value(config.renderer.post_render_scale)}"
        f"_vtok{_value(config.page.max_visual_tokens)}"
        f"_cvtok{_value(config.page.max_cache_visual_tokens)}"
    )


def _runtime_segment(config: RunConfig) -> str:
    return (
        f"{_slug(config.runtime.model)}"
        f"_fg{config.runtime.foreground_max_tokens}"
        f"_bgchunk{config.runtime.background_chunk_tokens}"
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-")
    return slug or "none"


def _value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return _slug(str(value))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value
