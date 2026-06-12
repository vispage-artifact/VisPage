#!/usr/bin/env python3
"""Offline workload locality profiler.

The profiler treats each query's retrieved top-k memory units as the system
observed working set, then compares unit-level temporal reuse against
embedding-space semantic locality.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from visual_memory_system.data.embeddings import load_embedding_cache
from visual_memory_system.data.generic_jsonl import GenericJsonlAdapter
from visual_memory_system.retrieval.embedding import EmbeddingRetriever
from visual_memory_system.schema import MemoryUnit, QueryRecord

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy is expected in experiment envs.
    raise RuntimeError("numpy is required for locality profiling") from exc


@dataclass(frozen=True)
class RetrievedQuery:
    position: int
    query: QueryRecord
    unit_ids: tuple[str, ...]
    unit_set: frozenset[str]
    centroid: Any
    cluster_id: int | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--near-windows", default="1,3,5,10")
    parser.add_argument("--recent-windows", default="5,10,20")
    parser.add_argument("--far-gap", type=int, default=10)
    parser.add_argument("--bin-size", type=int, default=50)
    parser.add_argument("--cluster-threshold", type=float, default=0.82)
    parser.add_argument("--high-sem-threshold", type=float, default=0.82)
    parser.add_argument("--low-unit-threshold", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    near_windows = _parse_int_list(args.near_windows)
    recent_windows = _parse_int_list(args.recent_windows)
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.far_gap < 0:
        raise ValueError("--far-gap must be non-negative")
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive")

    adapter = GenericJsonlAdapter(args.memory, args.queries)
    memory_units = adapter.load_memory_units()
    queries = adapter.load_queries()
    if args.limit is not None:
        queries = queries[: args.limit]
    memory_embeddings, query_embeddings = load_embedding_cache(args.embeddings)
    retriever = EmbeddingRetriever(memory_embeddings, query_embeddings)
    memory_matrix = _normalized_memory_matrix(memory_units, memory_embeddings)
    retrieved = _retrieve_queries(
        queries=queries,
        memory_units=memory_units,
        retriever=retriever,
        topk=args.topk,
        memory_matrix=memory_matrix,
        cluster_threshold=args.cluster_threshold,
    )

    per_query = _per_query_metrics(
        retrieved,
        topk=args.topk,
        near_windows=near_windows,
        recent_windows=recent_windows,
        far_gap=args.far_gap,
        high_sem_threshold=args.high_sem_threshold,
        low_unit_threshold=args.low_unit_threshold,
    )
    bins = _bin_metrics(per_query, bin_size=args.bin_size)
    summary = _summary(
        retrieved=retrieved,
        per_query=per_query,
        bins=bins,
        args=args,
        near_windows=near_windows,
        recent_windows=recent_windows,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "summary.json", summary)
    _write_jsonl(args.output_dir / "per_query.jsonl", per_query)
    _write_jsonl(args.output_dir / "bins.jsonl", bins)
    _write_markdown(args.output_dir / "summary.md", summary, bins)
    print(json.dumps(summary["headline"], ensure_ascii=False, indent=2, sort_keys=True))


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"invalid positive integer list: {value!r}")
    return sorted(set(parsed))


def _retrieve_queries(
    *,
    queries: list[QueryRecord],
    memory_units: list[MemoryUnit],
    retriever: EmbeddingRetriever,
    topk: int,
    memory_matrix: dict[str, Any],
    cluster_threshold: float,
) -> list[RetrievedQuery]:
    result: list[RetrievedQuery] = []
    cluster_centroids: list[Any] = []
    for position, query in enumerate(queries, start=1):
        unit_ids = tuple(retriever.retrieve(query, memory_units, topk))
        centroid = _normalize(np.mean([memory_matrix[unit_id] for unit_id in unit_ids], axis=0))
        cluster_id = _assign_cluster(centroid, cluster_centroids, cluster_threshold)
        result.append(
            RetrievedQuery(
                position=position,
                query=query,
                unit_ids=unit_ids,
                unit_set=frozenset(unit_ids),
                centroid=centroid,
                cluster_id=cluster_id,
            )
        )
    return result


def _normalized_memory_matrix(
    memory_units: list[MemoryUnit],
    embeddings: dict[str, list[float]],
) -> dict[str, Any]:
    result = {}
    for unit in memory_units:
        if unit.unit_id not in embeddings:
            raise KeyError(f"missing embedding for memory unit {unit.unit_id}")
        result[unit.unit_id] = _normalize(np.asarray(embeddings[unit.unit_id], dtype=np.float32))
    return result


def _assign_cluster(centroid: Any, cluster_centroids: list[Any], threshold: float) -> int:
    if not cluster_centroids:
        cluster_centroids.append(centroid)
        return 0
    sims = [float(centroid @ cluster) for cluster in cluster_centroids]
    best_index = max(range(len(sims)), key=lambda index: sims[index])
    if sims[best_index] >= threshold:
        cluster_centroids[best_index] = _normalize((cluster_centroids[best_index] + centroid) / 2)
        return best_index
    cluster_centroids.append(centroid)
    return len(cluster_centroids) - 1


def _per_query_metrics(
    retrieved: list[RetrievedQuery],
    *,
    topk: int,
    near_windows: list[int],
    recent_windows: list[int],
    far_gap: int,
    high_sem_threshold: float,
    low_unit_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_position_by_cluster: dict[int, int] = {}
    for index, item in enumerate(retrieved):
        previous = retrieved[:index]
        row: dict[str, Any] = {
            "position": item.position,
            "query_id": item.query.query_id,
            "task_id": item.query.task_id,
            "query_index": item.query.query_index,
            "cluster_id": item.cluster_id,
            "retrieved_unit_ids": item.unit_ids,
        }

        overlap_1 = 0.0
        sim_1 = None
        if previous:
            prev = previous[-1]
            overlap_1 = _overlap(item.unit_set, prev.unit_set, topk)
            sim_1 = _sim(item.centroid, prev.centroid)
        row["unit_overlap_at_1"] = overlap_1
        row["centroid_sim_at_1"] = sim_1
        row["high_sem_low_unit_at_1"] = (
            bool(sim_1 is not None and sim_1 >= high_sem_threshold and overlap_1 <= low_unit_threshold)
        )

        for window in near_windows:
            near = previous[-window:]
            row[f"unit_overlap_max_at_{window}"] = (
                max((_overlap(item.unit_set, other.unit_set, topk) for other in near), default=0.0)
            )
            row[f"centroid_sim_max_at_{window}"] = (
                max((_sim(item.centroid, other.centroid) for other in near), default=None)
            )

        for window in recent_windows:
            recent = previous[-window:]
            recent_units = _union_units(recent)
            row[f"recent_unit_coverage_at_{window}"] = _overlap(item.unit_set, recent_units, topk)
            row[f"recent_unique_units_at_{window}"] = len(recent_units)
            row[f"recent_unique_ratio_at_{window}"] = len(recent_units) / topk

        far = previous[: max(0, len(previous) - far_gap)]
        row[f"far_centroid_sim_gap_{far_gap}"] = (
            max((_sim(item.centroid, other.centroid) for other in far), default=None)
        )
        row[f"far_unit_overlap_gap_{far_gap}"] = (
            max((_overlap(item.unit_set, other.unit_set, topk) for other in far), default=0.0)
        )

        if item.cluster_id is not None and item.cluster_id in last_position_by_cluster:
            gap = item.position - last_position_by_cluster[item.cluster_id]
        else:
            gap = None
        row["cluster_revisit_gap"] = gap
        row[f"cluster_revisit_gap_gt_{far_gap}"] = bool(gap is not None and gap > far_gap)
        if item.cluster_id is not None:
            last_position_by_cluster[item.cluster_id] = item.position

        rows.append(row)
    return rows


def _bin_metrics(per_query: list[dict[str, Any]], *, bin_size: int) -> list[dict[str, Any]]:
    bins = []
    for start in range(0, len(per_query), bin_size):
        rows = per_query[start : start + bin_size]
        end_position = rows[-1]["position"]
        bins.append(_aggregate_rows(rows, start_position=rows[0]["position"], end_position=end_position))
    return bins


def _summary(
    *,
    retrieved: list[RetrievedQuery],
    per_query: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    args: argparse.Namespace,
    near_windows: list[int],
    recent_windows: list[int],
) -> dict[str, Any]:
    all_rows = _aggregate_rows(per_query, start_position=1, end_position=len(per_query))
    task_counts = Counter(item.query.task_id for item in retrieved)
    cluster_counts = Counter(item.cluster_id for item in retrieved)
    temporal_key = "recent_unit_coverage_at_10" if 10 in recent_windows else f"recent_unit_coverage_at_{recent_windows[-1]}"
    semantic_key = "centroid_sim_max_at_5" if 5 in near_windows else f"centroid_sim_max_at_{near_windows[-1]}"
    high_sem_low = all_rows.get("high_sem_low_unit_at_1_rate", 0.0)
    headline = {
        "dataset": args.dataset_name,
        "queries": len(per_query),
        "topk": args.topk,
        "unit_temporal_score": all_rows.get(f"{temporal_key}_mean"),
        "near_semantic_score": all_rows.get(f"{semantic_key}_mean"),
        "high_sem_low_unit_rate_at_1": high_sem_low,
        "append_pressure_recent_unique_ratio_at_10": all_rows.get("recent_unique_ratio_at_10_mean"),
        "cluster_count": len(cluster_counts),
    }
    return {
        "headline": headline,
        "config": {
            "memory": str(args.memory),
            "queries": str(args.queries),
            "embeddings": str(args.embeddings),
            "topk": args.topk,
            "near_windows": near_windows,
            "recent_windows": recent_windows,
            "far_gap": args.far_gap,
            "bin_size": args.bin_size,
            "cluster_threshold": args.cluster_threshold,
            "high_sem_threshold": args.high_sem_threshold,
            "low_unit_threshold": args.low_unit_threshold,
            "limit": args.limit,
        },
        "tasks": {
            "count": len(task_counts),
            "query_counts": dict(sorted(task_counts.items())),
        },
        "clusters": {
            "count": len(cluster_counts),
            "top_counts": cluster_counts.most_common(20),
        },
        "overall": all_rows,
        "bins": bins,
    }


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    start_position: int,
    end_position: int,
) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    bool_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, bool)
        }
    )
    result: dict[str, Any] = {
        "start_position": start_position,
        "end_position": end_position,
        "queries": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "cluster_count": len({row["cluster_id"] for row in rows}),
    }
    for key in numeric_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        result[f"{key}_mean"] = statistics.fmean(values)
        result[f"{key}_p50"] = _percentile(values, 0.5)
        result[f"{key}_p90"] = _percentile(values, 0.9)
    for key in bool_keys:
        values = [bool(row.get(key)) for row in rows]
        result[f"{key}_rate"] = sum(values) / len(values) if values else 0.0
    return result


def _union_units(items: list[RetrievedQuery]) -> frozenset[str]:
    unit_ids: set[str] = set()
    for item in items:
        unit_ids.update(item.unit_set)
    return frozenset(unit_ids)


def _overlap(left: frozenset[str], right: frozenset[str], topk: int) -> float:
    if not left:
        return 0.0
    return len(left & right) / topk


def _sim(left: Any, right: Any) -> float:
    return float(left @ right)


def _normalize(vector: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("zero-norm vector")
    return vector / norm


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_markdown(path: Path, summary: dict[str, Any], bins: list[dict[str, Any]]) -> None:
    headline = summary["headline"]
    overall = summary["overall"]
    lines = [
        "# Locality Profile",
        "",
        f"- dataset: {headline.get('dataset')}",
        f"- queries: {headline.get('queries')}",
        f"- topk: {headline.get('topk')}",
        f"- temporal score: {_fmt(headline.get('unit_temporal_score'))}",
        f"- near semantic score: {_fmt(headline.get('near_semantic_score'))}",
        f"- high semantic / low unit rate: {_fmt(headline.get('high_sem_low_unit_rate_at_1'))}",
        f"- append pressure recent unique ratio: {_fmt(headline.get('append_pressure_recent_unique_ratio_at_10'))}",
        f"- cluster count: {headline.get('cluster_count')}",
        "",
        "## Overall",
        "",
        f"- unit_overlap@1 mean: {_fmt(overall.get('unit_overlap_at_1_mean'))}",
        f"- centroid_sim@1 mean: {_fmt(overall.get('centroid_sim_at_1_mean'))}",
        f"- recent_unit_coverage@10 mean: {_fmt(overall.get('recent_unit_coverage_at_10_mean'))}",
        f"- recent_unique_ratio@10 mean: {_fmt(overall.get('recent_unique_ratio_at_10_mean'))}",
        f"- far_centroid_sim gap mean: {_fmt(_first_key(overall, 'far_centroid_sim_gap_', '_mean'))}",
        "",
        "## Bins",
        "",
        "| positions | unit@1 | sem@1 | recent@10 | unique@10 | high-sem-low-unit | clusters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bins:
        lines.append(
            "| "
            f"{row['start_position']}-{row['end_position']} | "
            f"{_fmt(row.get('unit_overlap_at_1_mean'))} | "
            f"{_fmt(row.get('centroid_sim_at_1_mean'))} | "
            f"{_fmt(row.get('recent_unit_coverage_at_10_mean'))} | "
            f"{_fmt(row.get('recent_unique_ratio_at_10_mean'))} | "
            f"{_fmt(row.get('high_sem_low_unit_at_1_rate'))} | "
            f"{row.get('cluster_count')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_key(row: dict[str, Any], prefix: str, suffix: str) -> Any:
    for key in sorted(row):
        if key.startswith(prefix) and key.endswith(suffix):
            return row[key]
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
