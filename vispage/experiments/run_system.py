"""Run the full visual memory system from a JSON config."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import subprocess
import time

from visual_memory_system.config_loader import load_run_config
from visual_memory_system.experiments.factory import build_runner_from_config, load_queries_from_config
from visual_memory_system.runner.trace_logger import (
    TraceLogger,
    append_run_index,
    build_run_manifest,
    build_run_output_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config JSON.")
    parser.add_argument("--output-root", required=True, help="Directory for run artifacts.")
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed queries. Use 1 for every query; use 0 to disable.",
    )
    args = parser.parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = apply_run_tag(load_run_config(args.config), timestamp)
    runner = build_runner_from_config(config)
    queries = load_queries_from_config(config)
    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise ValueError("--query-limit must be positive")
        queries = queries[: args.query_limit]

    progress = (
        ProgressPrinter(total_queries=len(queries), every=args.progress_every)
        if args.progress_every
        else None
    )
    result = runner.run(queries, progress_callback=progress)
    output_root = Path(args.output_root)
    output_dir = build_run_output_dir(output_root, config, timestamp)
    manifest = build_run_manifest(
        config=config,
        config_path=args.config,
        output_root=output_root,
        output_dir=output_dir,
        timestamp=timestamp,
        query_limit=args.query_limit,
        git_commit=_git_commit(),
    )
    summary = TraceLogger(output_dir).write_run(
        config=config,
        result=result,
        manifest=manifest,
    )
    append_run_index(output_root, manifest=manifest, summary=summary)
    print(f"output_dir={output_dir}")
    print(f"queries={len(result.rows)} registered_pages={len(result.registered_pages)}")


class ProgressPrinter:
    def __init__(self, *, total_queries: int, every: int) -> None:
        if every <= 0:
            raise ValueError("progress interval must be positive")
        self.total_queries = total_queries
        self.every = every
        self.started_at = time.perf_counter()
        self.ttft_sum = 0.0
        self.warm_count = 0
        self.fallback_count = 0

    def __call__(self, completed: int, total: int, row) -> None:
        del total
        current_ttft = row.ttft_ms if row.ttft_ms is not None else row.engine_ttft_ms
        self.ttft_sum += current_ttft
        if row.mode == "warm_page":
            self.warm_count += 1
        else:
            self.fallback_count += 1
        if completed != self.total_queries and completed % self.every != 0:
            return
        elapsed = max(0.001, time.perf_counter() - self.started_at)
        pct = 100.0 * completed / max(1, self.total_queries)
        avg_ttft = self.ttft_sum / completed
        print(
            "progress "
            f"{completed}/{self.total_queries} ({pct:.1f}%) "
            f"current_query={row.query_id} "
            f"current_mode={row.mode} "
            f"current_page_coverage={row.coverage:.2f} "
            f"current_ttft_ms={current_ttft:.1f} "
            f"avg_ttft_ms={avg_ttft:.1f} "
            f"current_server_ttft_ms={_fmt_optional(row.server_ttft_ms)} "
            f"current_pre_scheduler_delay_ms={_fmt_optional(row.pre_scheduler_delay_ms)} "
            f"current_engine_ttft_ms={row.engine_ttft_ms:.1f} "
            f"amp={row.read_amplification:.2f} "
            f"warm={self.warm_count} "
            f"fallback={self.fallback_count} "
            f"registered_pages={row.registered_page_count} "
            f"inflight={row.background_inflight_count} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )


def _fmt_optional(value: float | None) -> str:
    return "na" if value is None else f"{value:.1f}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def apply_run_tag(config, timestamp: str):
    run_tag = f"{config.run_name}__{timestamp}"
    return replace(
        config,
        renderer=replace(
            config.renderer,
            run_tag=run_tag,
        ),
        page=replace(
            config.page,
            key_prefix=f"{config.page.key_prefix}:{run_tag}",
        ),
    )


if __name__ == "__main__":
    main()
