"""Session-level auto evaluation runner.

This script is intentionally a glue layer: each session writes temporary
normalized JSONL inputs and then calls the existing config loader, factory,
runner, and trace logger stack.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from visual_memory_system.config import RunConfig
from visual_memory_system.config_loader import load_run_config
from visual_memory_system.data.generic_jsonl import GenericJsonlAdapter
from visual_memory_system.data.embeddings import load_embedding_cache
from visual_memory_system.experiments.factory import build_runner_from_config
from visual_memory_system.runner.trace_logger import (
    TraceLogger,
    build_run_manifest,
)
from visual_memory_system.schema import MemoryUnit, QueryRecord


def main() -> None:
    args = _parse_args()
    batch_dir, config_paths = _resolve_batch(args)
    batch_dir.mkdir(parents=True, exist_ok=True)
    _write_batch_manifest(batch_dir, args=args, config_paths=config_paths)

    if args.schedule == "session":
        failures = run_configs_by_session(config_paths=config_paths, batch_dir=batch_dir, args=args)
    else:
        failures = run_configs_by_config(config_paths=config_paths, batch_dir=batch_dir, args=args)

    if failures:
        _write_json(batch_dir / "failures.json", {"failures": failures})
    elif args.plan_only:
        _write_json(
            batch_dir / "plan_completed.json",
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "config_count": len(config_paths),
            },
        )
    else:
        _write_json(
            batch_dir / "completed.json",
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "config_count": len(config_paths),
            },
        )
    print(f"batch_dir={batch_dir}", flush=True)


def run_configs_by_config(*, config_paths: list[Path], batch_dir: Path, args) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for config_order, config_path in enumerate(config_paths):
        try:
            completed = run_config(
                config_order=config_order,
                config_path=config_path,
                batch_dir=batch_dir,
                args=args,
            )
            if not completed:
                failures.append(
                    {
                        "config_order": config_order,
                        "config_path": str(config_path),
                        "error": "config has incomplete or failed sessions",
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                )
        except Exception as exc:
            failure = {
                "config_order": config_order,
                "config_path": str(config_path),
                "error": repr(exc),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            failures.append(failure)
            _write_json(batch_dir / "last_failure.json", failure)
            print(f"config failed order={config_order} path={config_path}: {exc}", flush=True)
            if args.stop_on_failure:
                raise
    return failures


def run_configs_by_session(*, config_paths: list[Path], batch_dir: Path, args) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for config_order, config_path in enumerate(config_paths):
        try:
            state = prepare_config_run(
                config_order=config_order,
                config_path=config_path,
                batch_dir=batch_dir,
                args=args,
            )
            if state["status"] == "failed":
                failures.append(
                    {
                        "config_order": config_order,
                        "config_path": str(config_path),
                        "error": "config preflight failed",
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                )
            elif state["status"] == "ready":
                states.append(state)
        except Exception as exc:
            failure = {
                "config_order": config_order,
                "config_path": str(config_path),
                "error": repr(exc),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            failures.append(failure)
            _write_json(batch_dir / "last_failure.json", failure)
            print(f"config failed order={config_order} path={config_path}: {exc}", flush=True)
            if args.stop_on_failure:
                raise

    if args.plan_only or not states:
        return failures

    _validate_interleaved_sessions(states)
    failed_by_config: dict[int, list[dict[str, Any]]] = {
        int(state["config_order"]): [] for state in states
    }
    session_count = len(states[0]["sessions"])
    for session_index in range(1, session_count + 1):
        task_id = states[0]["sessions"][session_index - 1].task_id
        print(
            f"interleaved_session {session_index}/{session_count} task_id={task_id} "
            f"configs={len(states)}",
            flush=True,
        )
        for state in states:
            config_order = int(state["config_order"])
            config_path = state["config_path"]
            session = state["sessions"][session_index - 1]
            try:
                run_session(
                    config=state["config"],
                    config_dir=state["config_dir"],
                    root_run_tag=state["root_run_tag"],
                    session=session,
                    session_index=session_index,
                    session_count=session_count,
                    args=args,
                )
            except Exception as exc:
                failure = {
                    "session_id": session.task_id,
                    "session_index": session_index,
                    "error": repr(exc),
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                failed_by_config[config_order].append(failure)
                _write_json(state["config_dir"] / "last_failure.json", failure)
                print(
                    f"session failed config_order={config_order} "
                    f"path={config_path} task_id={session.task_id}: {exc}",
                    flush=True,
                )
                if args.stop_on_failure:
                    raise

    for state in states:
        config_order = int(state["config_order"])
        config_dir = state["config_dir"]
        aggregate = write_config_aggregate(config_dir=config_dir, sessions=state["sessions"])
        config_failures = failed_by_config[config_order]
        if config_failures:
            _write_json(config_dir / "failures.json", {"failures": config_failures})
            failures.append(
                {
                    "config_order": config_order,
                    "config_path": str(state["config_path"]),
                    "error": "config has incomplete or failed sessions",
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
            )
        else:
            _unlink_if_exists(config_dir / "failures.json")
            _unlink_if_exists(config_dir / "last_failure.json")
            _write_json(
                config_dir / "completed.json",
                {
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "sessions": len(state["sessions"]),
                    "queries": aggregate["queries"],
                    "aggregate_summary": "aggregate_summary.json",
                },
            )
    return failures


def run_config(*, config_order: int, config_path: Path, batch_dir: Path, args) -> bool:
    state = prepare_config_run(
        config_order=config_order,
        config_path=config_path,
        batch_dir=batch_dir,
        args=args,
    )
    if state["status"] == "completed":
        return True
    if state["status"] == "failed":
        return {
            "status": "failed",
            "config_order": config_order,
            "config_path": config_path,
        }
    if args.plan_only:
        return True

    config = state["config"]
    config_dir = state["config_dir"]
    root_run_tag = state["root_run_tag"]
    sessions = state["sessions"]

    failed_sessions: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions, start=1):
        try:
            run_session(
                config=config,
                config_dir=config_dir,
                root_run_tag=root_run_tag,
                session=session,
                session_index=session_index,
                session_count=len(sessions),
                args=args,
            )
        except Exception as exc:
            failure = {
                "session_id": session.task_id,
                "session_index": session_index,
                "error": repr(exc),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            failed_sessions.append(failure)
            _write_json(config_dir / "last_failure.json", failure)
            print(f"session failed task_id={session.task_id}: {exc}", flush=True)
            if args.stop_on_failure:
                raise

    aggregate = write_config_aggregate(config_dir=config_dir, sessions=sessions)
    if failed_sessions:
        _write_json(config_dir / "failures.json", {"failures": failed_sessions})
        return {
            "status": "failed",
            "config_order": config_order,
            "config_path": config_path,
        }
    else:
        _unlink_if_exists(config_dir / "failures.json")
        _unlink_if_exists(config_dir / "last_failure.json")
        _write_json(
            config_dir / "completed.json",
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "sessions": len(sessions),
                "queries": aggregate["queries"],
                "aggregate_summary": "aggregate_summary.json",
            },
        )
        return True


def prepare_config_run(*, config_order: int, config_path: Path, batch_dir: Path, args) -> dict[str, Any]:
    config = load_run_config(config_path)
    if args.max_visual_tokens is not None:
        config = replace(
            config,
            page=replace(config.page, max_visual_tokens=args.max_visual_tokens),
        )
    if args.max_cache_visual_tokens is not None:
        config = replace(
            config,
            page=replace(config.page, max_cache_visual_tokens=args.max_cache_visual_tokens),
        )
    config_dir = _config_output_dir(batch_dir, config_order, config_path)
    config_dir.mkdir(parents=True, exist_ok=True)
    if (config_dir / "completed.json").exists() and not args.force:
        print(f"skip completed config order={config_order} path={config_path}", flush=True)
        return {"status": "completed", "config_order": config_order, "config_path": config_path}

    timestamp = batch_dir.name
    root_run_tag = f"{config.run_name}__{timestamp}__cfg{config_order:03d}"
    config_root = _apply_run_tag(config, root_run_tag)
    manifest = build_run_manifest(
        config=config_root,
        config_path=config_path,
        output_root=batch_dir,
        output_dir=config_dir,
        timestamp=timestamp,
        query_limit=args.query_limit,
        git_commit=_git_commit(),
    )
    TraceLogger(config_dir).write_json("run_config.json", config_root.to_dict())
    TraceLogger(config_dir).write_json("manifest.json", manifest)

    memory_units, queries = _load_dataset(config)
    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise ValueError("--query-limit must be positive")
        queries = queries[: args.query_limit]
    sessions = _build_sessions(memory_units, queries, atomicity=args.atomicity)
    if args.session_limit is not None:
        if args.session_limit <= 0:
            raise ValueError("--session-limit must be positive")
        sessions = sessions[: args.session_limit]

    preflight = _run_preflight(
        config=config,
        memory_units=memory_units,
        queries=queries,
        sessions=sessions,
        args=args,
    )
    _write_json(config_dir / "preflight.json", preflight)
    if preflight["errors"]:
        print(
            f"preflight failed order={config_order} path={config_path} "
            f"errors={len(preflight['errors'])}",
            flush=True,
        )
        _write_json(
            config_dir / "failures.json",
            {
                "failures": [
                    {
                        "session_id": None,
                        "session_index": None,
                        "error": "preflight failed",
                        "details": preflight["errors"],
                        "time": datetime.now().isoformat(timespec="seconds"),
                    }
                ]
            },
        )
        return {"status": "failed", "config_order": config_order, "config_path": config_path}

    print(
        f"config order={config_order} path={config_path} sessions={len(sessions)} "
        f"queries={sum(len(session.queries) for session in sessions)}",
        flush=True,
    )
    if args.plan_only:
        _write_jsonl(
            config_dir / "session_plan.jsonl",
            [
                {
                    "session_position": index,
                    "task_id": session.task_id,
                    "memory_count": len(session.memory_units),
                    "query_count": len(session.queries),
                }
                for index, session in enumerate(sessions, start=1)
            ],
        )
        _write_json(
            config_dir / "plan_only.json",
            {
                "config_path": str(config_path),
                "sessions": len(sessions),
                "queries": sum(len(session.queries) for session in sessions),
                "atomicity": args.atomicity,
            },
        )
    return {
        "status": "ready",
        "config_order": config_order,
        "config_path": config_path,
        "config": config,
        "config_dir": config_dir,
        "root_run_tag": root_run_tag,
        "sessions": sessions,
    }


def _validate_interleaved_sessions(states: list[dict[str, Any]]) -> None:
    if not states:
        return
    reference = tuple(session.task_id for session in states[0]["sessions"])
    for state in states[1:]:
        current = tuple(session.task_id for session in state["sessions"])
        if current != reference:
            raise ValueError(
                "session schedule requires all configs to have identical session order; "
                f"config_order={state['config_order']} differs"
            )


def run_session(
    *,
    config: RunConfig,
    config_dir: Path,
    root_run_tag: str,
    session: "Session",
    session_index: int,
    session_count: int,
    args,
) -> None:
    session_dir = config_dir / "sessions" / _slug(session.task_id)
    completed_path = session_dir / "completed.json"
    if completed_path.exists() and not args.force:
        print(
            f"skip completed session {session_index}/{session_count} task_id={session.task_id}",
            flush=True,
        )
        return

    session_dir.mkdir(parents=True, exist_ok=True)
    _write_session_inputs(session_dir, session)
    attempts_dir = session_dir / "attempts"
    attempts_dir.mkdir(exist_ok=True)

    for attempt in range(1, args.session_retries + 2):
        attempt_dir = attempts_dir / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_tag = f"{root_run_tag}__{_slug(session.task_id)}__a{attempt:03d}"
        attempt_config = _session_config(config, session_dir=session_dir, run_tag=attempt_tag)
        try:
            print(
                f"session {session_index}/{session_count} task_id={session.task_id} "
                f"queries={len(session.queries)} attempt={attempt}",
                flush=True,
            )
            progress = (
                SessionProgressPrinter(
                    config_dir=config_dir,
                    session_id=session.task_id,
                    session_index=session_index,
                    session_count=session_count,
                    attempt=attempt,
                    total_queries=len(session.queries),
                    every=args.progress_every,
                )
                if args.progress_every
                else None
            )
            runner = build_runner_from_config(attempt_config)
            result = runner.run(session.queries, progress_callback=progress)
            summary = TraceLogger(session_dir).write_run(
                config=attempt_config,
                result=result,
                manifest=_session_manifest(
                    config=attempt_config,
                    config_dir=config_dir,
                    session_dir=session_dir,
                    session=session,
                    attempt=attempt,
                ),
            )
            if result.background_errors and args.fail_on_background_errors:
                raise RuntimeError(
                    f"session {session.task_id} completed foreground queries but has "
                    f"{len(result.background_errors)} background errors"
                )
            _write_json(
                completed_path,
                {
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "task_id": session.task_id,
                    "attempt": attempt,
                    "queries": len(result.rows),
                    "summary": summary,
                },
            )
            _write_heartbeat(
                config_dir,
                {
                    "event": "session_completed",
                    "task_id": session.task_id,
                    "session_index": session_index,
                    "session_count": session_count,
                    "attempt": attempt,
                    "summary": summary,
                },
            )
            return
        except Exception as exc:
            error = {
                "task_id": session.task_id,
                "attempt": attempt,
                "error": repr(exc),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            _write_json(attempt_dir / "error.json", error)
            _write_heartbeat(config_dir, {"event": "session_attempt_failed", **error})
            if attempt > args.session_retries:
                raise
            time.sleep(args.retry_sleep)


class SessionProgressPrinter:
    def __init__(
        self,
        *,
        config_dir: Path,
        session_id: str,
        session_index: int,
        session_count: int,
        attempt: int,
        total_queries: int,
        every: int,
    ) -> None:
        if every <= 0:
            raise ValueError("progress interval must be positive")
        self.config_dir = config_dir
        self.session_id = session_id
        self.session_index = session_index
        self.session_count = session_count
        self.attempt = attempt
        self.total_queries = total_queries
        self.every = every
        self.started_at = time.perf_counter()
        self.ttft_sum = 0.0
        self.warm_count = 0
        self.fallback_count = 0
        self.visual_budget_fallback_count = 0

    def __call__(self, completed: int, total: int, row) -> None:
        del total
        current_ttft = row.ttft_ms if row.ttft_ms is not None else row.engine_ttft_ms
        self.ttft_sum += current_ttft
        if row.mode == "warm_page":
            self.warm_count += 1
        else:
            self.fallback_count += 1
        if row.metadata.get("visual_budget_fallback"):
            self.visual_budget_fallback_count += 1
        if completed != self.total_queries and completed % self.every != 0:
            return
        elapsed = max(0.001, time.perf_counter() - self.started_at)
        avg_ttft = self.ttft_sum / completed
        heartbeat = {
            "event": "session_progress",
            "session": self.session_id,
            "session_index": self.session_index,
            "session_count": self.session_count,
            "attempt": self.attempt,
            "completed_queries": completed,
            "total_queries": self.total_queries,
            "current_query": row.query_id,
            "mode": row.mode,
            "coverage": row.coverage,
            "ttft_ms": current_ttft,
            "avg_ttft_ms": avg_ttft,
            "server_ttft_ms": row.server_ttft_ms,
            "pre_scheduler_delay_ms": row.pre_scheduler_delay_ms,
            "engine_ttft_ms": row.engine_ttft_ms,
            "amp": row.read_amplification,
            "warm": self.warm_count,
            "fallback": self.fallback_count,
            "visual_budget_fallbacks": self.visual_budget_fallback_count,
            "registered_pages": row.registered_page_count,
            "inflight": row.background_inflight_count,
            "elapsed_s": elapsed,
        }
        _write_heartbeat(self.config_dir, heartbeat)
        print(
            "session_progress "
            f"session={self.session_id} "
            f"{completed}/{self.total_queries} "
            f"current_query={row.query_id} "
            f"mode={row.mode} "
            f"coverage={row.coverage:.2f} "
            f"ttft_ms={current_ttft:.1f} "
            f"avg_ttft_ms={avg_ttft:.1f} "
            f"server_ttft_ms={_fmt_optional(row.server_ttft_ms)} "
            f"pre_scheduler_delay_ms={_fmt_optional(row.pre_scheduler_delay_ms)} "
            f"engine_ttft_ms={row.engine_ttft_ms:.1f} "
            f"amp={row.read_amplification:.2f} "
            f"warm={self.warm_count} "
            f"fallback={self.fallback_count} "
            f"visual_budget_fallbacks={self.visual_budget_fallback_count} "
            f"registered_pages={row.registered_page_count} "
            f"inflight={row.background_inflight_count} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )


def _fmt_optional(value: float | None) -> str:
    return "na" if value is None else f"{value:.1f}"


def write_config_aggregate(*, config_dir: Path, sessions: list["Session"]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    session_index_rows: list[dict[str, Any]] = []
    completed_sessions = 0
    background_error_count = 0
    for session_position, session in enumerate(sessions, start=1):
        session_dir = config_dir / "sessions" / _slug(session.task_id)
        completed_path = session_dir / "completed.json"
        if not completed_path.exists():
            session_index_rows.append(
                {
                    "task_id": session.task_id,
                    "session_position": session_position,
                    "status": "incomplete",
                    "session_dir": str(session_dir),
                }
            )
            continue
        completed_sessions += 1
        summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
        background_error_count += int(summary.get("background_error_count", 0))
        trace_rows = _read_jsonl(session_dir / "trace.jsonl")
        page_rows = _read_jsonl(session_dir / "pages.jsonl")
        pages_by_key = {
            str(page.get("page", {}).get("page_key")): page
            for page in page_rows
            if page.get("page", {}).get("page_key") is not None
        }
        for local_position, row in enumerate(trace_rows, start=1):
            _enrich_trace_row_with_page(row, pages_by_key)
            row["task_id"] = session.task_id
            row["session_position"] = session_position
            row["local_position"] = local_position
            row["global_position"] = len(rows) + 1
            rows.append(row)
        for page in page_rows:
            page["task_id"] = session.task_id
            page["session_position"] = session_position
            pages.append(page)
        session_index_rows.append(
            {
                "task_id": session.task_id,
                "session_position": session_position,
                "status": "completed",
                "query_count": len(trace_rows),
                "page_count": len(page_rows),
                "session_dir": str(session_dir),
                "summary": summary,
            }
        )

    _write_jsonl(config_dir / "trace.jsonl", rows)
    _write_jsonl(config_dir / "pages.jsonl", pages)
    _write_jsonl(config_dir / "session_index.jsonl", session_index_rows)
    # Reconstruct the main summary directly from JSON rows to keep aggregate
    # independent of dataclass deserialization.
    aggregate = _summarize_json_rows(
        rows,
        registered_pages=len(pages),
        background_error_count=background_error_count,
    )
    aggregate["sessions"] = len(sessions)
    aggregate["completed_sessions"] = completed_sessions
    aggregate["incomplete_sessions"] = len(sessions) - completed_sessions
    _write_json(config_dir / "aggregate_summary.json", aggregate)
    return aggregate


def _enrich_trace_row_with_page(row: dict[str, Any], pages_by_key: dict[str, dict[str, Any]]) -> None:
    page_key = row.get("selected_page_key")
    if page_key is None:
        row.setdefault("selected_page_source", None)
        row.setdefault("selected_page_root_source", None)
        row.setdefault("selected_page_submitted_at", None)
        row.setdefault("selected_page_unit_count", 0)
        return
    page = pages_by_key.get(str(page_key))
    if page is None:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row.setdefault("selected_page_source", metadata.get("selected_page_source"))
        row.setdefault("selected_page_root_source", metadata.get("selected_page_root_source"))
        row.setdefault("selected_page_submitted_at", metadata.get("selected_page_submitted_at"))
        row.setdefault("selected_page_unit_count", metadata.get("selected_page_unit_count", 0))
        return
    row["selected_page_source"] = _page_source(page)
    row["selected_page_root_source"] = _page_root_source(str(page_key), pages_by_key)
    row["selected_page_submitted_at"] = page.get("submitted_at_query_index")
    row["selected_page_unit_count"] = len(page.get("page", {}).get("unit_ids", []))


def _page_root_source(page_key: str, pages_by_key: dict[str, dict[str, Any]]) -> str | None:
    current_key: str | None = page_key
    visited: set[str] = set()
    root_source: str | None = None
    while current_key is not None and current_key not in visited:
        visited.add(current_key)
        page = pages_by_key.get(current_key)
        if page is None:
            break
        source = _page_source(page)
        if source is not None:
            root_source = source
        if source != "foreground_residual":
            break
        metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
        selected_key = metadata.get("selected_page_key")
        current_key = str(selected_key) if selected_key else None
    return root_source


def _page_source(page: dict[str, Any]) -> str | None:
    metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
    visual_page = page.get("page") if isinstance(page.get("page"), dict) else {}
    page_metadata = (
        visual_page.get("metadata")
        if isinstance(visual_page.get("metadata"), dict)
        else {}
    )
    source = metadata.get("source") or page_metadata.get("source")
    if source is not None:
        return str(source)
    candidate_id = metadata.get("candidate_id") or page_metadata.get("candidate_id")
    if candidate_id is None:
        return None
    return str(candidate_id).split(":", 1)[0]


def _summarize_json_rows(
    rows: list[dict[str, Any]],
    *,
    registered_pages: int,
    background_error_count: int,
) -> dict[str, Any]:
    if not rows:
        return {
            "queries": 0,
            "registered_pages": registered_pages,
            "background_error_count": background_error_count,
        }
    engine = [float(row["engine_ttft_ms"]) for row in rows]
    server = [
        float(row["server_ttft_ms"])
        for row in rows
        if row.get("server_ttft_ms") is not None
    ]
    primary_ttft = [
        float(
            row.get("ttft_ms")
            if row.get("ttft_ms") is not None
            else row.get("server_ttft_ms")
            if row.get("server_ttft_ms") is not None
            else row["engine_ttft_ms"]
        )
        for row in rows
    ]
    residual = [int(row["residual_count"]) for row in rows]
    evidence = [float(row["evidence_coverage"]) for row in rows]
    amp = [float(row["read_amplification"]) for row in rows]
    submitted = {key for row in rows for key in row.get("submitted_page_keys", [])}
    used = {row["selected_page_key"] for row in rows if row.get("selected_page_key") is not None}
    wasted = len(submitted - used)
    return {
        "queries": len(rows),
        "baseline_cold_queries": sum(1 for row in rows if row["mode"] == "baseline_cold"),
        "warm_page_queries": sum(1 for row in rows if row["mode"] == "warm_page"),
        "full_hit_queries": sum(1 for row in rows if row["execution_path"] == "full_hit"),
        "partial_hit_queries": sum(1 for row in rows if row["execution_path"] == "partial_hit"),
        "fallback_queries": sum(1 for row in rows if row["execution_path"] == "fallback"),
        "registered_pages": registered_pages,
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
        "server_ttft_ms_mean": sum(server) / len(server) if server else None,
        "server_ttft_ms_min": min(server) if server else None,
        "server_ttft_ms_max": max(server) if server else None,
        "residual_count_mean": sum(residual) / len(residual),
        "evidence_coverage_mean": sum(evidence) / len(evidence),
        "missing_evidence_queries": sum(1 for value in evidence if value < 1.0),
        "read_amplification_mean": sum(amp) / len(amp),
        "submitted_pages": sum(len(row.get("submitted_page_keys", [])) for row in rows),
        "unique_submitted_pages": len(submitted),
        "used_pages": len(used),
        "warm_events": sum(1 for row in rows if row["mode"] == "warm_page"),
        "wasted_submitted_pages": wasted,
        "wasted_submitted_page_rate": wasted / len(submitted) if submitted else None,
        "background_error_count": background_error_count,
        "visual_budget_fallback_queries": sum(
            1
            for row in rows
            if isinstance(row.get("metadata"), dict)
            and row["metadata"].get("visual_budget_fallback")
        ),
    }


class Session:
    def __init__(self, *, task_id: str, memory_units: list[MemoryUnit], queries: list[QueryRecord]) -> None:
        self.task_id = task_id
        self.memory_units = memory_units
        self.queries = queries


def _group_sessions(memory_units: list[MemoryUnit], queries: list[QueryRecord]) -> list[Session]:
    memory_by_task: dict[str, list[MemoryUnit]] = {}
    for unit in memory_units:
        if unit.task_id is None:
            raise ValueError(f"memory unit {unit.unit_id} lacks task_id; cannot run session mode")
        memory_by_task.setdefault(unit.task_id, []).append(unit)
    query_by_task: dict[str, list[QueryRecord]] = {}
    order: list[str] = []
    for query in queries:
        if query.task_id not in query_by_task:
            order.append(query.task_id)
            query_by_task[query.task_id] = []
        query_by_task[query.task_id].append(query)
    sessions: list[Session] = []
    for task_id in order:
        if task_id not in memory_by_task:
            raise ValueError(f"task {task_id!r} has queries but no memory units")
        sessions.append(Session(task_id=task_id, memory_units=memory_by_task[task_id], queries=query_by_task[task_id]))
    return sessions


def _build_sessions(
    memory_units: list[MemoryUnit],
    queries: list[QueryRecord],
    *,
    atomicity: str,
) -> list[Session]:
    if atomicity == "task":
        return _group_sessions(memory_units, queries)
    if atomicity == "whole":
        return [
            Session(
                task_id="whole_trace",
                memory_units=memory_units,
                queries=queries,
            )
        ]
    raise ValueError(f"unsupported atomicity {atomicity!r}")


def _load_dataset(config: RunConfig) -> tuple[list[MemoryUnit], list[QueryRecord]]:
    adapter = GenericJsonlAdapter(config.dataset.memory_path, config.dataset.query_path)
    return adapter.load_memory_units(), adapter.load_queries()


def _run_preflight(
    *,
    config: RunConfig,
    memory_units: list[MemoryUnit],
    queries: list[QueryRecord],
    sessions: list["Session"],
    args,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "memory_units": len(memory_units),
        "queries": len(queries),
        "sessions": len(sessions),
        "atomicity": args.atomicity,
        "topk": config.retrieval.topk,
        "page_max_units_effective": _effective_page_max_units(config),
    }

    _preflight_paths(config, errors=errors)
    _preflight_dataset(config, memory_units, queries, sessions, errors=errors, stats=stats)
    _preflight_embeddings(config, memory_units, queries, errors=errors, stats=stats)
    _preflight_visual_budget(config, memory_units, warnings=warnings, stats=stats)
    if not args.plan_only and not args.skip_runtime_preflight:
        _preflight_runtime(config, errors=errors, warnings=warnings, stats=stats)

    return {
        "ok": not errors,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def _preflight_paths(config: RunConfig, *, errors: list[str]) -> None:
    for label, path in [
        ("dataset.memory_path", config.dataset.memory_path),
        ("dataset.query_path", config.dataset.query_path),
    ]:
        if not Path(path).exists():
            errors.append(f"{label} does not exist: {path}")
    if (
        config.retrieval.mode == "embedding_retrieve"
        or config.locality.mode == "embedding_ball"
    ):
        if not config.retrieval.embedding_cache_path:
            errors.append("embedding cache path is required")
        elif not Path(config.retrieval.embedding_cache_path).exists():
            errors.append(
                f"retrieval.embedding_cache_path does not exist: "
                f"{config.retrieval.embedding_cache_path}"
            )
    try:
        Path(config.renderer.output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"renderer.output_dir is not writable: {config.renderer.output_dir}: {exc}")


def _preflight_dataset(
    config: RunConfig,
    memory_units: list[MemoryUnit],
    queries: list[QueryRecord],
    sessions: list["Session"],
    *,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    if not memory_units:
        errors.append("dataset has no memory units")
    if not queries:
        errors.append("dataset has no queries")
    memory_ids = [unit.unit_id for unit in memory_units]
    query_ids = [query.query_id for query in queries]
    if len(memory_ids) != len(set(memory_ids)):
        errors.append("dataset has duplicate memory unit ids")
    if len(query_ids) != len(set(query_ids)):
        errors.append("dataset has duplicate query ids")

    known_memory = set(memory_ids)
    unknown_access = [
        (query.query_id, unit_id)
        for query in queries
        for unit_id in query.access_units
        if unit_id not in known_memory
    ]
    if unknown_access:
        errors.append(
            f"queries reference unknown access_units; first examples={unknown_access[:5]}"
        )

    try:
        from visual_memory_system.runner.system_runner import SystemRunner

        SystemRunner._validate_query_order(queries)
    except Exception as exc:
        errors.append(f"query order validation failed: {exc}")

    too_small = [
        (session.task_id, len(session.memory_units))
        for session in sessions
        if len(session.memory_units) < config.retrieval.topk
    ]
    if too_small:
        errors.append(
            f"topk={config.retrieval.topk} exceeds memory count for sessions; "
            f"first examples={too_small[:5]}"
        )
    stats["min_session_memory_units"] = min(
        (len(session.memory_units) for session in sessions),
        default=0,
    )
    stats["max_session_memory_units"] = max(
        (len(session.memory_units) for session in sessions),
        default=0,
    )
    stats["max_session_queries"] = max((len(session.queries) for session in sessions), default=0)


def _preflight_embeddings(
    config: RunConfig,
    memory_units: list[MemoryUnit],
    queries: list[QueryRecord],
    *,
    errors: list[str],
    stats: dict[str, Any],
) -> None:
    if not (
        config.retrieval.mode == "embedding_retrieve"
        or config.locality.mode == "embedding_ball"
    ):
        return
    if not config.retrieval.embedding_cache_path:
        return
    try:
        memory_embeddings, query_embeddings = load_embedding_cache(
            config.retrieval.embedding_cache_path
        )
    except Exception as exc:
        errors.append(f"failed to load embedding cache: {exc}")
        return
    missing_memory = [
        unit.unit_id for unit in memory_units if unit.unit_id not in memory_embeddings
    ]
    missing_queries = [
        query.query_id for query in queries if query.query_id not in query_embeddings
    ]
    if missing_memory:
        errors.append(f"embedding cache misses memory units; first examples={missing_memory[:5]}")
    if missing_queries:
        errors.append(f"embedding cache misses queries; first examples={missing_queries[:5]}")
    memory_dims = {len(value) for value in memory_embeddings.values()}
    query_dims = {len(value) for value in query_embeddings.values()}
    if len(memory_dims) != 1 or len(query_dims) != 1 or memory_dims != query_dims:
        errors.append(
            f"embedding dimensions are inconsistent: memory={sorted(memory_dims)[:5]} "
            f"query={sorted(query_dims)[:5]}"
        )
    stats["embedding_memory_count"] = len(memory_embeddings)
    stats["embedding_query_count"] = len(query_embeddings)
    stats["embedding_dims"] = sorted(memory_dims | query_dims)


def _preflight_visual_budget(
    config: RunConfig,
    memory_units: list[MemoryUnit],
    *,
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    if not memory_units:
        return
    unit_heights = [
        _estimated_unit_tile_height(config, unit)
        for unit in memory_units
    ]
    page_max_units = _effective_page_max_units(config)
    topk = min(config.retrieval.topk, len(unit_heights))
    page_cap = min(page_max_units or topk, len(unit_heights))
    worst_topk = _estimate_page_visual_tokens_from_heights(
        config,
        sorted(unit_heights, reverse=True)[:topk],
    )
    worst_page_cap = _estimate_page_visual_tokens_from_heights(
        config,
        sorted(unit_heights, reverse=True)[:page_cap],
    )
    per_unit_tokens = [
        _estimate_page_visual_tokens_from_heights(config, [height])
        for height in unit_heights
    ]
    stats["visual_estimate"] = {
        "unit_tokens_mean": sum(per_unit_tokens) / len(per_unit_tokens),
        "unit_tokens_p95": _percentile(per_unit_tokens, 95),
        "unit_tokens_max": max(per_unit_tokens),
        "worst_topk_tokens": worst_topk,
        "worst_page_cap_tokens": worst_page_cap,
        "budget": config.page.max_visual_tokens,
        "cache_budget": config.page.max_cache_visual_tokens,
    }
    if config.page.max_visual_tokens is None:
        warnings.append("page.max_visual_tokens is not set; oversized prompts can fail unattended")
        return
    if worst_topk > config.page.max_visual_tokens:
        warnings.append(
            f"estimated worst-case topk foreground uses {worst_topk} visual tokens, "
            f"above budget {config.page.max_visual_tokens}"
        )
    cache_budget = config.page.max_cache_visual_tokens or config.page.max_visual_tokens
    if worst_page_cap > cache_budget:
        warnings.append(
            f"estimated worst-case reusable page uses {worst_page_cap} visual tokens, "
            f"above cache budget {cache_budget}; budget-aware construction "
            "will trim candidates when possible"
        )


def _preflight_runtime(
    config: RunConfig,
    *,
    errors: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    base_url = config.runtime.base_url.rstrip("/")
    try:
        models = _get_json(
            f"{base_url}/v1/models",
            timeout=10,
            api_key=config.runtime.api_key,
        )
    except Exception as exc:
        errors.append(f"failed to reach vLLM /v1/models at {base_url}: {exc}")
        return
    model_ids = [
        str(item.get("id"))
        for item in models.get("data", [])
        if isinstance(item, dict) and item.get("id") is not None
    ]
    stats["runtime_models"] = model_ids
    if model_ids and config.runtime.model not in model_ids:
        errors.append(
            f"runtime.model={config.runtime.model!r} not found in /v1/models: {model_ids}"
        )
    try:
        select = _post_json(
            f"{base_url}/v1/am/cache/select",
            {"candidates": []},
            timeout=10,
            api_key=config.runtime.api_key,
        )
        stats["runtime_cache_select_empty"] = select
    except Exception as exc:
        errors.append(f"failed to reach vLLM /v1/am/cache/select at {base_url}: {exc}")
    warnings.append(
        "preflight does not send an image request; ensure vLLM was started with "
        "--allowed-local-media-path covering renderer.output_dir"
    )


def _get_json(url: str, *, timeout: float, api_key: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_http_headers(api_key), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return decoded


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=_http_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return decoded


def _http_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _effective_page_max_units(config: RunConfig) -> int | None:
    limits: list[int] = []
    if config.page.max_units is not None:
        limits.append(config.page.max_units)
    if config.page.max_amplification is not None:
        limits.append(int(config.retrieval.topk * config.page.max_amplification))
    return min(limits) if limits else None


def _estimated_unit_tile_height(config: RunConfig, unit: MemoryUnit) -> int:
    wrapped = textwrap.wrap(unit.text, width=config.renderer.chars_per_line) or [""]
    lines = 1 + len(wrapped)
    return config.renderer.padding * 2 + lines * config.renderer.line_height


def _estimated_run_tag_height(config: RunConfig) -> int:
    # Autoeval injects a non-empty run_tag for every attempt. Use a conservative
    # representative value for preflight budget estimates.
    tag = "run_tag: exp1__YYYYMMDD_HHMMSS__cfg000__task__a001"
    lines = textwrap.wrap(tag, width=config.renderer.chars_per_line) or [tag]
    return config.renderer.padding * 2 + len(lines) * config.renderer.line_height


def _estimate_page_visual_tokens_from_heights(config: RunConfig, heights: list[int]) -> int:
    if not heights:
        return 0
    tile_count = len(heights) + 1
    height = (
        _estimated_run_tag_height(config)
        + sum(heights)
        + config.renderer.tile_gap * max(0, tile_count - 1)
    )
    layout_width = math.ceil(config.renderer.unit_width * config.renderer.width_scale)
    width = max(1, math.ceil(layout_width * config.renderer.post_render_scale))
    output_height = max(1, math.ceil(height * config.renderer.post_render_scale))
    return math.ceil((width / 32) * (output_height / 32))


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def _write_session_inputs(session_dir: Path, session: Session) -> None:
    input_dir = session_dir / "input"
    input_dir.mkdir(exist_ok=True)
    _write_jsonl(input_dir / "memory_units.jsonl", session.memory_units)
    _write_jsonl(input_dir / "query_records.jsonl", session.queries)


def _session_config(config: RunConfig, *, session_dir: Path, run_tag: str) -> RunConfig:
    input_dir = session_dir / "input"
    session_dataset = replace(
        config.dataset,
        memory_path=str(input_dir / "memory_units.jsonl"),
        query_path=str(input_dir / "query_records.jsonl"),
    )
    return _apply_run_tag(replace(config, dataset=session_dataset), run_tag)


def _apply_run_tag(config: RunConfig, run_tag: str) -> RunConfig:
    return replace(
        config,
        renderer=replace(config.renderer, run_tag=run_tag),
        page=replace(config.page, key_prefix=f"{config.page.key_prefix}:{run_tag}"),
    )


def _session_manifest(
    *,
    config: RunConfig,
    config_dir: Path,
    session_dir: Path,
    session: Session,
    attempt: int,
) -> dict[str, Any]:
    return {
        "task_id": session.task_id,
        "attempt": attempt,
        "config_dir": str(config_dir),
        "session_dir": str(session_dir),
        "query_count": len(session.queries),
        "memory_count": len(session.memory_units),
        "config": config.to_dict(),
    }


def _config_output_dir(batch_dir: Path, config_order: int, config_path: Path) -> Path:
    return batch_dir / f"config_{config_order:03d}_{_slug(config_path.stem)}"


def _resolve_batch(args) -> tuple[Path, list[Path]]:
    if args.resume_batch_dir is not None:
        batch_dir = Path(args.resume_batch_dir)
        manifest_path = batch_dir / "batch_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_paths = [Path(path) for path in manifest["config_paths"]]
        if args.configs or args.config_list:
            config_paths = _collect_config_paths(args)
        return batch_dir, config_paths
    config_paths = _collect_config_paths(args)
    if not config_paths:
        raise ValueError("provide --configs/--config-list or --resume-batch-dir")
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.output_root) / "autoeval" / batch_id, config_paths


def _collect_config_paths(args) -> list[Path]:
    paths = [Path(path) for path in args.configs]
    if args.config_list is not None:
        with Path(args.config_list).open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                paths.append(Path(stripped))
    return paths


def _write_batch_manifest(batch_dir: Path, *, args, config_paths: list[Path]) -> None:
    payload = {
        "batch_dir": str(batch_dir),
        "created_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
        "config_paths": [str(path) for path in config_paths],
        "session_retries": args.session_retries,
        "retry_sleep": args.retry_sleep,
        "query_limit": args.query_limit,
        "session_limit": args.session_limit,
        "max_visual_tokens": args.max_visual_tokens,
        "max_cache_visual_tokens": args.max_cache_visual_tokens,
        "schedule": args.schedule,
        "skip_runtime_preflight": args.skip_runtime_preflight,
        "fail_on_background_errors": args.fail_on_background_errors,
        "stop_on_failure": args.stop_on_failure,
        "args": vars(args),
        "git_commit": _git_commit(),
    }
    _write_json(batch_dir / "batch_manifest.json", payload)


def _write_heartbeat(config_dir: Path, payload: dict[str, Any]) -> None:
    payload = {
        **payload,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(config_dir / "heartbeat.json", payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False, sort_keys=True))
            f.write("\n")
    tmp_path.replace(path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


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


def _slug(value: str) -> str:
    text = str(value).strip()
    out = []
    for char in text:
        if char.isalnum() or char in "._-":
            out.append(char)
        else:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "none"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=[], help="Config JSON paths in execution order.")
    parser.add_argument("--config-list", default=None, help="Text file containing config JSON paths in order.")
    parser.add_argument("--output-root", default="paper_results/evaluation/exp1")
    parser.add_argument("--resume-batch-dir", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--session-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--schedule",
        choices=["config", "session"],
        default="config",
        help="Run one config at a time, or interleave configs at each session boundary.",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=None,
        help="Override page.max_visual_tokens for all configs in this batch.",
    )
    parser.add_argument(
        "--max-cache-visual-tokens",
        type=int,
        default=None,
        help="Override page.max_cache_visual_tokens for reusable background pages.",
    )
    parser.add_argument(
        "--skip-runtime-preflight",
        action="store_true",
        help="Skip vLLM /v1/models and /v1/am/cache/select checks before each config.",
    )
    parser.add_argument(
        "--fail-on-background-errors",
        action="store_true",
        help="Treat background prewarm errors as session failures after writing artifacts.",
    )
    parser.add_argument(
        "--atomicity",
        choices=["task", "whole"],
        default="task",
        help="Checkpoint/retry unit. Use whole when cross-task input order must be preserved.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Write session plans without running vLLM.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Compatibility no-op: autoeval continues by default.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Fail fast instead of continuing to the next session/config.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed configs/sessions.")
    args = parser.parse_args()
    if args.session_retries < 0:
        raise ValueError("--session-retries must be non-negative")
    if args.retry_sleep < 0:
        raise ValueError("--retry-sleep must be non-negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.max_visual_tokens is not None and args.max_visual_tokens <= 0:
        raise ValueError("--max-visual-tokens must be positive")
    if args.max_cache_visual_tokens is not None and args.max_cache_visual_tokens <= 0:
        raise ValueError("--max-cache-visual-tokens must be positive")
    return args


if __name__ == "__main__":
    main()
