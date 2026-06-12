"""Microbenchmark workflow-aware async visual page computation.

This script directly sends OpenAI-compatible chat/completions requests to the
modified vLLM server. It is intentionally independent from the workload runner:
the goal is to isolate foreground/background scheduling, not page prediction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PAGE_JSONL = (
    "paper_results/evaluation/exp1/autoeval/20260607_235821/"
    "config_007_exp1_locomo_scale4_semantic_qwen3_embed_4b/pages.jsonl"
)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_salt = f"{run_id}:{time.time_ns()}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    fg_image, bg_image = resolve_images(args, image_dir=image_dir)
    manifest = {
        "run_id": run_id,
        "cache_salt": cache_salt,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "model": args.model,
        "foreground_image": str(fg_image),
        "background_image": str(bg_image),
        "foreground_target_image_tokens": args.foreground_target_image_tokens,
        "background_target_image_tokens": args.background_target_image_tokens,
        "chunks": args.chunks,
        "trials": args.trials,
        "warmup": args.warmup,
        "fg_delay_ms": args.fg_delay_ms,
        "fg_interval_ms": args.fg_interval_ms,
        "no_priority_chunk_tokens": args.no_priority_chunk_tokens,
        "max_fg_probes": args.max_fg_probes,
        "unique_images": not args.reuse_images,
        "primary_ttft_metric": "server_request_to_first_token_ms",
        "secondary_ttft_metric": "engine_queue_to_first_token_ms",
    }
    write_json(output_dir / "manifest.json", manifest)

    client = VllmClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    results_path = output_dir / "results.jsonl"
    requests_path = output_dir / "requests.jsonl"
    all_results: list[dict[str, Any]] = []

    def record(result: dict[str, Any]) -> None:
        all_results.append(result)
        append_jsonl(results_path, result)

    def make_image(source: Path, *, role: str, case: str, trial: int, probe: int = 0) -> Path:
        if args.reuse_images:
            return source.resolve()
        if not args.use_pages_jsonl and not (args.foreground_image and args.background_image):
            image_tokens = (
                args.foreground_target_image_tokens
                if role == "fg"
                else args.background_target_image_tokens
            )
            return create_synthetic_image(
                image_dir / f"{case}_{role}_t{trial:04d}_p{probe:03d}.jpg",
                image_tokens=image_tokens,
                label=f"{cache_salt} {case} {role} {trial} {probe}",
            )
        return materialize_unique_image(
            source,
            image_dir / f"{case}_{role}_t{trial:04d}_p{probe:03d}.jpg",
            label=f"{cache_salt} {case} {role} {trial} {probe}",
        )

    def cache_key(*parts: object) -> str:
        return ":".join([cache_salt, *(str(part) for part in parts)])

    total_trials = args.warmup + args.trials

    print(f"output_dir={output_dir}", flush=True)
    print(f"foreground_image={fg_image}", flush=True)
    print(f"background_image={bg_image}", flush=True)

    # Foreground-only baseline.
    for trial in range(total_trials):
        phase = "warmup" if trial < args.warmup else "measure"
        image = make_image(fg_image, role="fg", case="fg_only", trial=trial)
        req = RequestSpec(
            case="fg_only",
            role="foreground",
            trial=trial,
            phase=phase,
            image=image,
            text=f"Answer briefly. trial={trial}",
            cache_key=cache_key("fg_only", "fg", trial),
            max_tokens=args.foreground_max_tokens,
            vllm_xargs={"am_role": "foreground", "am_cache_key": cache_key("fg_only", "fg", trial)},
        )
        append_jsonl(requests_path, req.to_json())
        record(client.send(req))
        progress("fg_only", trial + 1, total_trials)

    fg_only_server_mean = mean_metric(
        all_results,
        case="fg_only",
        phase="measure",
        field="server_ttft_ms",
    )
    if not math.isfinite(fg_only_server_mean) or fg_only_server_mean <= 0:
        raise RuntimeError(
            "failed to measure server-side foreground-only TTFT; "
            "restart vLLM with server_request_to_first_token_ms enabled"
        )

    fg_only_engine_mean = mean_metric(
        all_results,
        case="fg_only",
        phase="measure",
        field="engine_ttft_ms",
    )
    if not math.isfinite(fg_only_engine_mean) or fg_only_engine_mean <= 0:
        fg_only_engine_mean = float("nan")

    if not args.skip_experiment_a:
        run_experiment_a(
            args=args,
            client=client,
            fg_image=fg_image,
            bg_image=bg_image,
            make_image=make_image,
            record=record,
            requests_path=requests_path,
            run_id=run_id,
            cache_key=cache_key,
            total_trials=total_trials,
            fg_only_server_mean=fg_only_server_mean,
            fg_only_engine_mean=fg_only_engine_mean,
        )

    if not args.skip_experiment_b:
        run_experiment_b(
            args=args,
            client=client,
            fg_image=fg_image,
            bg_image=bg_image,
            make_image=make_image,
            record=record,
            requests_path=requests_path,
            run_id=run_id,
            cache_key=cache_key,
            total_trials=total_trials,
            fg_only_server_mean=fg_only_server_mean,
            fg_only_engine_mean=fg_only_engine_mean,
        )

    summary = summarize(
        all_results,
        fg_only_server_mean=fg_only_server_mean,
        fg_only_engine_mean=fg_only_engine_mean,
    )
    write_json(output_dir / "summary.json", summary)
    print(f"summary={output_dir / 'summary.json'}", flush=True)


def run_experiment_a(
    *,
    args: argparse.Namespace,
    client: "VllmClient",
    fg_image: Path,
    bg_image: Path,
    make_image,
    record,
    requests_path: Path,
    run_id: str,
    cache_key,
    total_trials: int,
    fg_only_server_mean: float,
    fg_only_engine_mean: float,
) -> None:
    chunk = args.experiment_a_chunk
    for trial in range(total_trials):
        phase = "warmup" if trial < args.warmup else "measure"

        # No-priority/no-chunk simulation: the would-be BG request remains
        # background work, but explicitly refuses foreground-triggered pause.
        # Treating it as a normal foreground request is not equivalent because
        # vLLM's regular continuous batching may still admit the later FG.
        case = "a_no_priority"
        fake_bg = RequestSpec(
            case=case,
            role="fake_background_no_priority",
            trial=trial,
            phase=phase,
            image=make_image(bg_image, role="fake_bg", case=case, trial=trial),
            text=f"Prefill probe. trial={trial}",
            cache_key=cache_key(case, "fake_bg", trial),
            max_tokens=args.background_max_tokens,
            vllm_xargs={
                "am_role": "background",
                "am_cache_key": cache_key(case, "fake_bg", trial),
                "am_prefill_only": 1,
                "am_chunk_tokens": args.no_priority_chunk_tokens,
                "am_decode_overlap": 0,
                "am_pause_on_foreground": 0,
                "am_evictable": 1,
            },
        )
        fg = RequestSpec(
            case=case,
            role="foreground_probe",
            trial=trial,
            phase=phase,
            image=make_image(fg_image, role="fg", case=case, trial=trial),
            text=f"Answer briefly. trial={trial}",
            cache_key=cache_key(case, "fg", trial),
            max_tokens=args.foreground_max_tokens,
            vllm_xargs={
                "am_role": "foreground",
                "am_cache_key": cache_key(case, "fg", trial),
            },
        )
        run_pair(
            client=client,
            first=fake_bg,
            foregrounds=[fg],
            delay_ms=args.fg_delay_ms,
            record=record,
            requests_path=requests_path,
        )
        progress(case, trial + 1, total_trials)

        # Chunked background: BG is low-priority, prefill-only, and pausable.
        case = "a_chunked"
        bg = background_spec(
            args=args,
            run_id=run_id,
            cache_key=cache_key,
            case=case,
            trial=trial,
            phase=phase,
            image=make_image(bg_image, role="bg", case=case, trial=trial),
            chunk=chunk,
        )
        fg = RequestSpec(
            case=case,
            role="foreground_probe",
            trial=trial,
            phase=phase,
            image=make_image(fg_image, role="fg", case=case, trial=trial),
            text=f"Answer briefly. trial={trial}",
            cache_key=cache_key(case, "fg", trial),
            max_tokens=args.foreground_max_tokens,
            vllm_xargs={
                "am_role": "foreground",
                "am_cache_key": cache_key(case, "fg", trial),
            },
        )
        run_pair(
            client=client,
            first=bg,
            foregrounds=[fg],
            delay_ms=args.fg_delay_ms,
            record=record,
            requests_path=requests_path,
        )
        progress(case, trial + 1, total_trials)


def run_experiment_b(
    *,
    args: argparse.Namespace,
    client: "VllmClient",
    fg_image: Path,
    bg_image: Path,
    make_image,
    record,
    requests_path: Path,
    run_id: str,
    cache_key,
    total_trials: int,
    fg_only_server_mean: float,
    fg_only_engine_mean: float,
) -> None:
    for chunk in args.chunks:
        # BG-only completion.
        for trial in range(total_trials):
            phase = "warmup" if trial < args.warmup else "measure"
            case = f"b_bg_only_c{chunk}"
            bg = background_spec(
                args=args,
                run_id=run_id,
                cache_key=cache_key,
                case=case,
                trial=trial,
                phase=phase,
                image=make_image(bg_image, role="bg", case=case, trial=trial),
                chunk=chunk,
            )
            append_jsonl(requests_path, bg.to_json())
            record(client.send(bg))
            progress(case, trial + 1, total_trials)

        # Mixed: run one BG request while periodically sending FG probes.
        for trial in range(total_trials):
            phase = "warmup" if trial < args.warmup else "measure"
            case = f"b_mixed_c{chunk}"
            bg = background_spec(
                args=args,
                run_id=run_id,
                cache_key=cache_key,
                case=case,
                trial=trial,
                phase=phase,
                image=make_image(bg_image, role="bg", case=case, trial=trial),
                chunk=chunk,
            )
            foregrounds = [
                RequestSpec(
                    case=case,
                    role="foreground_probe",
                    trial=trial,
                    phase=phase,
                    probe=probe,
                    image=make_image(fg_image, role="fg", case=case, trial=trial, probe=probe),
                    text=f"Answer briefly. trial={trial} probe={probe}",
                    cache_key=cache_key(case, "fg", trial, probe),
                    max_tokens=args.foreground_max_tokens,
                    vllm_xargs={
                        "am_role": "foreground",
                        "am_cache_key": cache_key(case, "fg", trial, probe),
                    },
                )
                for probe in range(args.max_fg_probes)
            ]
            run_mixed(
                client=client,
                background=bg,
                foregrounds=foregrounds,
                interval_ms=args.fg_interval_ms,
                record=record,
                requests_path=requests_path,
            )
            progress(case, trial + 1, total_trials)


def run_pair(
    *,
    client: "VllmClient",
    first: "RequestSpec",
    foregrounds: list["RequestSpec"],
    delay_ms: int,
    record,
    requests_path: Path,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 + len(foregrounds)) as executor:
        append_jsonl(requests_path, first.to_json())
        first_future = executor.submit(client.send, first)
        time.sleep(delay_ms / 1000.0)
        fg_futures = []
        for fg in foregrounds:
            append_jsonl(requests_path, fg.to_json())
            fg_futures.append(executor.submit(client.send, fg))
        for future in fg_futures:
            record(future.result())
        record(first_future.result())


def run_mixed(
    *,
    client: "VllmClient",
    background: "RequestSpec",
    foregrounds: list["RequestSpec"],
    interval_ms: int,
    record,
    requests_path: Path,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 + len(foregrounds)) as executor:
        append_jsonl(requests_path, background.to_json())
        bg_future = executor.submit(client.send, background)
        for fg in foregrounds:
            if bg_future.done():
                break
            time.sleep(interval_ms / 1000.0)
            if bg_future.done():
                break
            append_jsonl(requests_path, fg.to_json())
            future = executor.submit(client.send, fg)
            record(future.result())
        record(bg_future.result())


def background_spec(
    *,
    args: argparse.Namespace,
    run_id: str,
    cache_key,
    case: str,
    trial: int,
    phase: str,
    image: Path,
    chunk: int,
) -> "RequestSpec":
    return RequestSpec(
        case=case,
        role="background",
        trial=trial,
        phase=phase,
        chunk=chunk,
        image=image,
        text="",
        cache_key=cache_key(case, "bg", trial),
        max_tokens=args.background_max_tokens,
        vllm_xargs={
            "am_role": "background",
            "am_cache_key": cache_key(case, "bg", trial),
            "am_prefill_only": 1,
            "am_chunk_tokens": chunk,
            "am_decode_overlap": 0,
            "am_pause_on_foreground": 1,
            "am_evictable": 1,
        },
    )


class RequestSpec:
    def __init__(
        self,
        *,
        case: str,
        role: str,
        trial: int,
        phase: str,
        image: Path,
        text: str,
        cache_key: str,
        max_tokens: int,
        vllm_xargs: dict[str, Any],
        chunk: int | None = None,
        probe: int = 0,
    ) -> None:
        self.case = case
        self.role = role
        self.trial = trial
        self.phase = phase
        self.image = image
        self.text = text
        self.cache_key = cache_key
        self.max_tokens = max_tokens
        self.vllm_xargs = vllm_xargs
        self.chunk = chunk
        self.probe = probe

    def payload(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[microbench-tag:{self.cache_key}]\n"},
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image.resolve().as_uri()},
                        },
                        {"type": "text", "text": self.text},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "stream": False,
            "vllm_xargs": self.vllm_xargs,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "role": self.role,
            "trial": self.trial,
            "phase": self.phase,
            "probe": self.probe,
            "chunk": self.chunk,
            "image": str(self.image),
            "cache_key": self.cache_key,
            "max_tokens": self.max_tokens,
            "vllm_xargs": self.vllm_xargs,
        }


class VllmClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def send(self, spec: RequestSpec) -> dict[str, Any]:
        payload = spec.payload(self.model)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        start_wall = time.time()
        start = time.perf_counter()
        error = None
        response: dict[str, Any] | None = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
        except Exception as exc:  # noqa: BLE001 - write failed trial into results.
            error = repr(exc)
        end = time.perf_counter()
        end_wall = time.time()

        metrics = response.get("vllm_internal_metrics") if isinstance(response, dict) else None
        usage = response.get("usage") if isinstance(response, dict) else None
        server_ttft = None
        engine_ttft = None
        pre_scheduler_delay = None
        if isinstance(metrics, dict):
            if metrics.get("server_request_to_first_token_ms") is not None:
                server_ttft = float(metrics["server_request_to_first_token_ms"])
            if metrics.get("engine_queue_to_first_token_ms") is not None:
                engine_ttft = float(metrics["engine_queue_to_first_token_ms"])
            if metrics.get("pre_scheduler_delay_ms") is not None:
                pre_scheduler_delay = float(metrics["pre_scheduler_delay_ms"])
        return {
            **spec.to_json(),
            "ok": error is None,
            "error": error,
            "client_start_unix": start_wall,
            "client_end_unix": end_wall,
            "client_duration_ms": (end - start) * 1000.0,
            "server_ttft_ms": server_ttft,
            "engine_ttft_ms": engine_ttft,
            "pre_scheduler_delay_ms": pre_scheduler_delay,
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "response_id": response.get("id") if isinstance(response, dict) else None,
            "finish_reason": extract_finish_reason(response),
            "vllm_internal_metrics": metrics,
        }


def summarize(
    rows: list[dict[str, Any]],
    *,
    fg_only_server_mean: float,
    fg_only_engine_mean: float,
) -> dict[str, Any]:
    measured = [row for row in rows if row.get("phase") == "measure" and row.get("ok")]
    cases = sorted({str(row["case"]) for row in measured})
    by_case: dict[str, Any] = {}
    for case in cases:
        case_rows = [row for row in measured if row["case"] == case]
        fg_rows = [
            row
            for row in case_rows
            if row["role"] in {"foreground", "foreground_probe"}
        ]
        bg_rows = [row for row in case_rows if row["role"] == "background"]
        fake_bg_rows = [row for row in case_rows if row["role"] == "fake_background_no_priority"]
        by_case[case] = {
            "rows": len(case_rows),
            "foreground": metric_summary(
                fg_rows,
            ),
            "background": metric_summary(
                bg_rows,
            ),
            "fake_background": metric_summary(
                fake_bg_rows,
            ),
        }
        fg_server = by_case[case]["foreground"].get("server_ttft_ms_mean")
        if fg_server is not None:
            by_case[case]["foreground"]["server_overhead_vs_fg_only_ms"] = max(
                0.0,
                float(fg_server) - fg_only_server_mean,
            )
            by_case[case]["foreground"]["server_slowdown_vs_fg_only"] = (
                float(fg_server) / fg_only_server_mean
            )
        fg_engine = by_case[case]["foreground"].get("engine_ttft_ms_mean")
        if fg_engine is not None and math.isfinite(fg_only_engine_mean):
            by_case[case]["foreground"]["engine_overhead_vs_fg_only_ms"] = max(
                0.0,
                float(fg_engine) - fg_only_engine_mean,
            )
            by_case[case]["foreground"]["engine_slowdown_vs_fg_only"] = (
                float(fg_engine) / fg_only_engine_mean
            )

    # Derived BG effective compute estimates for mixed cases:
    # mixed BG duration - foreground probe count * foreground-only duration.
    for case, summary in by_case.items():
        bg_server = summary["background"].get("server_ttft_ms_mean")
        bg_engine = summary["background"].get("engine_ttft_ms_mean")
        bg_client = summary["background"].get("client_duration_ms_mean")
        fg_count = summary["foreground"].get("rows", 0)
        bg_rows = max(1, summary["background"].get("rows", 1))
        if bg_server is not None:
            summary["background"]["estimated_bg_effective_server_ms"] = max(
                0.0,
                bg_server - fg_count * fg_only_server_mean / bg_rows,
            )
        if bg_engine is not None and math.isfinite(fg_only_engine_mean):
            summary["background"]["estimated_bg_effective_engine_ms"] = max(
                0.0,
                bg_engine - fg_count * fg_only_engine_mean / bg_rows,
            )
        if bg_client is not None:
            fg_only_client_mean = mean_metric(
                rows,
                case="fg_only",
                phase="measure",
                field="client_duration_ms",
            )
            if math.isfinite(fg_only_client_mean):
                summary["background"]["estimated_bg_effective_client_ms"] = max(
                    0.0,
                    bg_client - fg_count * fg_only_client_mean / bg_rows,
                )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "primary_ttft_metric": "server_request_to_first_token_ms",
        "secondary_ttft_metric": "engine_queue_to_first_token_ms",
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("ok")),
        "fg_only_server_ttft_ms_mean": fg_only_server_mean,
        "fg_only_engine_ttft_ms_mean": fg_only_engine_mean,
        "cases": by_case,
    }


def metric_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clients = [
        float(row["client_duration_ms"])
        for row in rows
        if row.get("client_duration_ms") is not None
    ]
    servers = [float(row["server_ttft_ms"]) for row in rows if row.get("server_ttft_ms") is not None]
    engines = [float(row["engine_ttft_ms"]) for row in rows if row.get("engine_ttft_ms") is not None]
    delays = [
        float(row["pre_scheduler_delay_ms"])
        for row in rows
        if row.get("pre_scheduler_delay_ms") is not None
    ]
    out = {
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.get("ok")),
        "client_duration_ms_mean": mean(clients),
        "client_duration_ms_p50": percentile(clients, 50),
        "client_duration_ms_p90": percentile(clients, 90),
        "client_duration_ms_p95": percentile(clients, 95),
        "server_ttft_ms_mean": mean(servers),
        "server_ttft_ms_p50": percentile(servers, 50),
        "server_ttft_ms_p90": percentile(servers, 90),
        "server_ttft_ms_p95": percentile(servers, 95),
        "engine_ttft_ms_mean": mean(engines),
        "engine_ttft_ms_p50": percentile(engines, 50),
        "engine_ttft_ms_p90": percentile(engines, 90),
        "engine_ttft_ms_p95": percentile(engines, 95),
        "pre_scheduler_delay_ms_mean": mean(delays),
        "pre_scheduler_delay_ms_p50": percentile(delays, 50),
        "pre_scheduler_delay_ms_p90": percentile(delays, 90),
        "pre_scheduler_delay_ms_p95": percentile(delays, 95),
    }
    return out


def mean_metric(rows: list[dict[str, Any]], *, case: str, phase: str, field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if row.get("case") == case
        and row.get("phase") == phase
        and row.get("ok")
        and row.get(field) is not None
    ]
    return mean(values) or float("nan")


def resolve_images(args: argparse.Namespace, *, image_dir: Path) -> tuple[Path, Path]:
    if args.foreground_image and args.background_image:
        return require_file(args.foreground_image), require_file(args.background_image)
    if not args.use_pages_jsonl:
        fg = create_synthetic_image(
            image_dir / f"source_fg_{args.foreground_target_image_tokens}vtok.jpg",
            image_tokens=args.foreground_target_image_tokens,
            label=f"foreground ~{args.foreground_target_image_tokens} visual tokens",
        )
        bg = create_synthetic_image(
            image_dir / f"source_bg_{args.background_target_image_tokens}vtok.jpg",
            image_tokens=args.background_target_image_tokens,
            label=f"background ~{args.background_target_image_tokens} visual tokens",
        )
        return fg, bg
    pages_jsonl = Path(args.pages_jsonl)
    if not pages_jsonl.exists():
        raise FileNotFoundError(
            f"provide --foreground-image/--background-image or a valid --pages-jsonl: {pages_jsonl}"
        )
    rows = []
    with pages_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            page = row.get("page", {})
            image = page.get("image_path")
            tokens = page.get("prompt_tokens_estimate")
            if image and tokens:
                rows.append((int(tokens), Path(image)))
    if len(rows) < 2:
        raise RuntimeError(f"not enough rendered pages in {pages_jsonl}")
    rows.sort(key=lambda item: item[0])
    fg = rows[max(0, len(rows) // 5)][1]
    bg = rows[-1][1]
    return require_file(fg), require_file(bg)


def create_synthetic_image(path: Path, *, image_tokens: int, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Qwen VL token accounting is approximately ceil(H/32) * ceil(W/32).
    # Use a near-square grid to avoid introducing extreme aspect ratios.
    grid_w = max(1, int(math.sqrt(image_tokens)))
    grid_h = max(1, math.ceil(image_tokens / grid_w))
    width = grid_w * 32
    height = grid_h * 32
    try:
        from PIL import Image, ImageDraw

        digest = hashlib.sha256(label.encode("utf-8")).digest()
        base = (
            220 + digest[0] % 28,
            220 + digest[1] % 28,
            220 + digest[2] % 28,
        )
        accent_a = (40 + digest[3] % 140, 50 + digest[4] % 130, 90 + digest[5] % 120)
        accent_b = (90 + digest[6] % 120, 40 + digest[7] % 140, 50 + digest[8] % 130)
        image = Image.new("RGB", (width, height), color=base)
        draw = ImageDraw.Draw(image)
        for i, y in enumerate(range(0, height, 96)):
            color = accent_a if i % 2 == 0 else accent_b
            draw.rectangle((0, y, width, min(height, y + 32)), fill=color)
        for i, x in enumerate(range(0, width, 128)):
            color = accent_b if i % 2 == 0 else accent_a
            draw.line((x, 0, width - x // 3, height), fill=color, width=3)
        for idx in range(12):
            byte = digest[idx % len(digest)]
            x0 = (byte * (idx + 17) * 13) % max(1, width)
            y0 = (byte * (idx + 11) * 17) % max(1, height)
            x1 = min(width, x0 + 80 + digest[(idx + 9) % len(digest)] % 220)
            y1 = min(height, y0 + 60 + digest[(idx + 15) % len(digest)] % 180)
            fill = accent_a if idx % 2 == 0 else accent_b
            draw.rectangle((x0, y0, x1, y1), outline=fill, width=5)
        box_w = min(width - 32, 920)
        draw.rectangle((16, 16, 16 + box_w, 118), fill=(255, 255, 255))
        draw.text((28, 32), f"{label[:96]}", fill=(0, 0, 0))
        draw.text((28, 72), f"size={width}x{height}; target_vtokens={image_tokens}", fill=(0, 0, 0))
        image.save(path, format="JPEG", quality=90)
    except Exception as exc:
        raise RuntimeError("Pillow is required to create synthetic microbench images") from exc
    return path.resolve()


def materialize_unique_image(source: Path, target: Path, *, label: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw

        with Image.open(source) as image:
            rgb = image.convert("RGB")
            draw = ImageDraw.Draw(rgb)
            x = 0
            y = 0
            draw.rectangle((x, y, min(rgb.width, 720), min(rgb.height, 56)), fill=(255, 255, 255))
            draw.text((x + 8, y + 8), label[:96], fill=(0, 0, 0))
            rgb.save(target, format="JPEG", quality=88)
    except Exception:
        shutil.copy2(source, target)
    return target.resolve()


def extract_finish_reason(response: dict[str, Any] | None) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    value = first.get("finish_reason")
    return None if value is None else str(value)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_file(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def progress(case: str, current: int, total: int) -> None:
    print(f"progress case={case} {current}/{total}", flush=True)


def parse_chunks(value: str) -> list[int]:
    chunks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not chunks or any(chunk <= 0 for chunk in chunks):
        raise argparse.ArgumentTypeError("--chunks must be a comma-separated list of positive ints")
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3vl-8b")
    parser.add_argument("--api-key")
    parser.add_argument("--foreground-image")
    parser.add_argument("--background-image")
    parser.add_argument("--pages-jsonl", default=DEFAULT_PAGE_JSONL)
    parser.add_argument("--use-pages-jsonl", action="store_true")
    parser.add_argument("--foreground-target-image-tokens", type=int, default=500)
    parser.add_argument("--background-target-image-tokens", type=int, default=5000)
    parser.add_argument("--output-root", default="paper_results/evaluation/microbench_async_chunk")
    parser.add_argument("--run-id")
    parser.add_argument("--chunks", type=parse_chunks, default=parse_chunks("512,1024,2048,4096"))
    parser.add_argument("--experiment-a-chunk", type=int, default=1024)
    parser.add_argument("--no-priority-chunk-tokens", type=int, default=100000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--fg-delay-ms", type=int, default=100)
    parser.add_argument("--fg-interval-ms", type=int, default=100)
    parser.add_argument("--max-fg-probes", type=int, default=8)
    parser.add_argument("--foreground-max-tokens", type=int, default=1)
    parser.add_argument("--background-max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--reuse-images", action="store_true")
    parser.add_argument("--skip-experiment-a", action="store_true")
    parser.add_argument("--skip-experiment-b", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.experiment_a_chunk <= 0:
        parser.error("--experiment-a-chunk must be positive")
    if args.no_priority_chunk_tokens <= 0:
        parser.error("--no-priority-chunk-tokens must be positive")
    if args.foreground_target_image_tokens <= 0:
        parser.error("--foreground-target-image-tokens must be positive")
    if args.background_target_image_tokens <= 0:
        parser.error("--background-target-image-tokens must be positive")
    return args


if __name__ == "__main__":
    main()
