"""LLM judge for generated answers in visual memory experiments.

The script is intentionally dataset-lightweight: it only extracts the question
and golden answer from processed query records. Correctness is judged by an
OpenAI-compatible chat completion endpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict but fair answer judge.

Given a question, a golden answer, and a model answer, decide whether the model
answer is semantically correct. Accept paraphrases and equivalent answers. Do
not require exact wording. Penalize answers that contradict the gold answer,
omit the key information, choose the wrong option, or only restate the question.

Return only one JSON object with these fields:
- score: 1.0 for correct, 0.5 for partially correct, 0.0 for incorrect
- label: one of "correct", "partial", "incorrect"
- reason: one concise sentence
"""


@dataclass(frozen=True)
class TargetRun:
    config_dir: Path
    trace_path: Path | None
    trace_paths: tuple[Path, ...]
    run_config_path: Path | None
    run_config: dict[str, Any]


@dataclass(frozen=True)
class JudgeItem:
    row_index: int
    dataset: str
    method: str
    query_id: str
    task_id: str | None
    question: str
    gold_answer: str
    model_answer: str
    trace_row: dict[str, Any]


@dataclass
class TargetWork:
    index: int
    target: TargetRun
    output_dir: Path
    name: str
    items: list[JudgeItem]
    pending: list[JudgeItem]
    quality_path: Path


def main() -> None:
    args = parse_args()
    targets = discover_targets(args.inputs)
    if not targets:
        raise SystemExit("no trace targets found")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [str(path) for path in args.inputs],
        "judge": {
            "base_url": args.base_url,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_workers": args.max_workers,
            "retries": args.retries,
        },
        "limit": args.limit,
        "sample": args.sample,
        "seed": args.seed,
        "schedule": args.schedule,
        "targets": [str(target.config_dir) for target in targets],
    }
    write_json(run_dir / "manifest.json", manifest)

    if args.schedule == "session":
        run_targets_by_session(targets=targets, run_dir=run_dir, args=args)
    else:
        for target_index, target in enumerate(targets, start=1):
            run_target(
                target=target,
                output_dir=target_output_dir(run_dir, target_index, target),
                index=target_index,
                target_count=len(targets),
                args=args,
            )

    write_combined_summary(run_dir)
    print(f"quality_dir={run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Config dirs, batch dirs, or trace.jsonl files to judge.",
    )
    parser.add_argument("--output-root", default="paper_results/quality")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many rows per target after loading trace rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--schedule",
        choices=("target", "session"),
        default="target",
        help="Judge one config at a time, or interleave configs by task/session id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rejudge rows even if quality.jsonl already contains them.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N completed judge rows. Use 0 to disable.",
    )
    return parser.parse_args()


def discover_targets(inputs: list[Path]) -> list[TargetRun]:
    targets_by_dir: dict[Path, TargetRun] = {}
    for raw_path in inputs:
        path = raw_path.resolve()
        if path.is_file():
            if path.name != "trace.jsonl":
                raise ValueError(f"input file must be trace.jsonl: {path}")
            target = target_from_trace(path)
            targets_by_dir[target.config_dir] = target
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        if (path / "run_config.json").exists():
            target = target_from_config_dir(path)
            targets_by_dir[target.config_dir] = target
            continue
        for trace_path in sorted(path.rglob("trace.jsonl")):
            if trace_path.parent.name == "sessions":
                continue
            if "sessions" in trace_path.parts:
                continue
            run_config_path = trace_path.parent / "run_config.json"
            if not run_config_path.exists():
                continue
            target = target_from_trace(trace_path)
            targets_by_dir[target.config_dir] = target
    return list(targets_by_dir.values())


def target_from_trace(trace_path: Path) -> TargetRun:
    config_dir = trace_path.parent
    run_config_path = config_dir / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    return TargetRun(
        config_dir=config_dir,
        trace_path=trace_path,
        trace_paths=(trace_path,),
        run_config_path=run_config_path if run_config_path.exists() else None,
        run_config=run_config,
    )


def target_from_config_dir(config_dir: Path) -> TargetRun:
    run_config_path = config_dir / "run_config.json"
    run_config = read_json(run_config_path)
    trace_path = config_dir / "trace.jsonl"
    if trace_path.exists():
        trace_paths = (trace_path,)
        primary_trace_path: Path | None = trace_path
    else:
        trace_paths = tuple(sorted((config_dir / "sessions").glob("*/trace.jsonl")))
        primary_trace_path = None
    if not trace_paths:
        raise FileNotFoundError(f"no trace.jsonl found under {config_dir}")
    return TargetRun(
        config_dir=config_dir,
        trace_path=primary_trace_path,
        trace_paths=trace_paths,
        run_config_path=run_config_path,
        run_config=run_config,
    )


def target_output_dir(run_dir: Path, target_index: int, target: TargetRun) -> Path:
    return run_dir / f"{target_index:03d}_{slug(target.config_dir)}"


def run_target(
    *,
    target: TargetRun,
    output_dir: Path,
    index: int,
    target_count: int,
    args: argparse.Namespace,
) -> None:
    work = prepare_target_work(
        target=target,
        output_dir=output_dir,
        index=index,
        target_count=target_count,
        args=args,
    )
    if work.pending:
        with work.quality_path.open("a", encoding="utf-8") as out:
            run_judges(items=work.pending, out=out, args=args, context=work.name)
    finalize_target(work)


def run_targets_by_session(
    *,
    targets: list[TargetRun],
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    works = [
        prepare_target_work(
            target=target,
            output_dir=target_output_dir(run_dir, target_index, target),
            index=target_index,
            target_count=len(targets),
            args=args,
        )
        for target_index, target in enumerate(targets, start=1)
    ]
    session_order = ordered_sessions(works)
    for session_index, session_id in enumerate(session_order, start=1):
        print(
            f"quality_session {session_index}/{len(session_order)} "
            f"task_id={session_id} targets={len(works)}",
            flush=True,
        )
        for work in works:
            session_items = [
                item for item in work.pending if normalized_task_id(item) == session_id
            ]
            if not session_items:
                continue
            print(
                f"quality target={work.name} session={session_id} "
                f"rows={len(session_items)}",
                flush=True,
            )
            with work.quality_path.open("a", encoding="utf-8") as out:
                run_judges(
                    items=session_items,
                    out=out,
                    args=args,
                    context=f"{work.name} session={session_id}",
                )
            print_partial_target_summary(work, session_id=session_id)
    for work in works:
        finalize_target(work)


def prepare_target_work(
    *,
    target: TargetRun,
    output_dir: Path,
    index: int,
    target_count: int,
    args: argparse.Namespace,
) -> TargetWork:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"quality target {index}/{target_count} "
        f"config_dir={target.config_dir}",
        flush=True,
    )
    query_path = query_path_for_target(target)
    queries = load_queries(query_path)
    rows = read_target_rows(target)
    items = build_items(target=target, rows=rows, queries=queries)
    if args.sample is not None and args.sample < len(items):
        rng = random.Random(args.seed)
        items = sorted(rng.sample(items, args.sample), key=lambda item: item.row_index)
    if args.limit is not None:
        items = items[: args.limit]

    quality_path = output_dir / "quality.jsonl"
    if args.force and quality_path.exists():
        quality_path.unlink()
    existing = {} if args.force else load_existing(quality_path)
    pending = [item for item in items if item.query_id not in existing]

    write_json(
        output_dir / "target_manifest.json",
        {
            "config_dir": str(target.config_dir),
            "trace_path": str(target.trace_path) if target.trace_path else None,
            "trace_paths": [str(path) for path in target.trace_paths],
            "run_config_path": str(target.run_config_path) if target.run_config_path else None,
            "query_path": str(query_path),
            "total_trace_rows": len(rows),
            "judge_rows": len(items),
            "already_done": len(items) - len(pending),
            "pending": len(pending),
        },
    )

    return TargetWork(
        index=index,
        target=target,
        output_dir=output_dir,
        name=target.config_dir.name,
        items=items,
        pending=pending,
        quality_path=quality_path,
    )


def read_target_rows(target: TargetRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in target.trace_paths:
        session_rows = read_jsonl(trace_path)
        if target.trace_path is None:
            task_id = trace_path.parent.name.replace("-", ":")
            for local_position, row in enumerate(session_rows, start=1):
                row.setdefault("task_id", row.get("task_id") or task_id)
                row.setdefault("local_position", local_position)
                row.setdefault("session_dir", str(trace_path.parent))
        rows.extend(session_rows)
    return rows


def finalize_target(work: TargetWork) -> None:
    judged_rows = (
        dedupe_by_query_id(read_jsonl(work.quality_path))
        if work.quality_path.exists()
        else []
    )
    summary = summarize(judged_rows)
    write_json(work.output_dir / "summary.json", summary)
    print(
        f"quality summary target={work.target.config_dir.name} "
        f"rows={summary['rows']} mean_score={summary['score_mean']:.4f} "
        f"correct_rate={summary['correct_rate']:.4f}",
        flush=True,
    )


def print_partial_target_summary(work: TargetWork, *, session_id: str) -> None:
    judged_rows = (
        dedupe_by_query_id(read_jsonl(work.quality_path))
        if work.quality_path.exists()
        else []
    )
    summary = summarize(judged_rows)
    print(
        f"quality_partial target={work.name} "
        f"after_session={session_id} rows={summary['rows']} "
        f"mean_score={summary['score_mean']:.4f} "
        f"correct_rate={summary['correct_rate']:.4f} "
        f"partial_or_correct_rate={summary['partial_or_correct_rate']:.4f}",
        flush=True,
    )


def ordered_sessions(works: list[TargetWork]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for work in works:
        for item in work.items:
            task_id = normalized_task_id(item)
            if task_id in seen:
                continue
            seen.add(task_id)
            ordered.append(task_id)
    return ordered


def normalized_task_id(item: JudgeItem) -> str:
    return item.task_id if item.task_id is not None else "__no_task__"


def write_combined_summary(run_dir: Path) -> None:
    targets: list[dict[str, Any]] = []
    all_rows = 0
    all_ok_rows = 0
    weighted_score_sum = 0.0
    weighted_correct_sum = 0.0
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        summary = read_json(summary_path)
        ok_rows = int(summary.get("ok_rows", 0))
        score_mean = float(summary.get("score_mean", 0.0))
        correct_rate = float(summary.get("correct_rate", 0.0))
        targets.append(
            {
                "target": summary_path.parent.name,
                "summary_path": str(summary_path),
                **summary,
            }
        )
        all_rows += int(summary.get("rows", 0))
        all_ok_rows += ok_rows
        weighted_score_sum += score_mean * ok_rows
        weighted_correct_sum += correct_rate * ok_rows
    write_json(
        run_dir / "summary.json",
        {
            "targets": len(targets),
            "rows": all_rows,
            "ok_rows": all_ok_rows,
            "score_mean": weighted_score_sum / all_ok_rows if all_ok_rows else 0.0,
            "correct_rate": weighted_correct_sum / all_ok_rows if all_ok_rows else 0.0,
            "target_summaries": targets,
        },
    )


def query_path_for_target(target: TargetRun) -> Path:
    dataset = target.run_config.get("dataset")
    if isinstance(dataset, dict) and dataset.get("query_path"):
        return Path(str(dataset["query_path"]))
    raise ValueError(f"cannot infer dataset.query_path from {target.config_dir}")


def load_queries(query_path: Path) -> dict[str, dict[str, Any]]:
    if not query_path.exists():
        raise FileNotFoundError(query_path)
    queries: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(query_path):
        query_id = row.get("query_id")
        if query_id is not None:
            queries[str(query_id)] = row
    return queries


def build_items(
    *,
    target: TargetRun,
    rows: list[dict[str, Any]],
    queries: dict[str, dict[str, Any]],
) -> list[JudgeItem]:
    dataset = str(target.run_config.get("dataset", {}).get("name", "unknown"))
    method = method_for_target(target)
    items: list[JudgeItem] = []
    for index, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id"))
        query = queries.get(query_id)
        if query is None:
            raise KeyError(f"query_id {query_id!r} from trace is missing in query records")
        output_text = row.get("output_text")
        if output_text is None:
            output_text = ""
        items.append(
            JudgeItem(
                row_index=index,
                dataset=dataset,
                method=method,
                query_id=query_id,
                task_id=row.get("task_id") or query.get("task_id"),
                question=str(query.get("query_text", "")),
                gold_answer=gold_answer_for_query(dataset, query),
                model_answer=str(output_text),
                trace_row=row,
            )
        )
    return items


def method_for_target(target: TargetRun) -> str:
    locality = target.run_config.get("locality")
    if isinstance(locality, dict):
        mode = str(locality.get("mode", "unknown"))
        if mode == "embedding_ball":
            return "semantic"
        if mode == "append":
            return "temporal"
        if mode == "random_anchor":
            return "random"
        if mode == "baseline":
            return "baseline"
        return mode
    return "unknown"


def gold_answer_for_query(dataset: str, query: dict[str, Any]) -> str:
    metadata = query.get("metadata") if isinstance(query.get("metadata"), dict) else {}
    if dataset == "perma":
        label = metadata.get("gold_label")
        options = str(metadata.get("options", ""))
        selected = option_text(options, str(label)) if label is not None else None
        if selected:
            return f"Correct option {label}: {selected}"
        return f"Correct option {label}\n{options}"
    answer = metadata.get("answer")
    if isinstance(answer, (list, tuple)):
        return "\n".join(str(value) for value in answer)
    return "" if answer is None else str(answer)


def option_text(options: str, label: str) -> str | None:
    prefix = f"{label}:"
    lines = options.splitlines()
    captured: list[str] = []
    active = False
    for line in lines:
        if len(line) >= 2 and line[1:2] == ":" and line[:1].isalpha():
            if active:
                break
            active = line.startswith(prefix)
        if active:
            captured.append(line)
    if not captured:
        return None
    text = "\n".join(captured)
    return text.split(":", 1)[1].strip() if ":" in text else text.strip()


def run_judges(
    *,
    items: list[JudgeItem],
    out,
    args: argparse.Namespace,
    context: str,
) -> None:
    completed = 0
    ok_count = 0
    error_count = 0
    score_sum = 0.0
    correct_count = 0
    partial_or_correct_count = 0
    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {
            executor.submit(judge_one_with_retries, item, args): item for item in items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
            except Exception as exc:  # Keep long judge runs alive.
                result = {
                    "judge_status": "error",
                    "score": None,
                    "label": "error",
                    "reason": repr(exc),
                    "raw_judge_response": None,
                }
            row = output_row(item, result)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            completed += 1
            if result.get("judge_status") == "ok" and result.get("score") is not None:
                ok_count += 1
                score = float(result["score"])
                score_sum += score
                if score >= 1.0:
                    correct_count += 1
                if score >= 0.5:
                    partial_or_correct_count += 1
            else:
                error_count += 1
            if args.progress_every and (
                completed == len(items) or completed % args.progress_every == 0
            ):
                elapsed = max(0.001, time.perf_counter() - started_at)
                mean_score = score_sum / ok_count if ok_count else 0.0
                correct_rate = correct_count / ok_count if ok_count else 0.0
                partial_rate = partial_or_correct_count / ok_count if ok_count else 0.0
                print(
                    f"judge_progress context={context} "
                    f"{completed}/{len(items)} "
                    f"rate={completed / elapsed:.2f}/s "
                    f"ok={ok_count} errors={error_count} "
                    f"mean_score={mean_score:.4f} "
                    f"correct_rate={correct_rate:.4f} "
                    f"partial_or_correct_rate={partial_rate:.4f} "
                    f"current_query={item.query_id}",
                    flush=True,
                )


def judge_one_with_retries(item: JudgeItem, args: argparse.Namespace) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, args.retries + 2):
        try:
            return judge_one(item, args)
        except Exception as exc:
            last_exc = exc
            if attempt > args.retries:
                break
            time.sleep(args.retry_sleep * attempt)
    assert last_exc is not None
    raise last_exc


def judge_one(item: JudgeItem, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(item)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    response = post_json(
        base_url=args.base_url,
        path="/v1/chat/completions",
        payload=payload,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    content = extract_message_content(response)
    parsed = parse_judge_json(content)
    score = parsed.get("score")
    if score not in (0, 0.0, 0.5, 1, 1.0):
        raise ValueError(f"invalid judge score {score!r}: {content}")
    label = str(parsed.get("label", "")).lower()
    if label not in {"correct", "partial", "incorrect"}:
        label = score_to_label(float(score))
    return {
        "judge_status": "ok",
        "score": float(score),
        "label": label,
        "reason": str(parsed.get("reason", "")),
        "raw_judge_response": content,
    }


def user_prompt(item: JudgeItem) -> str:
    return (
        f"Dataset: {item.dataset}\n"
        f"Question:\n{item.question}\n\n"
        f"Golden answer:\n{item.gold_answer}\n\n"
        f"Model answer:\n{item.model_answer}\n\n"
        "Judge whether the model answer correctly answers the question according "
        "to the golden answer."
    )


def post_json(
    *,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach judge endpoint: {exc}") from exc
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("judge response is not a JSON object")
    return decoded


def extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"judge response has no choices: {response}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(f"judge response has no message: {response}")
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"judge response has empty content: {response}")
    return str(content)


def parse_judge_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"judge did not return JSON: {content}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"judge JSON is not an object: {content}")
    return parsed


def score_to_label(score: float) -> str:
    if score >= 1.0:
        return "correct"
    if score >= 0.5:
        return "partial"
    return "incorrect"


def output_row(item: JudgeItem, result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "row_index": item.row_index,
        "dataset": item.dataset,
        "method": item.method,
        "query_id": item.query_id,
        "task_id": item.task_id,
        "question": item.question,
        "gold_answer": item.gold_answer,
        "model_answer": item.model_answer,
        **result,
        "trace": {
            "mode": item.trace_row.get("mode"),
            "coverage": item.trace_row.get("coverage"),
            "execution_path": item.trace_row.get("execution_path"),
            "engine_ttft_ms": item.trace_row.get("engine_ttft_ms"),
            "selected_page_key": item.trace_row.get("selected_page_key"),
            "selected_page_source": item.trace_row.get("selected_page_source"),
            "selected_page_root_source": item.trace_row.get("selected_page_root_source"),
            "read_amplification": item.trace_row.get("read_amplification"),
        },
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("judge_status") == "ok"]
    scores = [float(row["score"]) for row in ok_rows if row.get("score") is not None]
    labels: dict[str, int] = {}
    by_task: dict[str, list[float]] = {}
    for row in ok_rows:
        labels[str(row.get("label"))] = labels.get(str(row.get("label")), 0) + 1
        if row.get("score") is not None:
            task_id = str(row.get("task_id"))
            by_task.setdefault(task_id, []).append(float(row["score"]))
    task_scores = {
        task_id: {
            "rows": len(values),
            "score_mean": sum(values) / len(values),
            "correct_rate": sum(1 for value in values if value >= 1.0) / len(values),
        }
        for task_id, values in sorted(by_task.items())
        if values
    }
    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "error_rows": len(rows) - len(ok_rows),
        "score_mean": sum(scores) / len(scores) if scores else 0.0,
        "correct_rate": sum(1 for value in scores if value >= 1.0) / len(scores)
        if scores
        else 0.0,
        "partial_or_correct_rate": sum(1 for value in scores if value >= 0.5) / len(scores)
        if scores
        else 0.0,
        "labels": labels,
        "tasks": task_scores,
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("judge_status") == "ok" and row.get("query_id") is not None:
            existing[str(row["query_id"])] = row
    return existing


def dedupe_by_query_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        query_id = row.get("query_id")
        if query_id is None:
            passthrough.append(row)
        else:
            deduped[str(query_id)] = row
    return passthrough + list(deduped.values())


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def slug(path: Path) -> str:
    text = str(path).strip("/").replace("/", "__")
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(keep)[-180:]


if __name__ == "__main__":
    main()
