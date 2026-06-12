"""Summarize per-session speedup from autoeval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for batch in args.batches:
        rows.extend(load_batch(Path(batch), label=args.label))
    if not rows:
        raise SystemExit("no comparable session rows found")
    print_summary(rows)
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)


def load_batch(batch_dir: Path, *, label: str | None) -> list[dict[str, Any]]:
    configs = [path for path in sorted(batch_dir.glob("config_*")) if path.is_dir()]
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    config_names: dict[str, dict[str, str]] = {}
    for config_dir in configs:
        method = infer_method(config_dir.name)
        if method is None:
            continue
        group_key = infer_group_key(config_dir.name, method)
        summaries: dict[str, dict[str, Any]] = {}
        for summary_path in sorted(config_dir.glob("sessions/*/summary.json")):
            session_id = summary_path.parent.name
            summaries[session_id] = json.loads(summary_path.read_text(encoding="utf-8"))
        if summaries:
            grouped.setdefault(group_key, {})[method] = summaries
            config_names.setdefault(group_key, {})[method] = config_dir.name
    rows: list[dict[str, Any]] = []
    for group_key, by_method in sorted(grouped.items()):
        if "baseline" not in by_method:
            continue
        baseline = by_method["baseline"]
        for method, summaries in sorted(by_method.items()):
            if method == "baseline":
                continue
            common_sessions = sorted(set(baseline) & set(summaries))
            for session_id in common_sessions:
                base = baseline[session_id]
                current = summaries[session_id]
                base_ttft = _primary_ttft_mean(base)
                current_ttft = _primary_ttft_mean(current)
                rows.append(
                    {
                        "label": label or group_key,
                        "group": group_key,
                        "batch_dir": str(batch_dir),
                        "baseline_config": config_names[group_key]["baseline"],
                        "method_config": config_names[group_key][method],
                        "method": method,
                        "session_id": session_id,
                        "queries": int(current.get("queries", 0)),
                        "ttft_metric": str(
                            current.get("ttft_metric")
                            or base.get("ttft_metric")
                            or "engine_queue_to_first_token_ms"
                        ),
                        "baseline_ttft_ms": base_ttft,
                        "method_ttft_ms": current_ttft,
                        "speedup": base_ttft / current_ttft if current_ttft > 0 else None,
                        "warm_page_queries": int(current.get("warm_page_queries", 0)),
                        "fallback_queries": int(current.get("fallback_queries", 0)),
                        "full_hit_queries": int(current.get("full_hit_queries", 0)),
                        "partial_hit_queries": int(current.get("partial_hit_queries", 0)),
                        "registered_pages": int(current.get("registered_pages", 0)),
                        "used_pages": int(current.get("used_pages", 0)),
                        "residual_count_mean": float(current.get("residual_count_mean", 0.0)),
                        "read_amplification_mean": float(current.get("read_amplification_mean", 0.0)),
                    }
                )
    return rows


def _primary_ttft_mean(summary: dict[str, Any]) -> float:
    for key in ("ttft_ms_mean", "server_ttft_ms_mean", "engine_ttft_ms_mean"):
        value = summary.get(key)
        if value is not None:
            return float(value)
    raise KeyError("summary lacks ttft_ms_mean/server_ttft_ms_mean/engine_ttft_ms_mean")


def infer_method(config_name: str) -> str | None:
    for method in ("baseline", "random", "temporal", "semantic", "append"):
        if f"_{method}_" in config_name or config_name.endswith(f"_{method}"):
            return "temporal" if method == "append" else method
    return None


def infer_group_key(config_name: str, method: str) -> str:
    if config_name.startswith("config_"):
        parts = config_name.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            config_name = parts[2]
    for token in (method, "append" if method == "temporal" else method):
        config_name = config_name.replace(f"_{token}_", "_")
        if config_name.endswith(f"_{token}"):
            config_name = config_name[: -len(token) - 1]
    return config_name


def print_summary(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["label"], row["group"], row["method"]), []).append(row)
    for (label, group_key, method), group in sorted(groups.items()):
        speedups = sorted(float(row["speedup"]) for row in group if row["speedup"] is not None)
        if not speedups:
            continue
        baseline_time = sum(float(row["baseline_ttft_ms"]) * int(row["queries"]) for row in group)
        method_time = sum(float(row["method_ttft_ms"]) * int(row["queries"]) for row in group)
        query_count = sum(int(row["queries"]) for row in group)
        pooled_speedup = baseline_time / method_time if method_time > 0 else None
        print(f"\n{label} {group_key} {method}: sessions={len(speedups)}")
        print(
            "  speedup "
            f"pooled={pooled_speedup:.3f} "
            f"mean_session={statistics.fmean(speedups):.3f} "
            f"median={percentile(speedups, 50):.3f} "
            f"p10={percentile(speedups, 10):.3f} "
            f"p90={percentile(speedups, 90):.3f} "
            f"min={speedups[0]:.3f} "
            f"max={speedups[-1]:.3f}"
        )
        print(
            "  pooled_ttft "
            f"baseline={baseline_time / query_count:.1f}ms "
            f"method={method_time / query_count:.1f}ms "
            f"queries={query_count}"
        )
        top = sorted(group, key=lambda row: float(row["speedup"]), reverse=True)[:3]
        bottom = sorted(group, key=lambda row: float(row["speedup"]))[:3]
        print("  best  " + ", ".join(f"{r['session_id']}={float(r['speedup']):.2f}x" for r in top))
        print("  worst " + ", ".join(f"{r['session_id']}={float(r['speedup']):.2f}x" for r in bottom))


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "group",
        "method",
        "session_id",
        "queries",
        "ttft_metric",
        "speedup",
        "baseline_ttft_ms",
        "method_ttft_ms",
        "warm_page_queries",
        "fallback_queries",
        "full_hit_queries",
        "partial_hit_queries",
        "registered_pages",
        "used_pages",
        "residual_count_mean",
        "read_amplification_mean",
        "batch_dir",
        "baseline_config",
        "method_config",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("batches", nargs="+", help="autoeval batch directories")
    parser.add_argument("--label", default=None, help="Label applied to all input batches")
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
