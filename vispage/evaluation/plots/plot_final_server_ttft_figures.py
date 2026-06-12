"""Generate final evaluation figures from server-side TTFT runs.

This script intentionally does not depend on the older session_speedups CSVs,
because those were produced before the server-side TTFT metric became the
paper-facing latency metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path("paper_results/figures")


@dataclass(frozen=True)
class RunPair:
    model: str
    workload: str
    baseline_dir: Path
    semantic_dir: Path


RUN_PAIRS = [
    RunPair(
        model="8B",
        workload="LoCoMO",
        baseline_dir=Path(
            "paper_results/evaluation/exp1_8b_server_ttft_quick/locomo_8000/autoeval/20260611_085418/"
            "config_000_locomo_baseline_8000"
        ),
        semantic_dir=Path(
            "paper_results/evaluation/exp1_8b_server_ttft_quick/locomo_8000/autoeval/20260611_085418/"
            "config_001_locomo_semantic_8000"
        ),
    ),
    RunPair(
        model="8B",
        workload="EventQA",
        baseline_dir=Path(
            "paper_results/evaluation/exp1_8b_server_ttft_quick/eventqa_8001/autoeval/20260611_092311/"
            "config_000_eventqa_baseline_8001"
        ),
        semantic_dir=Path(
            "paper_results/evaluation/exp1_8b_server_ttft_quick/eventqa_8001/autoeval/20260611_092311/"
            "config_001_eventqa_semantic_8001"
        ),
    ),
    RunPair(
        model="8B",
        workload="PERMA",
        baseline_dir=Path(
            "paper_results/evaluation/exp1_8b_server_ttft_quick/perma_8002/autoeval/20260611_092319/"
            "config_000_perma_baseline_8002"
        ),
        semantic_dir=Path(
            "paper_results/evaluation/exp1_8b_server_ttft_quick/perma_8002/autoeval/20260611_092319/"
            "config_001_perma_semantic_8002"
        ),
    ),
    RunPair(
        model="32B",
        workload="LoCoMO",
        baseline_dir=Path(
            "paper_results/evaluation/exp1_32b_server_ttft_quick/locomo_8010/autoeval/20260611_140039/"
            "config_000_locomo_baseline_8010"
        ),
        semantic_dir=Path(
            "paper_results/evaluation/exp1_32b_server_ttft_quick/locomo_8010/autoeval/20260611_140039/"
            "config_001_locomo_semantic_8010"
        ),
    ),
    RunPair(
        model="32B",
        workload="EventQA",
        baseline_dir=Path(
            "paper_results/evaluation/exp1_32b_server_ttft_quick/eventqa_8011/autoeval/20260611_140048/"
            "config_000_eventqa_baseline_8011"
        ),
        semantic_dir=Path(
            "paper_results/evaluation/exp1_32b_server_ttft_quick/eventqa_8011/autoeval/20260611_140048/"
            "config_001_eventqa_semantic_8011"
        ),
    ),
    RunPair(
        model="32B",
        workload="PERMA",
        baseline_dir=Path(
            "paper_results/evaluation/exp1_32b_server_ttft_quick/perma_8012/autoeval/20260611_140101/"
            "config_000_perma_baseline_8012"
        ),
        semantic_dir=Path(
            "paper_results/evaluation/exp1_32b_server_ttft_quick/perma_8012/autoeval/20260611_140101/"
            "config_001_perma_semantic_8012"
        ),
    ),
]


RANDOM_8B_LOCOMO = Path(
    "paper_results/evaluation/exp1_8b_server_ttft_quick/autoeval/20260611_141456/"
    "config_000_locomo_random_8013"
)

AMP_DIR = Path("paper_results/evaluation/sensitivity_8b_locomo_quick/amp_8020/autoeval/20260611_184202")
CACHE_UTIL_DIR = Path("paper_results/evaluation/sensitivity_8b_locomo_quick/cache_util")

WORKLOAD_ORDER = ["LoCoMO", "EventQA", "PERMA"]
MODEL_ORDER = ["8B", "32B"]
PATH_ORDER = ["fallback", "partial_hit", "full_hit"]
PATH_LABELS = {"fallback": "Fallback", "partial_hit": "Partial", "full_hit": "Full"}


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session_rows, aggregate_rows = collect_main_results(RUN_PAIRS)
    reuse_ratio, reuse_ttft = collect_reuse_breakdown(RUN_PAIRS)
    random_rows, random_path_rows = collect_random_ablation()
    amp_rows, cache_rows = collect_sensitivity()

    write_csv(pd.DataFrame(session_rows), args.output_dir / "paper_session_speedups_data.csv")
    write_csv(pd.DataFrame(aggregate_rows), args.output_dir / "paper_main_results_data.csv")
    write_csv(reuse_ratio, args.output_dir / "paper_reuse_path_ratio_data.csv")
    write_csv(reuse_ttft, args.output_dir / "paper_reuse_path_ttft_data.csv")
    write_csv(random_rows, args.output_dir / "paper_random_layout_data.csv")
    write_csv(random_path_rows, args.output_dir / "paper_random_layout_path_data.csv")
    write_csv(amp_rows, args.output_dir / "paper_sensitivity_amp_data.csv")
    write_csv(cache_rows, args.output_dir / "paper_sensitivity_cache_data.csv")

    plot_paper_main_and_distribution(
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(session_rows),
        args.output_dir / "paper_main_results_and_distribution.pdf",
    )
    plot_paper_reuse_breakdown_double(
        reuse_ratio,
        reuse_ttft,
        args.output_dir / "paper_reuse_breakdown_double.pdf",
    )
    plot_paper_random_ablation(
        random_rows,
        random_path_rows,
        args.output_dir / "paper_random_layout_single.pdf",
    )
    plot_paper_sensitivity(
        amp_rows,
        cache_rows,
        args.output_dir / "paper_sensitivity_single.pdf",
    )

    print(f"wrote figures and CSVs to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def collect_main_results(run_pairs: list[RunPair]) -> tuple[list[dict], list[dict]]:
    session_rows = []
    aggregate_rows = []

    for pair in run_pairs:
        baseline = load_session_summaries(pair.baseline_dir)
        semantic = load_session_summaries(pair.semantic_dir)
        common = sorted(set(baseline) & set(semantic))
        if not common:
            raise RuntimeError(f"no common sessions for {pair}")

        for session_id in common:
            b = baseline[session_id]
            s = semantic[session_id]
            queries = int(s["queries"])
            b_ttft = metric_mean(b)
            s_ttft = metric_mean(s)
            session_rows.append(
                {
                    "model": pair.model,
                    "workload": pair.workload,
                    "session_id": session_id,
                    "queries": queries,
                    "baseline_server_ttft_ms": b_ttft,
                    "vispage_server_ttft_ms": s_ttft,
                    "speedup": b_ttft / s_ttft,
                    "fallback_queries": int(s.get("fallback_queries", 0)),
                    "partial_hit_queries": int(s.get("partial_hit_queries", 0)),
                    "full_hit_queries": int(s.get("full_hit_queries", 0)),
                    "submitted_pages": int(s.get("submitted_pages", s.get("registered_pages", 0))),
                    "used_pages": int(s.get("used_pages", 0)),
                    "source_baseline": str(pair.baseline_dir),
                    "source_vispage": str(pair.semantic_dir),
                }
            )

        q = np.array([int(semantic[s]["queries"]) for s in common], dtype=float)
        baseline_ttft = np.array([metric_mean(baseline[s]) for s in common], dtype=float)
        vispage_ttft = np.array([metric_mean(semantic[s]) for s in common], dtype=float)
        aggregate_rows.append(
            {
                "model": pair.model,
                "workload": pair.workload,
                "sessions": len(common),
                "queries": int(q.sum()),
                "baseline_server_ttft_ms": weighted_mean(baseline_ttft, q),
                "vispage_server_ttft_ms": weighted_mean(vispage_ttft, q),
                "speedup": weighted_mean(baseline_ttft, q) / weighted_mean(vispage_ttft, q),
                "fallback_queries": sum(int(semantic[s].get("fallback_queries", 0)) for s in common),
                "partial_hit_queries": sum(int(semantic[s].get("partial_hit_queries", 0)) for s in common),
                "full_hit_queries": sum(int(semantic[s].get("full_hit_queries", 0)) for s in common),
                "submitted_pages": sum(int(semantic[s].get("submitted_pages", semantic[s].get("registered_pages", 0))) for s in common),
                "used_pages": sum(int(semantic[s].get("used_pages", 0)) for s in common),
                "source_baseline": str(pair.baseline_dir),
                "source_vispage": str(pair.semantic_dir),
            }
        )

    return session_rows, aggregate_rows


def collect_reuse_breakdown(run_pairs: list[RunPair]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio_rows = []
    ttft_rows = []

    for pair in run_pairs:
        baseline_sessions = set(load_session_summaries(pair.baseline_dir))
        semantic_sessions = load_session_summaries(pair.semantic_dir)
        common = sorted(baseline_sessions & set(semantic_sessions))
        trace_values: dict[str, list[float]] = {path: [] for path in PATH_ORDER}
        counts = {path: 0 for path in PATH_ORDER}
        total = 0

        for session_id in common:
            trace_path = pair.semantic_dir / "sessions" / session_id / "trace.jsonl"
            for trace in iter_jsonl(trace_path):
                path = path_category(trace)
                if path not in counts:
                    continue
                counts[path] += 1
                total += 1
                value = trace.get("server_ttft_ms", trace.get("ttft_ms", trace.get("engine_ttft_ms")))
                if value is not None:
                    trace_values[path].append(float(value))

        for path in PATH_ORDER:
            ratio_rows.append(
                {
                    "model": pair.model,
                    "workload": pair.workload,
                    "path": path,
                    "path_label": PATH_LABELS[path],
                    "queries": total,
                    "path_queries": counts[path],
                    "path_ratio": counts[path] / total if total else math.nan,
                }
            )
            values = trace_values[path]
            ttft_rows.append(
                {
                    "model": pair.model,
                    "workload": pair.workload,
                    "path": path,
                    "path_label": PATH_LABELS[path],
                    "samples": len(values),
                    "server_ttft_ms_mean": fmean(values) if values else math.nan,
                    "server_ttft_ms_p50": percentile(values, 50),
                    "server_ttft_ms_p90": percentile(values, 90),
                }
            )

    return pd.DataFrame(ratio_rows), pd.DataFrame(ttft_rows)


def collect_random_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    locomo_pair = next(pair for pair in RUN_PAIRS if pair.model == "8B" and pair.workload == "LoCoMO")
    baseline = load_session_summaries(locomo_pair.baseline_dir)
    semantic = load_session_summaries(locomo_pair.semantic_dir)
    random = load_session_summaries(RANDOM_8B_LOCOMO)
    common = sorted(set(baseline) & set(semantic) & set(random))
    if not common:
        raise RuntimeError("no common sessions for random ablation")

    method_dirs = {"Random": RANDOM_8B_LOCOMO, "VISPAGE": locomo_pair.semantic_dir}
    summaries = {"Random": random, "VISPAGE": semantic}
    rows = []
    path_rows = []

    for method, current in summaries.items():
        q = np.array([int(current[s]["queries"]) for s in common], dtype=float)
        b_ttft = np.array([metric_mean(baseline[s]) for s in common], dtype=float)
        m_ttft = np.array([metric_mean(current[s]) for s in common], dtype=float)
        submitted = sum(int(current[s].get("submitted_pages", current[s].get("registered_pages", 0))) for s in common)
        used = sum(int(current[s].get("used_pages", 0)) for s in common)
        rows.append(
            {
                "method": method,
                "sessions": len(common),
                "queries": int(q.sum()),
                "baseline_server_ttft_ms": weighted_mean(b_ttft, q),
                "method_server_ttft_ms": weighted_mean(m_ttft, q),
                "speedup": weighted_mean(b_ttft, q) / weighted_mean(m_ttft, q),
                "submitted_pages": submitted,
                "used_pages": used,
                "wasted_page_rate": 1.0 - used / submitted if submitted else math.nan,
                "source": str(method_dirs[method]),
            }
        )
        counts = {"fallback": 0, "partial_hit": 0, "full_hit": 0}
        total = 0
        for session_id in common:
            for trace in iter_jsonl(method_dirs[method] / "sessions" / session_id / "trace.jsonl"):
                path = path_category(trace)
                if path in counts:
                    counts[path] += 1
                    total += 1
        for path in PATH_ORDER:
            path_rows.append(
                {
                    "method": method,
                    "path": path,
                    "path_label": PATH_LABELS[path],
                    "queries": total,
                    "path_queries": counts[path],
                    "path_ratio": counts[path] / total if total else math.nan,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(path_rows)


def collect_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_summary = load_json(
        RUN_PAIRS[0].baseline_dir / "sessions" / "locomo-conv-26" / "summary.json"
    )
    amp5_summary = load_json(
        RUN_PAIRS[0].semantic_dir / "sessions" / "locomo-conv-26" / "summary.json"
    )
    baseline_ttft = metric_mean(baseline_summary)
    amp_rows = []
    for path in sorted(AMP_DIR.glob("config_*/aggregate_summary.json")):
        summary = load_json(path)
        cfg = load_json(path.parent / "run_config.json")
        amp = float(cfg["page"]["max_amplification"])
        amp_rows.append(sensitivity_row("amp", amp, summary, baseline_ttft, path.parent))

    amp_rows.append(sensitivity_row("amp", 5.0, amp5_summary, baseline_ttft, RUN_PAIRS[0].semantic_dir))
    amp_rows = pd.DataFrame(amp_rows).sort_values("value").reset_index(drop=True)

    cache_rows = []
    for path in sorted(CACHE_UTIL_DIR.glob("util_*/autoeval/*/config_*/aggregate_summary.json")):
        summary = load_json(path)
        cfg = load_json(path.parent / "run_config.json")
        util = float(cfg.get("metadata", {}).get("sensitivity_value"))
        cache_rows.append(sensitivity_row("cache_util", util, summary, baseline_ttft, path.parent))
    cache_rows.append(sensitivity_row("cache_util", 0.9, amp5_summary, baseline_ttft, RUN_PAIRS[0].semantic_dir))
    cache_rows = pd.DataFrame(cache_rows).sort_values("value").reset_index(drop=True)

    return amp_rows, cache_rows


def sensitivity_row(axis: str, value: float, summary: dict, baseline_ttft: float, source: Path) -> dict:
    ttft = metric_mean(summary)
    submitted = int(summary.get("submitted_pages", summary.get("registered_pages", 0)))
    used = int(summary.get("used_pages", 0))
    return {
        "axis": axis,
        "value": value,
        "queries": int(summary["queries"]),
        "server_ttft_ms": ttft,
        "speedup": baseline_ttft / ttft,
        "warm_page_queries": int(summary.get("warm_page_queries", 0)),
        "fallback_queries": int(summary.get("fallback_queries", 0)),
        "full_hit_queries": int(summary.get("full_hit_queries", 0)),
        "partial_hit_queries": int(summary.get("partial_hit_queries", 0)),
        "submitted_pages": submitted,
        "used_pages": used,
        "wasted_page_rate": 1.0 - used / submitted if submitted else math.nan,
        "source": str(source),
    }


def plot_main_results(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), sharey=True)
    baseline_color = "#B8C0CC"
    vispage_color = "#2F6F9F"
    speed_color = "#B33A3A"
    hatches = {"Baseline": "///", "VISPAGE": "\\\\\\"}
    width = 0.34
    ymax = round_up(max(data["baseline_server_ttft_ms"].max(), data["vispage_server_ttft_ms"].max()) * 1.14, 500)
    symax = max(2.4, data["speedup"].max() * 1.18)

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        sub = data[data["model"] == model].set_index("workload").loc[WORKLOAD_ORDER].reset_index()
        x = np.arange(len(sub))
        hatched_bar(ax, x - width / 2, sub["baseline_server_ttft_ms"], width, baseline_color, hatches["Baseline"], "Baseline")
        hatched_bar(ax, x + width / 2, sub["vispage_server_ttft_ms"], width, vispage_color, hatches["VISPAGE"], "VISPAGE")
        ax.set_title(f"Qwen3-VL-{model}")
        ax.set_xticks(x, WORKLOAD_ORDER)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.55)
        clean_axis(ax)
        twin = ax.twinx()
        twin.plot(x, sub["speedup"], "o", markersize=4.0, color=speed_color, markeredgecolor="#7D1D1D", markeredgewidth=0.35)
        twin.set_ylim(0.8, symax)
        twin.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
        twin.set_yticklabels([f"{v:.2g}x" if v != 1.0 else "1.0x" for v in [1.0, 1.25, 1.5, 1.75, 2.0]])
        twin.tick_params(axis="y", colors=speed_color, width=0.55, length=2.4)
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_color(speed_color)
        if ax is axes[0]:
            ax.set_ylabel("TTFT (ms)")
        else:
            twin.set_ylabel("Speedup", color=speed_color)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig.tight_layout(pad=0.35, w_pad=0.9, rect=(0, 0, 1, 0.91))
    savefig(fig, output)


def plot_session_distribution(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.05), sharey=True)
    colors = {"LoCoMO": "#2F6F9F", "EventQA": "#6A7FDB", "PERMA": "#6C8E3F"}
    rng = np.random.default_rng(11)

    ymax = max(2.4, data["speedup"].max() * 1.15)
    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        sub = data[data["model"] == model]
        for i, workload in enumerate(WORKLOAD_ORDER):
            values = sub[sub["workload"] == workload]["speedup"].to_numpy(float)
            x = np.full(len(values), i, dtype=float) + rng.uniform(-0.10, 0.10, len(values))
            ax.scatter(x, values, s=18, color=colors[workload], edgecolor="white", linewidth=0.25, alpha=0.9)
            if len(values):
                ax.hlines(np.median(values), i - 0.22, i + 0.22, color="#1C1C1C", linewidth=1.05)
        ax.axhline(1.0, color="#8D9299", linewidth=0.75, linestyle="--")
        ax.set_title(f"Qwen3-VL-{model}")
        ax.set_xticks(range(len(WORKLOAD_ORDER)), WORKLOAD_ORDER)
        ax.set_ylim(0.65, ymax)
        ax.grid(axis="y", color="#E0E5EB", linewidth=0.5)
        clean_axis(ax)
    axes[0].set_ylabel("Session speedup")
    fig.tight_layout(pad=0.35, w_pad=0.8)
    savefig(fig, output)


def plot_reuse_breakdown(ratio: pd.DataFrame, ttft: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.0))
    colors = {"fallback": "#B75B55", "partial_hit": "#3F78A8", "full_hit": "#5D8B54"}
    hatches = {"fallback": "///", "partial_hit": "\\\\\\", "full_hit": "---"}
    width = 0.24

    for col, model in enumerate(MODEL_ORDER):
        draw_path_ratio_axis(axes[col * 2], ratio[ratio["model"] == model], colors, hatches, width, f"{model} paths")
        draw_path_ttft_axis(axes[col * 2 + 1], ttft[ttft["model"] == model], colors, hatches, width, f"{model} TTFT")
    axes[0].set_ylabel("Query ratio")
    axes[1].set_ylabel("TTFT (ms)")
    axes[3].set_ylabel("TTFT (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(pad=0.35, w_pad=0.6, rect=(0, 0, 1, 0.93))
    savefig(fig, output)


def draw_path_ratio_axis(ax: plt.Axes, data: pd.DataFrame, colors: dict, hatches: dict, width: float, title: str) -> None:
    x = np.arange(len(WORKLOAD_ORDER))
    bottom = np.zeros(len(WORKLOAD_ORDER))
    matrix = data.set_index(["workload", "path"])
    for path in PATH_ORDER:
        values = np.array([matrix.loc[(w, path), "path_ratio"] for w in WORKLOAD_ORDER], dtype=float)
        bars = ax.bar(x, values, width=0.58, bottom=bottom, color=colors[path], edgecolor="white", linewidth=0.0, label=PATH_LABELS[path])
        for bar in bars:
            bar.set_hatch(hatches[path])
            bar.set_edgecolor("white")
            bar.set_linewidth(0.0)
        bottom += values
    ax.set_title(title)
    ax.set_xticks(x, WORKLOAD_ORDER)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color="#E0E5EB", linewidth=0.5)
    clean_axis(ax)


def draw_path_ttft_axis(ax: plt.Axes, data: pd.DataFrame, colors: dict, hatches: dict, width: float, title: str) -> None:
    x = np.arange(len(WORKLOAD_ORDER))
    matrix = data.set_index(["workload", "path"])
    offsets = np.linspace(-width, width, len(PATH_ORDER))
    max_value = 0.0
    for offset, path in zip(offsets, PATH_ORDER, strict=True):
        values = np.array([matrix.loc[(w, path), "server_ttft_ms_mean"] for w in WORKLOAD_ORDER], dtype=float)
        max_value = max(max_value, np.nanmax(values))
        hatched_bar(ax, x + offset, values, width * 0.82, colors[path], hatches[path], PATH_LABELS[path])
    ax.set_title(title)
    ax.set_xticks(x, WORKLOAD_ORDER)
    ax.set_ylim(0, round_up(max_value * 1.18, 500))
    ax.grid(axis="y", color="#E0E5EB", linewidth=0.5)
    clean_axis(ax)


def plot_random_ablation(rows: pd.DataFrame, path_rows: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.88))
    method_order = ["Random", "VISPAGE"]
    colors = {"Random": "#8F97A5", "VISPAGE": "#2F6F9F"}
    hatches = {"Random": "///", "VISPAGE": "\\\\\\"}
    x = np.arange(len(method_order))
    rows = rows.set_index("method").loc[method_order].reset_index()
    hatched_bar(axes[0], x, rows["speedup"], 0.58, [colors[m] for m in method_order], [hatches[m] for m in method_order], None)
    axes[0].axhline(1.0, color="#8D9299", linestyle="--", linewidth=0.7)
    axes[0].set_xticks(x, ["Ran.", "Sem."])
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Layout speedup")
    axes[0].set_ylim(0, max(1.8, rows["speedup"].max() * 1.2))
    axes[0].grid(axis="y", color="#E0E5EB", linewidth=0.5)
    clean_axis(axes[0])

    path_colors = {"fallback": "#B75B55", "partial_hit": "#3F78A8", "full_hit": "#5D8B54"}
    path_hatches = {"fallback": "///", "partial_hit": "\\\\\\", "full_hit": "---"}
    matrix = path_rows.set_index(["method", "path"])
    bottom = np.zeros(len(method_order))
    for path in PATH_ORDER:
        vals = np.array([matrix.loc[(m, path), "path_ratio"] for m in method_order], dtype=float)
        bars = axes[1].bar(x, vals, width=0.58, bottom=bottom, color=path_colors[path], edgecolor="white", linewidth=0, label=PATH_LABELS[path])
        for bar in bars:
            bar.set_hatch(path_hatches[path])
            bar.set_edgecolor("white")
        bottom += vals
    axes[1].set_xticks(x, ["Ran.", "Sem."])
    axes[1].set_title("Reuse paths")
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([0, 0.5, 1.0])
    axes[1].grid(axis="y", color="#E0E5EB", linewidth=0.5)
    clean_axis(axes[1])
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=3, handlelength=1.0, columnspacing=0.6)
    fig.tight_layout(pad=0.25, w_pad=0.55)
    savefig(fig, output)


def plot_sensitivity(amp: pd.DataFrame, cache: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.05))
    line_color = "#2F6F9F"
    bar_color = "#B8C0CC"
    speed_color = "#B33A3A"
    for ax, data, title, xlabel in [
        (axes[0], amp, "Amplification budget", "Max amplification"),
        (axes[1], cache, "Cache budget", "GPU memory utilization"),
    ]:
        x = np.arange(len(data))
        labels = [format_value(v) for v in data["value"]]
        hatched_bar(ax, x, data["server_ttft_ms"], 0.58, bar_color, "///", "TTFT")
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TTFT (ms)")
        ax.set_ylim(0, round_up(float(data["server_ttft_ms"].max()) * 1.18, 500))
        ax.grid(axis="y", color="#E0E5EB", linewidth=0.5)
        clean_axis(ax)
        twin = ax.twinx()
        twin.plot(x, data["speedup"], marker="o", markersize=3.8, color=speed_color, linewidth=1.0)
        twin.set_ylabel("Speedup", color=speed_color)
        twin.tick_params(axis="y", colors=speed_color, width=0.55, length=2.4)
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_color(speed_color)
        twin.set_ylim(0.9, max(1.8, float(data["speedup"].max()) * 1.15))
    fig.tight_layout(pad=0.35, w_pad=1.0)
    savefig(fig, output)


def plot_paper_main_and_distribution(aggregate: pd.DataFrame, sessions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.05, 1.48),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0]},
    )
    grey = "#B8C0CC"
    blue = "#2F6F9F"
    light_blue = "#8DB9E2"
    mid_blue = "#4F86B8"
    dark_blue = "#174A7C"
    speed_red = "#B33A3A"
    workload_colors = {"LoCoMO": dark_blue, "EventQA": mid_blue, "PERMA": light_blue}
    speed_axes = []

    latency_ymax = round_up(max(aggregate["baseline_server_ttft_ms"].max(), aggregate["vispage_server_ttft_ms"].max()) * 1.16, 500)
    speed_ymax = max(2.25, aggregate["speedup"].max() * 1.14)
    session_ymax = max(3.2, sessions["speedup"].max() * 1.12)

    for ax, model in zip(axes[:2], MODEL_ORDER, strict=True):
        sub = aggregate[aggregate["model"] == model].set_index("workload").loc[WORKLOAD_ORDER].reset_index()
        x = np.arange(len(sub))
        width = 0.34
        hatched_bar(ax, x - width / 2, sub["baseline_server_ttft_ms"], width, grey, "///", "Baseline")
        hatched_bar(ax, x + width / 2, sub["vispage_server_ttft_ms"], width, blue, "\\\\\\", "VISPAGE")
        ax.set_title(f"{model} latency", pad=2)
        ax.set_xticks(x, ["LCM", "EQA", "PMA"])
        ax.set_ylim(0, latency_ymax)
        ax.grid(axis="y", color="#E2E7ED", linewidth=0.5)
        clean_axis(ax)
        if model == "8B":
            ax.set_ylabel("TTFT (ms)")

        twin = ax.twinx()
        twin.plot(x, sub["speedup"], "o", color=speed_red, markeredgecolor="#7D1D1D", markeredgewidth=0.25, markersize=3.2)
        twin.set_ylim(0.85, speed_ymax)
        twin.tick_params(axis="y", colors=speed_red, width=0.5, length=2.2)
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_color(speed_red)
        if model == "32B":
            twin.set_ylabel("Speedup", color=speed_red, labelpad=2)
        else:
            twin.tick_params(axis="y", right=False, labelright=False)
            twin.spines["right"].set_visible(False)
        speed_axes.append(twin)

    rng = np.random.default_rng(11)
    for ax, model in zip(axes[2:], MODEL_ORDER, strict=True):
        sub = sessions[sessions["model"] == model]
        for pos, workload in enumerate(WORKLOAD_ORDER):
            vals = sub[sub["workload"] == workload]["speedup"].to_numpy(float)
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=13,
                color=workload_colors[workload],
                edgecolor="white",
                linewidth=0.22,
                alpha=0.9,
            )
            if len(vals):
                ax.hlines(np.median(vals), pos - 0.18, pos + 0.18, color="#1D2935", linewidth=0.95)
        ax.axhline(1.0, color="#8D9299", linestyle="--", linewidth=0.65)
        ax.set_title(f"{model} sessions", pad=2)
        ax.set_xticks(range(len(WORKLOAD_ORDER)), ["LCM", "EQA", "PMA"])
        ax.set_ylim(0.65, session_ymax)
        ax.grid(axis="y", color="#E2E7ED", linewidth=0.5)
        clean_axis(ax)
    axes[2].set_ylabel("Session speedup", labelpad=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.30, 1.05),
        handlelength=1.4,
        columnspacing=1.2,
    )
    panel_width = 0.172
    panel_bottom = 0.20
    panel_height = 0.55
    gap_inside = 0.035
    gap_middle = 0.130
    x0 = 0.065
    positions = [
        x0,
        x0 + panel_width + gap_inside,
        x0 + 2 * panel_width + gap_inside + gap_middle,
        x0 + 3 * panel_width + 2 * gap_inside + gap_middle,
    ]
    for ax, left in zip(axes, positions, strict=True):
        ax.set_position([left, panel_bottom, panel_width, panel_height])
    for twin, left in zip(speed_axes, positions[:2], strict=True):
        twin.set_position([left, panel_bottom, panel_width, panel_height])
    savefig(fig, output)


def plot_paper_reuse_breakdown(ratio: pd.DataFrame, ttft: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.0), gridspec_kw={"height_ratios": [1.0, 1.05]})
    colors = {"fallback": "#B8C0CC", "partial_hit": "#4F86B8", "full_hit": "#174A7C"}
    hatches = {"fallback": "///", "partial_hit": "\\\\\\", "full_hit": "---"}
    labels = paper_case_labels()

    ratio_matrix = ratio.set_index(["model", "workload", "path"])
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for path in PATH_ORDER:
        vals = np.array([ratio_matrix.loc[(m, w, path), "path_ratio"] for m, w in paper_cases()], dtype=float)
        bars = axes[0].bar(x, vals, width=0.62, bottom=bottom, color=colors[path], edgecolor="white", linewidth=0, label=PATH_LABELS[path])
        for bar in bars:
            bar.set_hatch(hatches[path])
            bar.set_edgecolor("white")
        bottom += vals
    axes[0].set_title("Reuse path ratio")
    axes[0].set_ylabel("Ratio")
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([0, 0.5, 1.0])
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", color="#E2E7ED", linewidth=0.5)
    clean_axis(axes[0])

    ttft_matrix = ttft.set_index(["model", "workload", "path"])
    width = 0.20
    offsets = [-width, 0.0, width]
    max_value = 0.0
    for offset, path in zip(offsets, PATH_ORDER, strict=True):
        vals = np.array([ttft_matrix.loc[(m, w, path), "server_ttft_ms_mean"] for m, w in paper_cases()], dtype=float)
        max_value = max(max_value, float(np.nanmax(vals)))
        hatched_bar(axes[1], x + offset, vals, width * 0.9, colors[path], hatches[path], PATH_LABELS[path])
    axes[1].set_title("Reuse path latency")
    axes[1].set_ylabel("TTFT (ms)")
    axes[1].set_ylim(0, round_up(max_value * 1.18, 500))
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", color="#E2E7ED", linewidth=0.5)
    clean_axis(axes[1])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.1, columnspacing=0.8)
    fig.tight_layout(pad=0.25, h_pad=0.65, rect=(0, 0, 1, 0.94))
    savefig(fig, output)


def plot_paper_reuse_breakdown_double(ratio: pd.DataFrame, ttft: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.05, 1.48),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0]},
    )
    colors = {"fallback": "#B8C0CC", "partial_hit": "#4F86B8", "full_hit": "#174A7C"}
    hatches = {"fallback": "///", "partial_hit": "\\\\\\", "full_hit": "---"}
    x = np.arange(len(WORKLOAD_ORDER))
    labels = ["LCM", "EQA", "PMA"]

    ratio_matrix = ratio.set_index(["model", "workload", "path"])
    ttft_matrix = ttft.set_index(["model", "workload", "path"])
    ttft_ymax = round_up(float(ttft["server_ttft_ms_mean"].max()) * 1.16, 1000)

    for model, ratio_ax, ttft_ax in [
        ("8B", axes[0], axes[1]),
        ("32B", axes[2], axes[3]),
    ]:
        bottom = np.zeros(len(WORKLOAD_ORDER))
        for path in PATH_ORDER:
            vals = np.array([ratio_matrix.loc[(model, w, path), "path_ratio"] for w in WORKLOAD_ORDER], dtype=float)
            bars = ratio_ax.bar(
                x,
                vals,
                width=0.58,
                bottom=bottom,
                color=colors[path],
                edgecolor="white",
                linewidth=0,
                label=PATH_LABELS[path],
            )
            for bar in bars:
                bar.set_hatch(hatches[path])
                bar.set_edgecolor("white")
            bottom += vals
        ratio_ax.set_title(f"{model} ratio", pad=2)
        ratio_ax.set_ylim(0, 1)
        ratio_ax.set_yticks([0, 0.5, 1.0])
        ratio_ax.set_xticks(x, labels)
        ratio_ax.grid(axis="y", color="#E2E7ED", linewidth=0.45)
        clean_axis(ratio_ax)

        width = 0.20
        offsets = [-width, 0.0, width]
        for offset, path in zip(offsets, PATH_ORDER, strict=True):
            vals = np.array([ttft_matrix.loc[(model, w, path), "server_ttft_ms_mean"] for w in WORKLOAD_ORDER], dtype=float)
            hatched_bar(ttft_ax, x + offset, vals, width * 0.9, colors[path], hatches[path], PATH_LABELS[path])
        ttft_ax.set_title(f"{model} TTFT", pad=2)
        ttft_ax.set_ylim(0, ttft_ymax)
        ttft_ax.set_yticks([0, ttft_ymax / 2, ttft_ymax])
        ttft_ax.set_xticks(x, labels)
        ttft_ax.grid(axis="y", color="#E2E7ED", linewidth=0.45)
        clean_axis(ttft_ax)

    axes[0].set_ylabel("Ratio")
    axes[1].set_ylabel("TTFT (ms)")
    axes[2].tick_params(axis="y", labelleft=False)
    axes[3].tick_params(axis="y", labelleft=False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        handlelength=1.0,
        columnspacing=0.9,
        fontsize=6.6,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.20, top=0.77, wspace=0.42)
    savefig(fig, output)


def plot_paper_random_ablation(rows: pd.DataFrame, path_rows: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.48), gridspec_kw={"width_ratios": [0.86, 1.0]})
    methods = ["Random", "VISPAGE"]
    grey = "#B8C0CC"
    blue = "#2F6F9F"
    colors = {"Random": grey, "VISPAGE": blue}
    hatches = {"Random": "///", "VISPAGE": "\\\\\\"}
    rows = rows.set_index("method").loc[methods].reset_index()
    x = np.arange(len(methods))
    hatched_bar(axes[0], x, rows["speedup"], 0.58, [colors[m] for m in methods], [hatches[m] for m in methods], None)
    axes[0].axhline(1.0, color="#8D9299", linestyle="--", linewidth=0.65)
    axes[0].set_xticks(x, ["Ran.", "Sem."])
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Layout", pad=2)
    axes[0].set_ylim(0, max(1.75, rows["speedup"].max() * 1.18))
    axes[0].grid(axis="y", color="#E2E7ED", linewidth=0.5)
    clean_axis(axes[0])

    path_colors = {"fallback": grey, "partial_hit": "#4F86B8", "full_hit": "#174A7C"}
    path_hatches = {"fallback": "///", "partial_hit": "\\\\\\", "full_hit": "---"}
    matrix = path_rows.set_index(["method", "path"])
    bottom = np.zeros(len(methods))
    for path in PATH_ORDER:
        vals = np.array([matrix.loc[(m, path), "path_ratio"] for m in methods], dtype=float)
        bars = axes[1].bar(x, vals, width=0.58, bottom=bottom, color=path_colors[path], edgecolor="white", linewidth=0, label=PATH_LABELS[path])
        for bar in bars:
            bar.set_hatch(path_hatches[path])
            bar.set_edgecolor("white")
        bottom += vals
    axes[1].set_title("Paths", pad=2)
    axes[1].set_xticks(x, ["Ran.", "Sem."])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_yticks([0, 0.5, 1.0])
    axes[1].grid(axis="y", color="#E2E7ED", linewidth=0.5)
    clean_axis(axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.0,
        loc="upper center",
        bbox_to_anchor=(0.62, 1.06),
        ncol=3,
        handlelength=0.9,
        columnspacing=0.55,
    )
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.20, top=0.72, wspace=0.38)
    savefig(fig, output)


def plot_paper_sensitivity(amp: pd.DataFrame, cache: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.48), sharey=False)
    grey = "#B8C0CC"
    speed_red = "#B33A3A"
    for ax, data, title, xlabel in [
        (axes[0], amp, "Amplification budget", "Max amplification"),
        (axes[1], cache, "Cache budget", "GPU memory utilization"),
    ]:
        x = np.arange(len(data))
        labels = [format_value(v) for v in data["value"]]
        hatched_bar(ax, x, data["server_ttft_ms"], 0.58, grey, "///", "TTFT")
        ax.set_title(title, pad=2)
        ax.set_xticks(x, labels)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylim(0, round_up(float(data["server_ttft_ms"].max()) * 1.18, 500))
        ax.grid(axis="y", color="#E2E7ED", linewidth=0.5)
        clean_axis(ax)
        twin = ax.twinx()
        twin.plot(x, data["speedup"], marker="o", markersize=3.2, color=speed_red, linewidth=0.95)
        twin.tick_params(axis="y", colors=speed_red, width=0.5, length=2.2)
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_color(speed_red)
        twin.set_ylim(0.9, max(1.8, float(data["speedup"].max()) * 1.15))
        if ax is axes[1]:
            twin.set_ylabel("Speedup", color=speed_red)
        else:
            twin.tick_params(axis="y", right=False, labelright=False)
            twin.spines["right"].set_visible(False)
    axes[0].set_ylabel("TTFT (ms)")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.27, top=0.79, wspace=0.42)
    savefig(fig, output)


def paper_cases() -> list[tuple[str, str]]:
    return [(model, workload) for model in MODEL_ORDER for workload in WORKLOAD_ORDER]


def paper_case_labels() -> list[str]:
    abbr = {"LoCoMO": "LCM", "EventQA": "EQA", "PERMA": "PMA"}
    return [f"{model}\n{abbr[workload]}" for model, workload in paper_cases()]


def load_session_summaries(config_dir: Path) -> dict[str, dict]:
    summaries = {}
    for path in sorted((config_dir / "sessions").glob("*/summary.json")):
        summaries[path.parent.name] = load_json(path)
    if not summaries:
        raise RuntimeError(f"no session summaries under {config_dir}")
    return summaries


def metric_mean(summary: dict) -> float:
    return float(summary.get("server_ttft_ms_mean", summary.get("ttft_ms_mean", summary.get("engine_ttft_ms_mean"))))


def path_category(trace: dict) -> str:
    path = trace.get("execution_path")
    if path in {"fallback", "partial_hit", "full_hit"}:
        return path
    if trace.get("mode") == "baseline_cold":
        return "fallback"
    coverage = float(trace.get("coverage", 0.0) or 0.0)
    if coverage >= 0.999:
        return "full_hit"
    if coverage > 0:
        return "partial_hit"
    return "fallback"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.array(values, dtype=float), p))


def round_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def format_value(value: float) -> str:
    if value >= 1:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return f"{value:.1f}"


def configure_matplotlib() -> None:
    font_family = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.dpi": 300,
            "hatch.linewidth": 0.18,
        }
    )


def choose_serif_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def hatched_bar(ax: plt.Axes, x, heights, width, color, hatch, label):
    bars = ax.bar(x, heights, width=width, color=color, edgecolor="white", linewidth=0.0, label=label)
    if isinstance(hatch, list):
        for bar, one_hatch in zip(bars, hatch, strict=True):
            bar.set_hatch(one_hatch)
            bar.set_edgecolor("white")
    else:
        for bar in bars:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
    return bars


def savefig(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
