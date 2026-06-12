"""Plot 8B random-vs-semantic layout ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = "paper_results/figures/random_layout_8b.pdf"
DEFAULT_DATA_OUTPUT = "paper_results/figures/random_layout_8b_data.csv"

WORKLOADS = ["LoCoMO", "EventQA", "PERMA"]
WORKLOAD_ABBR = {"LoCoMO": "LCM", "EventQA": "EQA", "PERMA": "PMA"}
METHODS = ["Random", "Semantic"]
PATH_ORDER = ["fallback", "partial_hit", "full_hit"]
PATH_LABELS = {
    "fallback": "Fallback",
    "partial_hit": "Partial",
    "full_hit": "Full",
}


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    data = load_data()
    write_csv(data, Path(args.data_output))
    plot(data, Path(args.output))
    print(f"wrote {args.output}")
    print(f"wrote {args.data_output}")


def load_data() -> pd.DataFrame:
    rows = [
        load_locomo_from_summaries(
            method="Random",
            method_dir=Path(
                "paper_results/evaluation/exp1_locomo_scale4_random_fixed_v2/autoeval/20260609_234531/"
                "config_000_exp1_locomo_scale4_random_fixed_qwen3_embed_4b/sessions"
            ),
            source_group="exp1_locomo_scale4_random_fixed",
        ),
        load_locomo_from_summaries(
            method="Semantic",
            method_dir=Path(
                "paper_results/evaluation/exp1/autoeval/20260607_235821/"
                "config_007_exp1_locomo_scale4_semantic_qwen3_embed_4b/sessions"
            ),
            source_group="exp1_locomo_scale4_semantic",
        ),
        load_csv_method(
            workload="EventQA",
            method="Random",
            group="exp1_eventqa_scale1_qwen3_embed_4b",
            csv_path=Path("evaluation/autoeval/session_speedups/8b_exp1.csv"),
        ),
        load_csv_method(
            workload="EventQA",
            method="Semantic",
            group="exp1_eventqa_scale1_qwen3_embed_4b",
            csv_path=Path("evaluation/autoeval/session_speedups/8b_exp1.csv"),
        ),
        load_csv_method(
            workload="PERMA",
            method="Random",
            group="exp1_perma_scale1_qwen3_embed_4b_8001",
            csv_path=Path("evaluation/autoeval/session_speedups/8b_exp1.csv"),
        ),
        load_csv_method(
            workload="PERMA",
            method="Semantic",
            group="exp1_perma_scale1_qwen3_embed_4b_8001",
            csv_path=Path("evaluation/autoeval/session_speedups/8b_exp1.csv"),
        ),
    ]
    return pd.DataFrame(rows)


def load_csv_method(*, workload: str, method: str, group: str, csv_path: Path) -> dict[str, object]:
    raw = pd.read_csv(csv_path)
    rows = raw[(raw["group"] == group) & (raw["method"] == method.lower())].copy()
    if rows.empty:
        raise RuntimeError(f"missing {method} rows for {group}")
    q = rows["queries"].astype(int)
    baseline = float((rows["baseline_ttft_ms"] * q).sum() / q.sum())
    method_ttft = float((rows["method_ttft_ms"] * q).sum() / q.sum())
    return summarize_rows(
        workload=workload,
        method=method,
        sessions=int(len(rows)),
        queries=int(q.sum()),
        baseline_ttft_ms=baseline,
        method_ttft_ms=method_ttft,
        fallback_queries=int(rows["fallback_queries"].astype(int).sum()),
        partial_hit_queries=int(rows["partial_hit_queries"].astype(int).sum()),
        full_hit_queries=int(rows["full_hit_queries"].astype(int).sum()),
        registered_pages=int(rows["registered_pages"].astype(int).sum()),
        used_pages=int(rows["used_pages"].astype(int).sum()),
        source=str(csv_path),
        source_group=group,
    )


def load_locomo_from_summaries(*, method: str, method_dir: Path, source_group: str) -> dict[str, object]:
    baseline_dir = Path(
        "paper_results/evaluation/exp1/autoeval/20260607_235821/"
        "config_004_exp1_locomo_scale4_baseline_qwen3_embed_4b/sessions"
    )
    baseline = load_session_summaries(baseline_dir)
    current = load_session_summaries(method_dir)
    common = sorted(set(baseline) & set(current))
    if not common:
        raise RuntimeError(f"no common LoCoMO sessions for {method}")
    queries = sum(int(current[session]["queries"]) for session in common)
    baseline_ttft = (
        sum(float(baseline[session]["engine_ttft_ms_mean"]) * int(current[session]["queries"]) for session in common)
        / queries
    )
    method_ttft = (
        sum(float(current[session]["engine_ttft_ms_mean"]) * int(current[session]["queries"]) for session in common)
        / queries
    )
    return summarize_rows(
        workload="LoCoMO",
        method=method,
        sessions=len(common),
        queries=queries,
        baseline_ttft_ms=baseline_ttft,
        method_ttft_ms=method_ttft,
        fallback_queries=sum(int(current[session]["fallback_queries"]) for session in common),
        partial_hit_queries=sum(int(current[session]["partial_hit_queries"]) for session in common),
        full_hit_queries=sum(int(current[session]["full_hit_queries"]) for session in common),
        registered_pages=sum(int(current[session]["registered_pages"]) for session in common),
        used_pages=sum(int(current[session]["used_pages"]) for session in common),
        source=str(method_dir.parent),
        source_group=source_group,
    )


def summarize_rows(
    *,
    workload: str,
    method: str,
    sessions: int,
    queries: int,
    baseline_ttft_ms: float,
    method_ttft_ms: float,
    fallback_queries: int,
    partial_hit_queries: int,
    full_hit_queries: int,
    registered_pages: int,
    used_pages: int,
    source: str,
    source_group: str,
) -> dict[str, object]:
    return {
        "workload": workload,
        "method": method,
        "sessions": sessions,
        "queries": queries,
        "baseline_ttft_ms": baseline_ttft_ms,
        "method_ttft_ms": method_ttft_ms,
        "speedup": baseline_ttft_ms / method_ttft_ms,
        "fallback_queries": fallback_queries,
        "partial_hit_queries": partial_hit_queries,
        "full_hit_queries": full_hit_queries,
        "registered_pages": registered_pages,
        "used_pages": used_pages,
        "source": source,
        "source_group": source_group,
    }


def load_session_summaries(session_dir: Path) -> dict[str, dict]:
    summaries = {}
    for path in session_dir.glob("*/summary.json"):
        summaries[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def configure_matplotlib() -> None:
    font_family = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 7.2,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "figure.dpi": 300,
            "hatch.linewidth": 0.22,
        }
    )


def choose_serif_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def plot(data: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_speedup, ax_stack) = plt.subplots(
        1,
        2,
        figsize=(3.42, 1.88),
        constrained_layout=False,
    )
    data = data.set_index(["workload", "method"]).loc[(WORKLOADS, METHODS), :].reset_index()
    x = np.arange(len(WORKLOADS)) * 1.12
    method_colors = {
        "Random": "#AEB7C2",
        "Semantic": "#2F6F9F",
    }
    method_hatches = {
        "Random": "/",
        "Semantic": "\\",
    }
    bar_width = 0.34
    method_offsets = {"Random": -bar_width / 2, "Semantic": bar_width / 2}

    for method in METHODS:
        sub = data[data["method"] == method].set_index("workload").loc[WORKLOADS]
        draw_white_hatch_bars(
            ax_speedup,
            (x + method_offsets[method]).tolist(),
            sub["speedup"].to_list(),
            bar_width,
            bottom=None,
            color=method_colors[method],
            hatch=method_hatches[method],
        )
    ax_speedup.axhline(1.0, color="#777777", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax_speedup.set_title("Speedup (R=Ran., S=Sem.)", pad=3)
    ax_speedup.set_ylabel("Speedup")
    set_method_xticks(ax_speedup, x, method_offsets)
    ax_speedup.set_ylim(0, max(3.05, float(data["speedup"].max()) * 1.22))
    ax_speedup.set_yticks([0, 1, 2, 3])
    ax_speedup.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8)
    add_workload_labels(ax_speedup, x)
    style_axis(ax_speedup)

    path_colors = {
        "fallback": "#C4CAD3",
        "partial_hit": "#5C82B3",
        "full_hit": "#7D6FB2",
    }
    stack_width = bar_width
    for method in METHODS:
        method_data = data[data["method"] == method].set_index("workload").loc[WORKLOADS]
        bottom = np.zeros(len(WORKLOADS))
        xpos = x + method_offsets[method]
        for path in PATH_ORDER:
            heights = (
                method_data[f"{path}_queries"].to_numpy(dtype=float)
                / method_data["queries"].to_numpy(dtype=float)
            ).tolist()
            draw_white_hatch_bars(
                ax_stack,
                xpos.tolist(),
                heights,
                stack_width,
                bottom=bottom.tolist(),
                color=path_colors[path],
                hatch=method_hatches[method],
            )
            bottom += np.array(heights)

    ax_stack.set_title("Reuse (F/P/Full)", pad=3)
    ax_stack.set_ylabel("Ratio")
    set_method_xticks(ax_stack, x, method_offsets)
    ax_stack.set_ylim(0, 1.0)
    ax_stack.set_yticks([0, 0.5, 1.0])
    ax_stack.set_yticklabels(["0", "50%", "100%"])
    ax_stack.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8)
    add_workload_labels(ax_stack, x)
    style_axis(ax_stack)

    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.31, top=0.86, wspace=0.35)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def draw_white_hatch_bars(
    ax,
    x: list[float],
    height: list[float],
    width: float,
    *,
    bottom: list[float] | None,
    color: str,
    hatch: str,
) -> None:
    if bottom is None:
        bottom = [0.0] * len(x)
    ax.bar(
        x,
        height,
        width=width,
        bottom=bottom,
        color=color,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    ax.bar(
        x,
        height,
        width=width,
        bottom=bottom,
        color="none",
        edgecolor="white",
        linewidth=0,
        hatch=hatch,
        zorder=3,
    )


def add_method_header(ax, colors: dict[str, str], hatches: dict[str, str], *, y: float) -> None:
    xs = [0.03, 0.44]
    labels = {"Random": "Ran.", "Semantic": "Sem."}
    for x0, method in zip(xs, METHODS, strict=True):
        ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y - 0.020),
                0.048,
                0.040,
                transform=ax.transAxes,
                facecolor=colors[method],
                edgecolor="none",
                clip_on=False,
                zorder=5,
            )
        )
        ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y - 0.020),
                0.048,
                0.040,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor="white",
                linewidth=0,
                hatch=hatches[method],
                clip_on=False,
                zorder=6,
            )
        )
        ax.text(
            x0 + 0.058,
            y,
            labels[method],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            color="#262626",
            clip_on=False,
        )


def add_path_header(ax, colors: dict[str, str], *, y: float) -> None:
    entries = [("fallback", 0.03, "F"), ("partial_hit", 0.25, "P"), ("full_hit", 0.47, "Full")]
    for path, x0, label in entries:
        ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y - 0.020),
                0.048,
                0.040,
                transform=ax.transAxes,
                facecolor=colors[path],
                edgecolor="none",
                clip_on=False,
                zorder=5,
            )
        )
        ax.text(
            x0 + 0.058,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            color="#262626",
            clip_on=False,
        )


def set_method_xticks(ax, x: np.ndarray, offsets: dict[str, float]) -> None:
    positions = []
    labels = []
    for idx in x:
        positions.extend([idx + offsets["Random"], idx + offsets["Semantic"]])
        labels.extend(["R", "S"])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.2)


def add_workload_labels(ax, x: np.ndarray) -> None:
    for idx, workload in zip(x, WORKLOADS, strict=True):
        ax.text(
            idx,
            -0.30,
            WORKLOAD_ABBR[workload],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.3,
            color="#262626",
            clip_on=False,
        )


def annotate_method_positions(ax, x: np.ndarray, offsets: dict[str, float]) -> None:
    for idx in x:
        ax.text(
            idx + offsets["Random"],
            -0.105,
            "R",
            ha="center",
            va="top",
            fontsize=7.0,
            color="#4C5561",
            clip_on=False,
        )
        ax.text(
            idx + offsets["Semantic"],
            -0.105,
            "S",
            ha="center",
            va="top",
            fontsize=7.0,
            color="#2F5876",
            clip_on=False,
        )


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.08)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", default=DEFAULT_DATA_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
