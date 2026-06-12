"""Plot VISPAGE reuse-path breakdown and path-level latency."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = "paper_results/figures/reuse_path_breakdown.pdf"
DEFAULT_RATIO_DATA = "paper_results/figures/reuse_path_ratio_data.csv"
DEFAULT_TTFT_DATA = "paper_results/figures/reuse_path_ttft_data.csv"


@dataclass(frozen=True)
class ResultSpec:
    label: str
    model: str
    workload: str
    csv_path: str
    group: str
    method: str = "semantic"


SPECS = [
    ResultSpec(
        label="8B\nLoCoMO",
        model="Qwen3-VL-8B",
        workload="LoCoMO",
        csv_path="evaluation/autoeval/session_speedups/8b_exp1.csv",
        group="exp1_locomo_scale4_qwen3_embed_4b",
    ),
    ResultSpec(
        label="8B\nEventQA",
        model="Qwen3-VL-8B",
        workload="EventQA",
        csv_path="evaluation/autoeval/session_speedups/8b_exp1.csv",
        group="exp1_eventqa_scale1_qwen3_embed_4b",
    ),
    ResultSpec(
        label="8B\nPERMA",
        model="Qwen3-VL-8B",
        workload="PERMA",
        csv_path="evaluation/autoeval/session_speedups/8b_exp1.csv",
        group="exp1_perma_scale1_qwen3_embed_4b_8001",
    ),
    ResultSpec(
        label="32B\nLoCoMO",
        model="Qwen3-VL-32B",
        workload="LoCoMO",
        csv_path="evaluation/autoeval/session_speedups/32b_locomo_scale1.csv",
        group="exp1_32b_locomo_scale1_qwen3_embed_4b",
    ),
    ResultSpec(
        label="32B\nEventQA",
        model="Qwen3-VL-32B",
        workload="EventQA",
        csv_path="evaluation/autoeval/session_speedups/32b_eventqa.csv",
        group="exp1_32b_eventqa_scale1_qwen3_embed_4b_8003",
    ),
    ResultSpec(
        label="32B\nPERMA",
        model="Qwen3-VL-32B",
        workload="PERMA",
        csv_path="evaluation/autoeval/session_speedups/32b_perma.csv",
        group="exp1_32b_perma_scale1_qwen3_embed_4b_8003",
    ),
]

PATH_ORDER = ["fallback", "partial_hit", "full_hit"]
PATH_LABELS = {
    "fallback": "Fallback",
    "partial_hit": "Partial",
    "full_hit": "Full",
}


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    ratio, ttft = load_breakdown(SPECS)
    write_csv(ratio, Path(args.ratio_data))
    write_csv(ttft, Path(args.ttft_data))
    plot(ratio, ttft, Path(args.output))
    print(f"wrote {args.output}")
    print(f"wrote {args.ratio_data}")
    print(f"wrote {args.ttft_data}")


def load_breakdown(specs: list[ResultSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratio_rows = []
    ttft_rows = []

    for position, spec in enumerate(specs):
        raw = pd.read_csv(spec.csv_path)
        selected = raw[(raw["group"] == spec.group) & (raw["method"] == spec.method)].copy()
        if selected.empty:
            raise RuntimeError(f"missing rows for {spec}")

        total_queries = int(selected["queries"].astype(int).sum())
        counts = {
            "fallback": int(selected["fallback_queries"].astype(int).sum()),
            "partial_hit": int(selected["partial_hit_queries"].astype(int).sum()),
            "full_hit": int(selected["full_hit_queries"].astype(int).sum()),
        }
        warm = counts["partial_hit"] + counts["full_hit"]
        for path in PATH_ORDER:
            ratio_rows.append(
                {
                    "position": position,
                    "label": spec.label.replace("\n", " "),
                    "model": spec.model,
                    "workload": spec.workload,
                    "path": path,
                    "path_label": PATH_LABELS[path],
                    "queries": total_queries,
                    "path_queries": counts[path],
                    "path_ratio": counts[path] / total_queries,
                    "warm_queries": warm,
                    "warm_ratio": warm / total_queries,
                    "sessions": int(len(selected)),
                    "source_csv": spec.csv_path,
                    "source_group": spec.group,
                }
            )

        trace_values: dict[str, list[float]] = {path: [] for path in PATH_ORDER}
        for _, row in selected.iterrows():
            trace_path = (
                Path(str(row["batch_dir"]))
                / str(row["method_config"])
                / "sessions"
                / str(row["session_id"])
                / "trace.jsonl"
            )
            if not trace_path.exists():
                raise FileNotFoundError(trace_path)
            with trace_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    trace = json.loads(line)
                    path = trace.get("execution_path")
                    ttft = trace.get("engine_ttft_ms")
                    if path in trace_values and ttft is not None:
                        trace_values[path].append(float(ttft))

        for path in PATH_ORDER:
            values = trace_values[path]
            ttft_rows.append(
                {
                    "position": position,
                    "label": spec.label.replace("\n", " "),
                    "model": spec.model,
                    "workload": spec.workload,
                    "path": path,
                    "path_label": PATH_LABELS[path],
                    "samples": len(values),
                    "engine_ttft_ms_mean": fmean(values),
                    "engine_ttft_ms_p50": percentile(values, 50),
                    "engine_ttft_ms_p90": percentile(values, 90),
                }
            )

    return pd.DataFrame(ratio_rows), pd.DataFrame(ttft_rows)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def configure_matplotlib() -> None:
    font_family = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
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


def plot(ratio: pd.DataFrame, ttft: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.05, 1.95),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0]},
        constrained_layout=False,
    )
    path_colors = {
        "fallback": "#E15759",
        "partial_hit": "#4E79A7",
        "full_hit": "#59A14F",
    }
    path_hatches = {
        "fallback": "/",
        "partial_hit": "\\",
        "full_hit": "-",
    }
    edge = "#2A2A2A"

    draw_ratio_axis(
        axes[0],
        ratio[ratio["model"] == "Qwen3-VL-8B"],
        title="8B reuse",
        path_colors=path_colors,
        path_hatches=path_hatches,
        edge=edge,
        show_ylabel=True,
    )
    draw_ttft_axis(
        axes[1],
        ttft[ttft["model"] == "Qwen3-VL-8B"],
        title="8B TTFT",
        path_colors=path_colors,
        path_hatches=path_hatches,
        edge=edge,
        show_ylabel=True,
    )
    draw_ratio_axis(
        axes[2],
        ratio[ratio["model"] == "Qwen3-VL-32B"],
        title="32B reuse",
        path_colors=path_colors,
        path_hatches=path_hatches,
        edge=edge,
        show_ylabel=False,
    )
    draw_ttft_axis(
        axes[3],
        ttft[ttft["model"] == "Qwen3-VL-32B"],
        title="32B TTFT",
        path_colors=path_colors,
        path_hatches=path_hatches,
        edge=edge,
        show_ylabel=False,
    )

    path_handles, path_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        path_handles,
        path_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=1.4,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.73, wspace=0.32)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def draw_ratio_axis(
    ax,
    data: pd.DataFrame,
    *,
    title: str,
    path_colors: dict[str, str],
    path_hatches: dict[str, str],
    edge: str,
    show_ylabel: bool,
) -> None:
    workloads = ["LoCoMO", "EventQA", "PERMA"]
    labels = ["LoCoMO", "EventQA", "PERMA"]
    x = np.arange(len(workloads))
    bottom = np.zeros(len(workloads))
    for path in PATH_ORDER:
        sub = data[data["path"] == path].set_index("workload").loc[workloads]
        heights = sub["path_ratio"].to_numpy()
        draw_white_hatch_bars(
            ax,
            x.tolist(),
            heights.tolist(),
            0.62,
            bottom=bottom.tolist(),
            color=path_colors[path],
            edge=edge,
            hatch=path_hatches[path],
            label=PATH_LABELS[path],
        )
        bottom += heights

    ax.set_title(title, pad=4)
    if show_ylabel:
        ax.set_ylabel("Query ratio")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.08)


def draw_ttft_axis(
    ax,
    data: pd.DataFrame,
    *,
    title: str,
    path_colors: dict[str, str],
    path_hatches: dict[str, str],
    edge: str,
    show_ylabel: bool,
) -> None:
    workloads = ["LoCoMO", "EventQA", "PERMA"]
    labels = ["LoCoMO", "EventQA", "PERMA"]
    x = np.arange(len(workloads))
    width = 0.22
    offsets = {"fallback": -width, "partial_hit": 0.0, "full_hit": width}
    for path in PATH_ORDER:
        sub = data[data["path"] == path].set_index("workload").loc[workloads]
        heights = [
            float(value) if not pd.isna(value) else np.nan
            for value in sub["engine_ttft_ms_mean"].to_list()
        ]
        draw_white_hatch_bars(
            ax,
            (x + offsets[path]).tolist(),
            heights,
            width,
            bottom=None,
            color=path_colors[path],
            edge=edge,
            hatch=path_hatches[path],
            label=None,
        )

    ax.set_title(title, pad=4)
    ax.set_yscale("log")
    if show_ylabel:
        ax.set_ylabel("TTFT (ms)")
    ax.set_xticks(x, labels)
    ax.set_ylim(40, 4000)
    ax.set_yticks([50, 100, 250, 500, 1000, 2500])
    ax.set_yticklabels(["50", "100", "250", "500", "1000", "2500"])
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8, which="major")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.08)


def draw_white_hatch_bars(
    ax,
    x: list[float],
    height: list[float],
    width: float,
    *,
    bottom: list[float] | None,
    color: str,
    edge: str,
    hatch: str,
    label: str | None,
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
        label=label,
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


def fmean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, pct))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--ratio-data", default=DEFAULT_RATIO_DATA)
    parser.add_argument("--ttft-data", default=DEFAULT_TTFT_DATA)
    return parser.parse_args()


if __name__ == "__main__":
    main()
