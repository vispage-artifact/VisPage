"""Plot main end-to-end latency results for VISPAGE.

The figure is intended for a double-column paper layout:
two panels for Qwen3-VL-8B and Qwen3-VL-32B, grouped by workload.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT = "paper_results/figures/main_results_latency.pdf"
DEFAULT_SINGLE_OUTPUT = "paper_results/figures/main_results_latency_single.pdf"
DEFAULT_DATA_OUTPUT = "paper_results/figures/main_results_latency_data.csv"


@dataclass(frozen=True)
class ResultSpec:
    model: str
    workload: str
    csv_path: str
    group: str
    method: str = "semantic"


SPECS = [
    ResultSpec(
        model="Qwen3-VL-8B",
        workload="LoCoMO",
        csv_path="evaluation/autoeval/session_speedups/8b_exp1.csv",
        group="exp1_locomo_scale4_qwen3_embed_4b",
    ),
    ResultSpec(
        model="Qwen3-VL-8B",
        workload="EventQA",
        csv_path="evaluation/autoeval/session_speedups/8b_exp1.csv",
        group="exp1_eventqa_scale1_qwen3_embed_4b",
    ),
    ResultSpec(
        model="Qwen3-VL-8B",
        workload="PERMA",
        csv_path="evaluation/autoeval/session_speedups/8b_exp1.csv",
        group="exp1_perma_scale1_qwen3_embed_4b_8001",
    ),
    ResultSpec(
        model="Qwen3-VL-32B",
        workload="LoCoMO",
        csv_path="evaluation/autoeval/session_speedups/32b_locomo_scale1.csv",
        group="exp1_32b_locomo_scale1_qwen3_embed_4b",
    ),
    ResultSpec(
        model="Qwen3-VL-32B",
        workload="EventQA",
        csv_path="evaluation/autoeval/session_speedups/32b_eventqa.csv",
        group="exp1_32b_eventqa_scale1_qwen3_embed_4b_8003",
    ),
    ResultSpec(
        model="Qwen3-VL-32B",
        workload="PERMA",
        csv_path="evaluation/autoeval/session_speedups/32b_perma.csv",
        group="exp1_32b_perma_scale1_qwen3_embed_4b_8003",
    ),
]


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    data = load_results(SPECS)
    write_aggregated_csv(data, Path(args.data_output))
    if args.layout == "double":
        plot_double_column(data, Path(args.output))
    else:
        plot_single_column(data, Path(args.output))
    print(f"wrote {args.output}")
    print(f"wrote {args.data_output}")


def load_results(specs: list[ResultSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        raw = pd.read_csv(spec.csv_path)
        selected = raw[(raw["group"] == spec.group) & (raw["method"] == spec.method)].copy()
        if selected.empty:
            raise RuntimeError(f"missing rows for {spec}")
        query_count = selected["queries"].astype(int)
        total_queries = int(query_count.sum())
        baseline = weighted_mean(selected["baseline_ttft_ms"], query_count)
        vispage = weighted_mean(selected["method_ttft_ms"], query_count)
        rows.append(
            {
                "model": spec.model,
                "workload": spec.workload,
                "sessions": int(len(selected)),
                "queries": total_queries,
                "baseline_ttft_ms": baseline,
                "vispage_ttft_ms": vispage,
                "speedup": baseline / vispage,
                "reduction_pct": (1.0 - vispage / baseline) * 100.0,
                "source_csv": spec.csv_path,
                "source_group": spec.group,
            }
        )
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float((values.astype(float) * weights).sum() / weights.sum())


def write_aggregated_csv(data: pd.DataFrame, path: Path) -> None:
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


def plot_double_column(data: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model_order = ["Qwen3-VL-8B", "Qwen3-VL-32B"]
    workload_order = ["LoCoMO", "EventQA", "PERMA"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.45),
        sharey=True,
        constrained_layout=False,
    )

    baseline_color = "#AEB7C2"
    vispage_color = "#2F6F9F"
    edge = "#2A2A2A"
    speedup_color = "#B33A3A"
    bar_width = 0.34

    ymax = max(float(data["baseline_ttft_ms"].max()), float(data["vispage_ttft_ms"].max()))
    ymax = round_up(ymax * 1.18, 250)
    speed_ymax = max(3.0, float(data["speedup"].max()) * 1.18)
    speed_axes = []

    for ax, model in zip(axes, model_order, strict=True):
        sub = data[data["model"] == model].set_index("workload").loc[workload_order]
        x = list(range(len(workload_order)))
        baseline = sub["baseline_ttft_ms"].to_list()
        vispage = sub["vispage_ttft_ms"].to_list()
        speedups = sub["speedup"].to_list()

        draw_hatched_bars(
            ax,
            [v - bar_width / 2 for v in x],
            baseline,
            bar_width,
            color=baseline_color,
            edge=edge,
            hatch="/",
            label="Baseline",
        )
        draw_hatched_bars(
            ax,
            [v + bar_width / 2 for v in x],
            vispage,
            bar_width,
            color=vispage_color,
            edge=edge,
            hatch="\\",
            label="VISPAGE",
        )

        speed_ax = ax.twinx()
        speed_ax.scatter(
            x,
            speedups,
            s=22,
            color=speedup_color,
            edgecolor="#7A1717",
            linewidth=0.35,
            zorder=5,
            label="Speedup",
        )
        speed_ax.set_ylim(0.8, speed_ymax)
        speed_ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
        speed_ax.set_yticklabels(["1.0x", "1.5x", "2.0x", "2.5x", "3.0x"])
        speed_ax.tick_params(axis="y", colors=speedup_color, width=0.6, length=2.8)
        speed_ax.spines["top"].set_visible(False)
        speed_ax.spines["right"].set_color(speedup_color)
        speed_axes.append(speed_ax)

        ax.set_title(model, pad=5)
        ax.set_xticks(x, workload_order)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.08)

        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
            tick.set_ha("center")

    speed_axes[0].tick_params(axis="y", right=False, labelright=False)
    speed_axes[0].spines["right"].set_visible(False)
    speed_axes[1].set_ylabel("Speedup", color=speedup_color)
    axes[0].set_ylabel("TTFT (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    speed_handles, speed_labels = speed_axes[1].get_legend_handles_labels()
    handles += speed_handles
    labels += speed_labels
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.6,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.18, top=0.81, wspace=0.18)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def plot_single_column(data: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model_order = ["Qwen3-VL-8B", "Qwen3-VL-32B"]
    workload_order = ["LoCoMO", "EventQA", "PERMA"]
    indexed = data.set_index(["model", "workload"])

    fig, axes = plt.subplots(1, 2, figsize=(3.42, 2.35), sharey=True, constrained_layout=False)
    baseline_color = "#AEB7C2"
    vispage_color = "#2F6F9F"
    edge = "#2A2A2A"
    bar_width = 0.34
    ymax = max(3.1, float(data["speedup"].max()) * 1.22)
    legend_handles = None

    for ax, model in zip(axes, model_order, strict=True):
        speedups = [float(indexed.loc[(model, workload), "speedup"]) for workload in workload_order]
        x = list(range(len(workload_order)))
        baseline = [1.0 for _ in workload_order]

        draw_hatched_bars(
            ax,
            [v - bar_width / 2 for v in x],
            baseline,
            bar_width,
            color=baseline_color,
            edge=edge,
            hatch="/",
            label="Baseline",
        )
        draw_hatched_bars(
            ax,
            [v + bar_width / 2 for v in x],
            speedups,
            bar_width,
            color=vispage_color,
            edge=edge,
            hatch="\\",
            label="VISPAGE",
        )

        for xpos, speedup in zip(x, speedups, strict=True):
            ax.text(
                xpos + bar_width / 2,
                speedup + ymax * 0.025,
                f"{speedup:.1f}x",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#1F1F1F",
            )

        ax.set_title(model, pad=4)
        ax.set_xticks(x, workload_order)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 1, 2, 3] if ymax <= 3.4 else [0, 1, 2, 3, 4])
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.08)
        if legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Speedup over baseline")
    handles, labels = legend_handles if legend_handles is not None else ([], [])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.18, top=0.78, wspace=0.16)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def draw_hatched_bars(
    ax,
    x: list[float],
    height: list[float],
    width: float,
    *,
    color: str,
    edge: str,
    hatch: str,
    label: str,
) -> None:
    ax.bar(
        x,
        height,
        width=width,
        label=label,
        color=color,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    ax.bar(
        x,
        height,
        width=width,
        color="none",
        edgecolor="white",
        linewidth=0,
        hatch=hatch,
        zorder=3,
    )


def round_up(value: float, step: int) -> int:
    return int(((value + step - 1) // step) * step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--layout", choices=["double", "single"], default="double")
    return parser.parse_args()


if __name__ == "__main__":
    main()
