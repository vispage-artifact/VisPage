"""Combine main latency bars and session speedup distribution in one figure."""

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
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = "paper_results/figures/main_latency_speedupdist.pdf"
DEFAULT_DATA_OUTPUT = "paper_results/figures/main_latency_speedupdist_data.csv"


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
    sessions = load_sessions(SPECS)
    aggregate = aggregate_latency(sessions)
    dist = filter_common_sessions(sessions)
    write_combined_csv(aggregate, dist, Path(args.data_output))
    plot(aggregate, dist, Path(args.output), dist_yscale=args.dist_yscale)
    print(f"wrote {args.output}")
    print(f"wrote {args.data_output}")


def load_sessions(specs: list[ResultSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        raw = pd.read_csv(spec.csv_path)
        selected = raw[(raw["group"] == spec.group) & (raw["method"] == spec.method)].copy()
        if selected.empty:
            raise RuntimeError(f"missing rows for {spec}")
        for _, row in selected.iterrows():
            queries = int(row["queries"])
            baseline = float(row["baseline_ttft_ms"])
            vispage = float(row["method_ttft_ms"])
            rows.append(
                {
                    "model": spec.model,
                    "workload": spec.workload,
                    "session_id": str(row["session_id"]),
                    "queries": queries,
                    "baseline_ttft_ms": baseline,
                    "vispage_ttft_ms": vispage,
                    "speedup": baseline / vispage,
                    "source_csv": spec.csv_path,
                    "source_group": spec.group,
                }
            )
    return pd.DataFrame(rows)


def aggregate_latency(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, workload), group in sessions.groupby(["model", "workload"], sort=False):
        q = group["queries"].astype(int)
        baseline = float((group["baseline_ttft_ms"] * q).sum() / q.sum())
        vispage = float((group["vispage_ttft_ms"] * q).sum() / q.sum())
        rows.append(
            {
                "model": model,
                "workload": workload,
                "sessions": int(len(group)),
                "queries": int(q.sum()),
                "baseline_ttft_ms": baseline,
                "vispage_ttft_ms": vispage,
                "speedup": baseline / vispage,
            }
        )
    return pd.DataFrame(rows)


def filter_common_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    common = {}
    for workload, group in sessions.groupby("workload"):
        session_sets = [
            set(model_group["session_id"])
            for _, model_group in group.groupby("model")
        ]
        common[workload] = set.intersection(*session_sets)
    keep = sessions.apply(lambda row: row["session_id"] in common[row["workload"]], axis=1)
    return sessions[keep].copy()


def write_combined_csv(aggregate: pd.DataFrame, dist: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in aggregate.iterrows():
        payload = row.to_dict()
        payload["record_type"] = "latency_aggregate"
        payload["session_id"] = ""
        rows.append(payload)
    for _, row in dist.iterrows():
        payload = row.to_dict()
        payload["record_type"] = "session_speedup"
        rows.append(payload)
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def configure_matplotlib() -> None:
    font_family = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.3,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
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


def plot(aggregate: pd.DataFrame, dist: pd.DataFrame, output: Path, *, dist_yscale: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.05, 1.95),
        gridspec_kw={"width_ratios": [1.05, 0.95, 1.05, 0.95]},
        constrained_layout=False,
    )
    baseline_color = "#AEB7C2"
    vispage_color = "#2F6F9F"
    edge = "#2A2A2A"
    workload_colors = {
        "LoCoMO": "#2F6F9F",
        "EventQA": "#B55D4C",
        "PERMA": "#4F8A5B",
    }
    workload_order = ["LoCoMO", "EventQA", "PERMA"]
    latency_ymax = round_up(
        float(max(aggregate["baseline_ttft_ms"].max(), aggregate["vispage_ttft_ms"].max())) * 1.14,
        500,
    )

    draw_latency_axis(
        axes[0],
        aggregate[aggregate["model"] == "Qwen3-VL-8B"],
        title="8B latency",
        workload_order=workload_order,
        ymax=latency_ymax,
        baseline_color=baseline_color,
        vispage_color=vispage_color,
        edge=edge,
        show_ylabel=True,
    )
    draw_dist_axis(
        axes[1],
        dist[dist["model"] == "Qwen3-VL-8B"],
        title="8B speedup",
        workload_order=workload_order,
        workload_colors=workload_colors,
        edge=edge,
        show_ylabel=True,
        yscale=dist_yscale,
    )
    draw_latency_axis(
        axes[2],
        aggregate[aggregate["model"] == "Qwen3-VL-32B"],
        title="32B latency",
        workload_order=workload_order,
        ymax=latency_ymax,
        baseline_color=baseline_color,
        vispage_color=vispage_color,
        edge=edge,
        show_ylabel=False,
    )
    draw_dist_axis(
        axes[3],
        dist[dist["model"] == "Qwen3-VL-32B"],
        title="32B speedup",
        workload_order=workload_order,
        workload_colors=workload_colors,
        edge=edge,
        show_ylabel=False,
        yscale=dist_yscale,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    speed_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=edge,
            markeredgewidth=0.35,
            markersize=4.4,
            label=name,
        )
        for name, color in workload_colors.items()
    ]
    fig.legend(
        handles + speed_handles,
        labels + list(workload_colors),
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=1.25,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.73, wspace=0.36)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def draw_latency_axis(
    ax,
    data: pd.DataFrame,
    *,
    title: str,
    workload_order: list[str],
    ymax: float,
    baseline_color: str,
    vispage_color: str,
    edge: str,
    show_ylabel: bool,
) -> None:
    sub = data.set_index("workload").loc[workload_order]
    x = np.arange(len(workload_order))
    width = 0.34
    draw_white_hatch_bars(
        ax,
        (x - width / 2).tolist(),
        sub["baseline_ttft_ms"].to_list(),
        width,
        color=baseline_color,
        edge=edge,
        hatch="/",
        label="Baseline",
    )
    draw_white_hatch_bars(
        ax,
        (x + width / 2).tolist(),
        sub["vispage_ttft_ms"].to_list(),
        width,
        color=vispage_color,
        edge=edge,
        hatch="\\",
        label="VISPAGE",
    )
    ax.set_title(title, pad=4)
    if show_ylabel:
        ax.set_ylabel("TTFT (ms)")
    ax.set_ylim(0, ymax)
    ax.set_xticks(x, workload_order)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8)
    style_axis(ax)


def draw_dist_axis(
    ax,
    data: pd.DataFrame,
    *,
    title: str,
    workload_order: list[str],
    workload_colors: dict[str, str],
    edge: str,
    show_ylabel: bool,
    yscale: str,
) -> None:
    rng = np.random.default_rng(9)
    x = np.arange(len(workload_order))
    weighted_means = {}
    for idx, workload in enumerate(workload_order):
        sub = data[data["workload"] == workload]
        jitter = rng.uniform(-0.13, 0.13, size=len(sub))
        ax.scatter(
            idx + jitter,
            sub["speedup"].to_numpy(),
            s=19,
            color=workload_colors[workload],
            edgecolor=edge,
            linewidth=0.32,
            alpha=0.84,
            zorder=3,
        )
        q = sub["queries"].astype(int)
        baseline = float((sub["baseline_ttft_ms"] * q).sum() / q.sum())
        vispage = float((sub["vispage_ttft_ms"] * q).sum() / q.sum())
        weighted_means[workload] = baseline / vispage
        ax.hlines(weighted_means[workload], idx - 0.23, idx + 0.23, color="#111111", linewidth=1.0, zorder=4)

    ax.set_title(title, pad=4)
    if show_ylabel:
        ax.set_ylabel("Session speedup")
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(0.9, 8.0)
        ax.set_yticks([1.0, 2.0, 4.0, 7.0])
        ax.set_yticklabels(["1x", "2x", "4x", "7x"])
    else:
        ax.set_ylim(0.8, max(7.2, float(data["speedup"].max()) * 1.08))
    ax.axhline(1.0, color="#777777", linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(x, workload_order)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8, which="major")
    style_axis(ax)


def draw_white_hatch_bars(
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
        color="none",
        edgecolor="white",
        linewidth=0,
        hatch=hatch,
        zorder=3,
    )


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.08)


def round_up(value: float, step: int) -> int:
    return int(((value + step - 1) // step) * step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--dist-yscale", choices=["linear", "log"], default="log")
    return parser.parse_args()


if __name__ == "__main__":
    main()
