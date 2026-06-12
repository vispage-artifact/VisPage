"""Plot session-level speedup distribution for VISPAGE."""

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


DEFAULT_OUTPUT = "paper_results/figures/session_speedup_distribution.pdf"
DEFAULT_DATA_OUTPUT = "paper_results/figures/session_speedup_distribution_data.csv"


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


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    data, summary = load_data(SPECS)
    write_csv(data, Path(args.data_output))
    plot(data, summary, Path(args.output), yscale=args.yscale)
    print(f"wrote {args.output}")
    print(f"wrote {args.data_output}")


def load_data(specs: list[ResultSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_by_spec = []
    for position, spec in enumerate(specs):
        raw = pd.read_csv(spec.csv_path)
        selected = raw[(raw["group"] == spec.group) & (raw["method"] == spec.method)].copy()
        if selected.empty:
            raise RuntimeError(f"missing rows for {spec}")
        selected["queries"] = selected["queries"].astype(int)
        selected["baseline_ttft_ms"] = selected["baseline_ttft_ms"].astype(float)
        selected["method_ttft_ms"] = selected["method_ttft_ms"].astype(float)
        selected["speedup"] = selected["baseline_ttft_ms"] / selected["method_ttft_ms"]
        selected_by_spec.append((position, spec, selected))

    common_sessions = {}
    for workload in {spec.workload for spec in specs}:
        sets = [
            set(selected["session_id"].astype(str))
            for _, spec, selected in selected_by_spec
            if spec.workload == workload
        ]
        if sets:
            common_sessions[workload] = set.intersection(*sets)

    rows = []
    summaries = []
    for position, spec, selected in selected_by_spec:
        selected = selected[selected["session_id"].astype(str).isin(common_sessions[spec.workload])].copy()
        if selected.empty:
            raise RuntimeError(f"no common sessions for {spec.workload}")
        for _, row in selected.iterrows():
            rows.append(
                {
                    "position": position,
                    "label": spec.label.replace("\n", " "),
                    "model": spec.model,
                    "workload": spec.workload,
                    "session_id": row["session_id"],
                    "queries": int(row["queries"]),
                    "speedup": float(row["speedup"]),
                    "baseline_ttft_ms": float(row["baseline_ttft_ms"]),
                    "vispage_ttft_ms": float(row["method_ttft_ms"]),
                    "source_csv": spec.csv_path,
                    "source_group": spec.group,
                    "common_session_filter": True,
                }
            )

        q = selected["queries"]
        weighted_baseline = float((selected["baseline_ttft_ms"] * q).sum() / q.sum())
        weighted_vispage = float((selected["method_ttft_ms"] * q).sum() / q.sum())
        summaries.append(
            {
                "position": position,
                "label": spec.label.replace("\n", " "),
                "queries": int(q.sum()),
                "sessions": int(len(selected)),
                "weighted_speedup": weighted_baseline / weighted_vispage,
                "median_speedup": float(selected["speedup"].median()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


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
        }
    )


def choose_serif_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def plot(data: pd.DataFrame, summary: pd.DataFrame, output: Path, *, yscale: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7.05, 2.25), constrained_layout=False)

    colors = {
        "LoCoMO": "#2F6F9F",
        "EventQA": "#B55D4C",
        "PERMA": "#4F8A5B",
    }
    edge = "#252525"

    for pos in sorted(data["position"].unique()):
        sub = data[data["position"] == pos].copy()
        jitter = rng.uniform(-0.12, 0.12, size=len(sub))
        sizes = np.interp(sub["queries"].to_numpy(), (sub["queries"].min(), sub["queries"].max()), (22, 46))
        ax.scatter(
            sub["position"].to_numpy() + jitter,
            sub["speedup"].to_numpy(),
            s=sizes,
            color=[colors[w] for w in sub["workload"]],
            alpha=0.82,
            edgecolor=edge,
            linewidth=0.35,
            zorder=3,
        )

    for _, row in summary.iterrows():
        pos = float(row["position"])
        mean = float(row["weighted_speedup"])
        ax.hlines(mean, pos - 0.24, pos + 0.24, color="#111111", linewidth=1.25, zorder=4)
        ax.text(
            pos,
            mean + 0.12,
            f"{mean:.2f}x",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#111111",
        )

    labels = summary.sort_values("position")["label"].to_list()
    ax.set_xticks(range(len(labels)), [label.replace(" ", "\n", 1) for label in labels])
    ax.set_ylabel("Session speedup")
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(0.85, max(8.0, float(data["speedup"].max()) * 1.18))
        ax.set_yticks([1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
        ax.set_yticklabels(["1.0x", "1.5x", "2.0x", "3.0x", "5.0x", "7.0x"])
    else:
        ax.set_ylim(0.65, max(7.2, float(data["speedup"].max()) * 1.12))
    ax.axhline(1.0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(2.5, color="#C8CED6", linewidth=0.65, linestyle=(0, (2, 2)))
    heading_y = 0.98 if yscale == "log" else ax.get_ylim()[1] * 0.985
    heading_transform = ax.get_xaxis_transform() if yscale == "log" else ax.transData
    ax.text(1.0, heading_y, "Qwen3-VL-8B", ha="center", va="top", fontsize=8.0, transform=heading_transform)
    ax.text(4.0, heading_y, "Qwen3-VL-32B", ha="center", va="top", fontsize=8.0, transform=heading_transform)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.035)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor=edge,
            markeredgewidth=0.35,
            markersize=5.0,
            label=name,
        )
        for name, color in colors.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color="#111111", linewidth=1.25, label="Query-weighted mean")
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=False,
        ncol=4,
        handlelength=1.2,
        columnspacing=1.3,
    )

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.23, top=0.75)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    return parser.parse_args()


if __name__ == "__main__":
    main()
