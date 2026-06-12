"""Plot the workflow-aware chunking microbenchmark figure."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIRST_RUN = Path(
    "paper_results/evaluation/microbench_async_chunk/"
    "microbench_server_ttft_fg1000_bg10000_gap100"
)
LATER_RUN = Path(
    "paper_results/evaluation/microbench_async_chunk/"
    "microbench_ttft_fg1000_bg10000_later_collision_gap2000"
)
FIGURE_DIR = Path("paper_results/figures")

GREY = "#AEB7C2"
LIGHT_BLUE = "#9FC6E8"
BLUE = "#4F86B8"
DARK_BLUE = "#174A7C"
RED = "#B73A3A"


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    first = load_run(args.first_run)
    later = load_run(args.later_run)
    sensitivity = collect_chunk_sensitivity(args.first_run)
    validation = collect_validation_rows(first, later)

    validation.to_csv(output_dir / "paper_microbench_chunking_validation_data.csv", index=False)
    sensitivity.to_csv(output_dir / "paper_microbench_chunking_sensitivity_data.csv", index=False)

    plot_figure(
        validation=validation,
        sensitivity=sensitivity,
        output=output_dir / "paper_microbench_chunking_double.pdf",
    )
    plot_sensitivity_single(
        sensitivity=sensitivity,
        fg_only_ms=float(validation["fg_only_ttft_ms"].iloc[0]),
        output=output_dir / "paper_microbench_chunk_sensitivity_single.pdf",
    )
    sweep_lines = collect_sweep_lines(validation, sensitivity)
    sweep_lines.to_csv(output_dir / "paper_microbench_chunk_sweep_lines_data.csv", index=False)
    plot_chunk_sweep_lines(
        sweep_lines=sweep_lines,
        output=output_dir / "paper_microbench_chunk_sweep_1x2.pdf",
    )
    print(f"wrote {output_dir / 'paper_microbench_chunking_double.pdf'}")
    print(f"wrote {output_dir / 'paper_microbench_chunk_sensitivity_single.pdf'}")
    print(f"wrote {output_dir / 'paper_microbench_chunk_sweep_1x2.pdf'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, default=FIRST_RUN)
    parser.add_argument("--later-run", type=Path, default=LATER_RUN)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def load_run(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    return {"path": path, "summary": summary, "manifest": manifest}


def collect_validation_rows(first: dict[str, Any], later: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, run, arrival_s in [
        ("First", first, float(first["manifest"]["fg_delay_ms"]) / 1000.0),
        ("Later", later, float(later["manifest"]["fg_delay_ms"]) / 1000.0),
    ]:
        summary = run["summary"]
        for method, case, bg_role in [
            ("No priority", "a_no_priority", "fake_background"),
            ("Chunked", "a_chunked", "background"),
        ]:
            case_summary = summary["cases"][case]
            rows.append(
                {
                    "collision": label,
                    "method": method,
                    "arrival_s": arrival_s,
                    "ttft_ms": float(case_summary["foreground"]["server_ttft_ms_mean"]),
                    "engine_ttft_ms": float(case_summary["foreground"]["engine_ttft_ms_mean"]),
                    "pre_scheduler_delay_ms": float(
                        case_summary["foreground"]["pre_scheduler_delay_ms_mean"]
                    ),
                    "bg_ttft_ms": float(case_summary[bg_role]["server_ttft_ms_mean"]),
                    "fg_only_ttft_ms": float(summary["fg_only_server_ttft_ms_mean"]),
                    "source": str(run["path"]),
                }
            )
    return pd.DataFrame(rows)


def collect_chunk_sensitivity(path: Path) -> pd.DataFrame:
    manifest = json.loads((path / "manifest.json").read_text())
    salt = str(manifest["cache_salt"])
    rows = []
    with (path / "results.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("cache_key", "")).startswith(salt):
                continue
            if row.get("phase") != "measure":
                continue
            if row.get("role") != "foreground_probe":
                continue
            if not str(row.get("case", "")).startswith("b_mixed_c"):
                continue
            chunk = int(str(row["case"]).rsplit("c", 1)[1])
            probe_group = "First probe" if int(row.get("probe", 0)) == 0 else "Later probes"
            rows.append(
                {
                    "chunk_tokens": chunk,
                    "probe_group": probe_group,
                    "ttft_ms": float(row["server_ttft_ms"]),
                    "engine_ttft_ms": float(row["engine_ttft_ms"]),
                    "pre_scheduler_delay_ms": float(row["pre_scheduler_delay_ms"]),
                    "source": str(path),
                }
            )
    if not rows:
        raise RuntimeError(f"no sensitivity rows found in {path}")
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["chunk_tokens", "probe_group"], as_index=False)
        .agg(
            ttft_ms=("ttft_ms", "mean"),
            ttft_ms_p90=("ttft_ms", lambda values: float(np.percentile(values, 90))),
            engine_ttft_ms=("engine_ttft_ms", "mean"),
            pre_scheduler_delay_ms=("pre_scheduler_delay_ms", "mean"),
            rows=("ttft_ms", "count"),
            source=("source", "first"),
        )
        .sort_values(["chunk_tokens", "probe_group"])
    )
    return grouped


def collect_sweep_lines(validation: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    chunks = sorted(int(value) for value in sensitivity["chunk_tokens"].unique())
    collision_to_probe = {"First": "First probe", "Later": "Later probes"}
    for collision, probe_group in collision_to_probe.items():
        no_priority = float(
            validation[
                (validation["collision"] == collision) & (validation["method"] == "No priority")
            ]["ttft_ms"].iloc[0]
        )
        fg_only = float(
            validation[
                (validation["collision"] == collision) & (validation["method"] == "No priority")
            ]["fg_only_ttft_ms"].iloc[0]
        )
        for chunk in chunks:
            chunked = float(
                sensitivity[
                    (sensitivity["chunk_tokens"] == chunk)
                    & (sensitivity["probe_group"] == probe_group)
                ]["ttft_ms"].iloc[0]
            )
            rows.extend(
                [
                    {
                        "collision": collision,
                        "chunk_tokens": chunk,
                        "method": "Query only",
                        "ttft_ms": fg_only,
                    },
                    {
                        "collision": collision,
                        "chunk_tokens": chunk,
                        "method": "No chunking",
                        "ttft_ms": no_priority,
                    },
                    {
                        "collision": collision,
                        "chunk_tokens": chunk,
                        "method": "Chunked",
                        "ttft_ms": chunked,
                    },
                ]
            )
    return pd.DataFrame(rows)


def plot_figure(*, validation: pd.DataFrame, sensitivity: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 1.78),
        gridspec_kw={"width_ratios": [1.22, 1.0]},
    )
    draw_timeline_axis(axes[0], validation)
    draw_sensitivity_axis(axes[1], sensitivity, float(validation["fg_only_ttft_ms"].iloc[0]))
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.22, top=0.86, wspace=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def draw_timeline_axis(ax: plt.Axes, validation: pd.DataFrame) -> None:
    ax.set_title("(a) Chunking validation", pad=3)
    ax.set_xlim(0, 4.65)
    ax.set_ylim(-0.6, 3.65)
    ax.set_xlabel("Time since background request (s)", labelpad=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#DDE3EA", linewidth=0.45)
    ax.set_axisbelow(True)

    y_positions = {
        ("First", "No priority"): 3.0,
        ("First", "Chunked"): 2.1,
        ("Later", "No priority"): 1.05,
        ("Later", "Chunked"): 0.15,
    }
    label_x = -0.04
    for collision, y in [("First", 2.55), ("Later", 0.6)]:
        ax.text(label_x, y, collision, ha="right", va="center", fontsize=7.0, fontweight="bold")

    for _, row in validation.iterrows():
        y = y_positions[(row["collision"], row["method"])]
        arrival = float(row["arrival_s"])
        fg_end = arrival + float(row["ttft_ms"]) / 1000.0
        bg_end = float(row["bg_ttft_ms"]) / 1000.0
        is_chunked = row["method"] == "Chunked"

        if is_chunked:
            draw_chunked_bg(ax, y, bg_end)
        else:
            ax.broken_barh(
                [(0.0, bg_end)],
                (y - 0.13, 0.26),
                facecolors=DARK_BLUE,
                edgecolors="none",
                alpha=0.86,
            )

        ax.vlines(arrival, y - 0.29, y + 0.29, color=RED, linewidth=0.85)
        ax.annotate(
            "",
            xy=(fg_end, y + 0.23),
            xytext=(arrival, y + 0.23),
            arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": RED, "mutation_scale": 5.5},
        )
        ax.text(
            min(fg_end + 0.04, 4.45),
            y + 0.23,
            f"{row['ttft_ms'] / 1000.0:.2f}s",
            ha="left",
            va="center",
            fontsize=6.6,
            color=RED,
        )
        ax.text(
            0.03,
            y - 0.28,
            "No-prio" if not is_chunked else "Chunked",
            ha="left",
            va="top",
            fontsize=6.6,
            color="#26313D",
        )

    ax.text(0.1, 3.44, "FG arrival", color=RED, fontsize=6.4, ha="center")
    ax.text(2.0, 1.48, "FG arrival", color=RED, fontsize=6.4, ha="center")


def draw_chunked_bg(ax: plt.Axes, y: float, bg_end: float) -> None:
    chunk = 0.74
    gap = 0.035
    start = 0.0
    while start < bg_end:
        width = min(chunk, bg_end - start)
        ax.broken_barh(
            [(start, width)],
            (y - 0.13, 0.26),
            facecolors=BLUE,
            edgecolors="white",
            linewidth=0.35,
            alpha=0.9,
        )
        start += chunk + gap


def draw_sensitivity_axis(ax: plt.Axes, sensitivity: pd.DataFrame, fg_only_ms: float) -> None:
    ax.set_title("(b) Chunk-size sensitivity", pad=3)
    chunks = sorted(int(v) for v in sensitivity["chunk_tokens"].unique())
    x = np.arange(len(chunks))
    width = 0.34
    groups = ["First probe", "Later probes"]
    colors = {"First probe": DARK_BLUE, "Later probes": LIGHT_BLUE}
    hatches = {"First probe": "\\\\\\", "Later probes": "///"}

    for idx, group in enumerate(groups):
        sub = sensitivity[sensitivity["probe_group"] == group].set_index("chunk_tokens")
        values = [float(sub.loc[chunk, "ttft_ms"]) for chunk in chunks]
        offset = (-0.5 + idx) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[group],
            edgecolor="white",
            linewidth=0,
            label=group,
        )
        for bar in bars:
            bar.set_hatch(hatches[group])
            bar.set_edgecolor("white")

    ax.axhline(fg_only_ms, color=RED, linewidth=0.85, linestyle=(0, (3, 2)))
    ax.text(
        len(chunks) - 0.45,
        fg_only_ms + 60,
        "FG-only",
        color=RED,
        fontsize=6.5,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(chunk) for chunk in chunks])
    ax.set_xlabel("Background chunk size (tokens)", labelpad=2)
    ax.set_ylabel("TTFT (ms)", labelpad=2)
    ax.set_ylim(0, max(2600, sensitivity["ttft_ms"].max() * 1.18))
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.05),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.75,
        fontsize=6.5,
    )


def plot_sensitivity_single(*, sensitivity: pd.DataFrame, fg_only_ms: float, output: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.42, 1.48))
    chunks = sorted(int(v) for v in sensitivity["chunk_tokens"].unique())
    x = np.arange(len(chunks))
    width = 0.24
    series = [
        ("FG-only", [fg_only_ms for _ in chunks], GREY, "---"),
        (
            "First probe",
            [
                float(
                    sensitivity[
                        (sensitivity["chunk_tokens"] == chunk)
                        & (sensitivity["probe_group"] == "First probe")
                    ]["ttft_ms"].iloc[0]
                )
                for chunk in chunks
            ],
            DARK_BLUE,
            "\\\\\\",
        ),
        (
            "Later probes",
            [
                float(
                    sensitivity[
                        (sensitivity["chunk_tokens"] == chunk)
                        & (sensitivity["probe_group"] == "Later probes")
                    ]["ttft_ms"].iloc[0]
                )
                for chunk in chunks
            ],
            LIGHT_BLUE,
            "///",
        ),
    ]

    for idx, (label, values, color, hatch) in enumerate(series):
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0,
            label=label,
        )
        for bar in bars:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(chunk) for chunk in chunks])
    ax.set_xlabel("Background chunk size (tokens)", labelpad=2)
    ax.set_ylabel("TTFT (ms)", labelpad=2)
    ax.set_ylim(0, max(2600, sensitivity["ttft_ms"].max() * 1.18))
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.55,
        fontsize=6.2,
    )
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.25, top=0.78)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_chunk_sweep_lines(*, sweep_lines: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.56), sharex=True)
    style = {
        "No chunking": {"color": DARK_BLUE, "hatch": "\\\\\\"},
        "Chunked": {"color": BLUE, "hatch": "///"},
        "Query only": {"color": GREY, "hatch": "---"},
    }
    labels = {"First": "First probe", "Later": "Later probes"}
    chunks = sorted(int(value) for value in sweep_lines["chunk_tokens"].unique())
    x = np.arange(len(chunks))
    width = 0.23

    for ax, collision in zip(axes, ["First", "Later"], strict=True):
        sub = sweep_lines[sweep_lines["collision"] == collision]
        for idx, method in enumerate(["No chunking", "Chunked", "Query only"]):
            method_rows = sub[sub["method"] == method].sort_values("chunk_tokens")
            bars = ax.bar(
                x + (idx - 1) * width,
                method_rows["ttft_ms"],
                width=width,
                color=style[method]["color"],
                edgecolor="white",
                linewidth=0,
                label=method,
            )
            for bar in bars:
                bar.set_hatch(style[method]["hatch"])
                bar.set_edgecolor("white")
        ax.set_title(labels[collision], pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["0.5k", "1k", "2k", "4k"])
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Page chunk", labelpad=1)
        ymax = 3100 if collision == "First" else 2150
        ax.set_ylim(0, ymax)

    axes[0].set_ylabel("TTFT (ms)", labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        handlelength=0.9,
        columnspacing=0.45,
        fontsize=5.8,
    )
    for ax in axes:
        ax.tick_params(axis="both", labelsize=6.2, pad=1.5)
        ax.title.set_fontsize(7.2)
        ax.xaxis.label.set_fontsize(6.6)
        ax.yaxis.label.set_fontsize(6.6)
    fig.subplots_adjust(left=0.135, right=0.995, bottom=0.25, top=0.69, wspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def configure_matplotlib() -> None:
    font_family = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 6.8,
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


if __name__ == "__main__":
    main()
