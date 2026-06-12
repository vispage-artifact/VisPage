"""Plot semantic locality signal from the motivation table."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT = Path("paper_results/figures/paper_semantic_locality_signal_single.pdf")
DEFAULT_DATA = Path("paper_results/figures/paper_semantic_locality_signal_data.csv")

HORIZONS = [1, 20, 40, 80]
DATA = {
    5: {
        "Top semantic": [0.0217, 0.1807, 0.2688, 0.3691],
        "Random": [0.0065, 0.0980, 0.1612, 0.2389],
        "Bottom semantic": [0.0035, 0.0780, 0.1380, 0.2212],
    },
    10: {
        "Top semantic": [0.0359, 0.2772, 0.3846, 0.4835],
        "Random": [0.0127, 0.1640, 0.2666, 0.3555],
        "Bottom semantic": [0.0058, 0.1104, 0.1889, 0.2865],
    },
}

STYLE = {
    "Top semantic": {"color": "#1f5aa6", "marker": "o", "linestyle": "-"},
    "Random": {"color": "#7f8792", "marker": "s", "linestyle": "--"},
    "Bottom semantic": {"color": "#6f4aa8", "marker": "^", "linestyle": "-."},
}


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    rows = build_rows()
    write_csv(rows, args.data_output)
    plot(rows, args.output)
    print(f"wrote {args.output}")
    print(f"wrote {args.data_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


def build_rows() -> pd.DataFrame:
    rows = []
    for topk, methods in DATA.items():
        for method, values in methods.items():
            for horizon, hit_rate in zip(HORIZONS, values):
                rows.append(
                    {
                        "topk": topk,
                        "candidate_method": method,
                        "horizon": horizon,
                        "candidate_unit_hit_rate": hit_rate,
                    }
                )
    return pd.DataFrame(rows)


def write_csv(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def configure_matplotlib() -> None:
    font_family = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "figure.dpi": 300,
        }
    )


def choose_serif_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def plot(rows: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(3.48, 1.62), sharey=True)

    for ax, topk in zip(axes, [5, 10]):
        selected = rows[rows["topk"] == topk]
        for method in ["Top semantic", "Random", "Bottom semantic"]:
            method_rows = selected[selected["candidate_method"] == method].sort_values("horizon")
            style = STYLE[method]
            ax.plot(
                method_rows["horizon"],
                method_rows["candidate_unit_hit_rate"] * 100.0,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.15,
                markersize=3.2,
                markeredgewidth=0.0,
            )
        ax.set_title(f"Top-{topk} anchors", pad=2.0)
        ax.set_xlabel("Future horizon $H$")
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0, 83)
        ax.set_ylim(0, 52)
        ax.grid(axis="y", color="#d8dce2", linewidth=0.45, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Candidate hit rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.78, wspace=0.16)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()
