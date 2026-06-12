#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../vispage"

PYTHONPATH=src python evaluation/plots/plot_final_server_ttft_figures.py \
  --output-dir ../results/figures_reproduced

PYTHONPATH=src python evaluation/plots/plot_semantic_locality_signal.py \
  --output ../results/figures_reproduced/paper_semantic_locality_signal_single.pdf \
  --data-output ../results/figures_reproduced/paper_semantic_locality_signal_data.csv

