# VisPage Artifact Release

This directory is a staging copy for the VisPage paper artifact. It is separate
from the working repository and contains only the files needed to inspect,
reproduce, and extend the paper experiments.

## Layout

```text
vispage/
  src/                         VisPage system code
  dataset/                     data cleaning, embedding, and profiling scripts
  evaluation/                  auto-evaluation, quality judge, plotting scripts
  experiments/                 single-config experiment entry points
  configs/                     paper-oriented experiment configs
  tests/                       unit tests

data/
  schema_examples/             normalized data interface examples
  processed/                   placeholder for regenerated processed data

results/
  traces/                      final experiment traces and summaries
  figures/                     final paper figures as PDF
  figure_data/                 CSV files used to draw the figures
  quality_summaries/           selected LLM-judge summaries
  latency_summaries/           reserved for compact latency summaries

vllm_patch/
  vllm_vispage_artifact.patch  combined VisPage runtime patch for vLLM
  deploy_examples/             example vLLM server env files

scripts/
  run_smoke.sh                 small sanity run
  run_quality.sh               quality judge entry point
  plot_figures.sh              redraw paper figures
```

## What Is Included

- Schema examples for normalized memory/query files.
- Final experiment traces, summaries, figure data, and quality summaries.
- Main 8B/32B traces, random-layout ablation traces, sensitivity traces, and
  microbenchmark request/result logs.
- VisPage source code, evaluation scripts, quality judge, and plotting scripts.
- Final figure PDFs and the CSV data used to generate them.
- A vLLM patch note and example deployment env files.

## What Is Not Included

- Model weights.
- Rendered page images and full raw traces.
- Full raw or processed third-party datasets. See `DATA.md` for source notes.
- The local conda environment.
- The complete modified vLLM repository. See `VLLM.md`.

## Quick Start

```bash
cd vispage-artifact-release/vispage
python -m pip install -e .
```

Start a modified vLLM server as described in `VLLM.md`, then run a smoke
experiment:

```bash
../scripts/run_smoke.sh
```

To regenerate figures from the included CSV/summary artifacts:

```bash
../scripts/plot_figures.sh
```

The included `results/traces/` directory is the primary artifact for validating
the reported speedups and reuse-path behavior. It contains per-query traces and
session summaries, but excludes rendered page images to keep the artifact small.

## Notes

The configs use paths relative to `vispage/`. The `vispage/dataset/*/processed`
entries are relative symlinks into `../data/processed/*` in this release
directory.
