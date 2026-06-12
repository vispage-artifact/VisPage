# Artifact Manifest

## Source

- `vispage/src/visual_memory_system/`: core VisPage implementation.
- `vispage/dataset/`: data preparation, embedding construction, and locality profiling.
- `vispage/evaluation/autoeval/`: session-level experiment runner.
- `vispage/evaluation/quality/`: LLM-judge quality evaluation.
- `vispage/evaluation/plots/`: paper figure plotting scripts.
- `vispage/experiments/`: config-level experiment entry points.
- `vispage/tests/`: unit tests.

## Data Interface

- `data/schema_examples/`
- `data/processed/README.md`
- `data/candidate_sources.md`

## Results

- `results/figures/`: paper-ready PDF figures.
- `results/figure_data/`: CSV data used by plotting scripts.
- `results/quality_summaries/`: selected quality summaries, including the
  PERMA original-format sanity-check summary.
- `results/traces/main_8b/`: final 8B baseline and VisPage traces.
- `results/traces/main_32b/`: final 32B baseline and VisPage traces.
- `results/traces/random_layout/`: random-layout ablation trace.
- `results/traces/sensitivity/`: page amplification and cache-capacity traces.
- `results/traces/microbench/`: workflow-aware page-computation microbench logs.
- `results/traces/perma_original_options_quality/`: PERMA original-format
  sampled-session traces used for the quality sanity check.

## vLLM

- `vllm_patch/vllm_vispage_artifact.patch`: combined runtime patch.
- `vllm_patch/deploy_examples/qwen3vl-8b.env.example`
- `vllm_patch/deploy_examples/qwen3vl-32b.env.example`
