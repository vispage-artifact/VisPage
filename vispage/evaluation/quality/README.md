# LLM Quality Judge

This directory contains answer-quality evaluation utilities for experiment
traces. The judge is intentionally dataset-neutral: it extracts the question and
gold answer from processed query records, then asks an OpenAI-compatible LLM to
score the model output.

## Basic Usage

Judge one autoeval batch:

```bash
cd <VISPAGE_ROOT>
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/quality/judge_outputs.py \
  paper_results/evaluation/exp1_32b_eventqa_8003_vtok18k_cvtok12k/autoeval/20260609_115640 \
  --base-url http://127.0.0.1:8000 \
  --model qwen3vl-8b \
  --max-workers 32 \
  --output-root paper_results/quality
```

Run a small validation sample first:

```bash
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/quality/judge_outputs.py \
  paper_results/evaluation/exp1_32b_eventqa_8003_vtok18k_cvtok12k/autoeval/20260609_115640 \
  --base-url http://127.0.0.1:8000 \
  --model qwen3vl-8b \
  --sample 50 \
  --max-workers 16 \
  --run-id smoke_eventqa_32b
```

Interleave methods by session for quick comparison. This runs all `conv-26`
rows for baseline, then all `conv-26` rows for semantic, then advances to the
next LoCoMO conversation:

```bash
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/quality/judge_outputs.py \
  paper_results/evaluation/exp1/autoeval/20260607_235821/config_004_exp1_locomo_scale4_baseline_qwen3_embed_4b \
  paper_results/evaluation/exp1/autoeval/20260607_235821/config_007_exp1_locomo_scale4_semantic_qwen3_embed_4b \
  --base-url http://127.0.0.1:8000 \
  --model qwen3vl-8b \
  --schedule session \
  --max-workers 64 \
  --progress-every 20 \
  --run-id locomo_8b_scale4_baseline_semantic
```

## Outputs

Each target config gets:

- `quality.jsonl`: per-query judge result.
- `summary.json`: aggregate score, correct rate, and per-session score.
- `target_manifest.json`: trace path, query path, and row counts.

The per-query output includes trace metadata such as execution path, coverage,
TTFT, selected page source, and read amplification so speed and quality can be
correlated later.

Progress output includes live `mean_score`, `correct_rate`, and
`partial_or_correct_rate` for the current target/session chunk.

## Dataset Gold Fields

- LoCoMO: `metadata.answer`
- EventQA: `metadata.answer`
- PERMA: `metadata.gold_label` plus the matching option text from
  `metadata.options`
