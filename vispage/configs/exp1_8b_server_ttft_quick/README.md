# Exp1 8B Server-TTFT Quick Rerun

This reruns the three 8B workloads with the new server-side TTFT metric.

Scope:
- LoCoMO scale4 on port 8000 / GPU0.
- EventQA scale1 on port 8001 / GPU1.
- PERMA scale1 on port 8002 / GPU2.
- Methods: baseline and semantic only.
- Schedule: session-interleaved, so each session runs baseline then semantic.
- Limit: first 10 sessions per dataset.
- Background chunk size: 512 tokens.

Start vLLM servers:

```bash
cd <VLLM_DEPLOY_DIR>

./start.sh qwen3vl-8b.env
./start.sh qwen3vl-8b-gpu1.env
./start.sh <VISPAGE_ROOT>/configs/exp1_8b_server_ttft_quick/qwen3vl-8b-gpu2-8002.env
```

Run the three datasets in separate shells:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/locomo_8000.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/locomo_8000 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/eventqa_8001.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/eventqa_8001 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/perma_8002.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/perma_8002 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

Resume by reusing the printed `batch_dir`:

```bash
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/locomo_8000.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

Use the matching config list for EventQA or PERMA when resuming those runs.

## Copy-Paste Commands

Start servers:

```bash
cd <VLLM_DEPLOY_DIR>

./start.sh qwen3vl-8b.env
./start.sh qwen3vl-8b-gpu1.env
./start.sh <VISPAGE_ROOT>/configs/exp1_8b_server_ttft_quick/qwen3vl-8b-gpu2-8002.env
```

LoCoMO on GPU0 / 8000:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/locomo_8000.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/locomo_8000 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

EventQA on GPU1 / 8001:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/eventqa_8001.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/eventqa_8001 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

PERMA on GPU2 / 8002:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/perma_8002.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/perma_8002 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

Resume LoCoMO:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/locomo_8000.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

Resume EventQA:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/eventqa_8001.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

Resume PERMA:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/perma_8002.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

## LoCoMO Random on GPU3

This is a random-only LoCoMO scale4 rerun for the server-side TTFT metric.
It uses GPU3 / port 8013 and runs the first 10 sessions.

Start vLLM:

```bash
cd <VLLM_DEPLOY_DIR>

./start.sh <VISPAGE_ROOT>/configs/exp1_8b_server_ttft_quick/qwen3vl-8b-gpu3-8013.env
```

Run LoCoMO random:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/locomo_random_8013.list \
  --output-root paper_results/evaluation/exp1_8b_server_ttft_quick/locomo_random_8013 \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```

Resume LoCoMO random:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_8b_server_ttft_quick/locomo_random_8013.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 10 \
  --session-retries 1 \
  --progress-every 1
```
