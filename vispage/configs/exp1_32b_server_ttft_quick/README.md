# Exp1 32B Server-TTFT Quick Rerun

This reruns the three 32B workloads with the new server-side TTFT metric.

Scope:
- LoCoMO scale1 on port 8010 / GPU0.
- EventQA scale1 on port 8011 / GPU1.
- PERMA scale1 on port 8012 / GPU2.
- Methods: baseline and semantic only.
- Schedule: session-interleaved, so each session runs baseline then semantic.
- Limit: first 5 sessions per dataset.
- 32B visual budgets: foreground max 18000 visual tokens, reusable cache page max 12000 visual tokens.
- Background chunk size: 512 tokens.
- Ports 8010/8011/8012 are used to avoid colliding with the 8B 8000/8001/8002 servers.

Start vLLM servers:

```bash
cd <VLLM_DEPLOY_DIR>

./start.sh <VISPAGE_ROOT>/configs/exp1_32b_server_ttft_quick/qwen3vl-32b-gpu0-8010.env
./start.sh <VISPAGE_ROOT>/configs/exp1_32b_server_ttft_quick/qwen3vl-32b-gpu1-8011.env
./start.sh <VISPAGE_ROOT>/configs/exp1_32b_server_ttft_quick/qwen3vl-32b-gpu2-8012.env
```

LoCoMO on GPU0 / 8010:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/locomo_8010.list \
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/locomo_8010 \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

EventQA on GPU1 / 8011:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/eventqa_8011.list \
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/eventqa_8011 \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

PERMA on GPU2 / 8012:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/perma_8012.list \
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/perma_8012 \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

Resume by reusing the printed `batch_dir`:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/<dataset>_<port>.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

## Copy-Paste Commands

Start three 32B vLLM servers:

```bash
cd <VLLM_DEPLOY_DIR>

./start.sh <VISPAGE_ROOT>/configs/exp1_32b_server_ttft_quick/qwen3vl-32b-gpu0-8010.env
./start.sh <VISPAGE_ROOT>/configs/exp1_32b_server_ttft_quick/qwen3vl-32b-gpu1-8011.env
./start.sh <VISPAGE_ROOT>/configs/exp1_32b_server_ttft_quick/qwen3vl-32b-gpu2-8012.env
```

Run LoCoMO on GPU0 / 8010:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/locomo_8010.list \
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/locomo_8010 \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

Run EventQA on GPU1 / 8011:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/eventqa_8011.list \
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/eventqa_8011 \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

Run PERMA on GPU2 / 8012:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/perma_8012.list \
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/perma_8012 \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

Resume LoCoMO:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/locomo_8010.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

Resume EventQA:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/eventqa_8011.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```

Resume PERMA:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/exp1_32b_server_ttft_quick/perma_8012.list \
  --resume-batch-dir <batch_dir> \
  --schedule session \
  --session-limit 5 \
  --session-retries 1 \
  --progress-every 1
```
