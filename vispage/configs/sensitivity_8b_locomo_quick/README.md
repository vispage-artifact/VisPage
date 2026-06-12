# 8B LoCoMO Sensitivity Quick Runs

Scope:
- Dataset: LoCoMO scale4.
- Model: Qwen3-VL-8B.
- Method: semantic only.
- Background chunk size: 512.
- Amp sweep: amp 2, 3, 7 on port 8020 / GPU0 by default.
- Cache budget: GPU memory utilization 0.6 on port 8021 / GPU1 by default. The cache script starts/stops vLLM by itself.
- Use the existing 0.9 main result as the cache-budget reference.

Start the amp-sweep vLLM server:

```bash
cd <VLLM_DEPLOY_DIR>

./start.sh <VISPAGE_ROOT>/configs/sensitivity_8b_locomo_quick/qwen3vl-8b-sens-amp-gpu0-8020.env
```

Run amp sweep:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/sensitivity_8b_locomo_quick/amp_8020.list \
  --output-root paper_results/evaluation/sensitivity_8b_locomo_quick/amp_8020 \
  --schedule config \
  --session-limit 1 \
  --session-retries 1 \
  --progress-every 1
```

Run cache util 0.6. This script generates a temporary env/config, starts vLLM with the requested util, waits until it is ready, runs autoeval, then stops vLLM:

```bash
cd <VISPAGE_ROOT>

UTILS=0.60 \
CUDA_VISIBLE_DEVICES_OVERRIDE=1 \
PORT_OVERRIDE=8021 \
SESSION_LIMIT=1 \
bash evaluation/autoeval/run_sensitivity_cache_util_quick.sh
```

Resume amp sweep:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list configs/sensitivity_8b_locomo_quick/amp_8020.list \
  --resume-batch-dir <batch_dir> \
  --schedule config \
  --session-limit 1 \
  --session-retries 1 \
  --progress-every 1
```

To run another cache util value, change `UTILS`, for example:

```bash
cd <VISPAGE_ROOT>

UTILS=0.50 \
CUDA_VISIBLE_DEVICES_OVERRIDE=1 \
PORT_OVERRIDE=8021 \
SESSION_LIMIT=1 \
bash evaluation/autoeval/run_sensitivity_cache_util_quick.sh
```
