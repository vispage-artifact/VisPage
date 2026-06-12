#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../vispage"

PYTHONPATH=src python evaluation/autoeval/run_sessions.py \
  --configs configs/exp1_8b_server_ttft_quick/locomo_baseline_8000.json \
  --output-root ../results/smoke/locomo_baseline \
  --schedule config \
  --session-limit 1 \
  --query-limit 10 \
  --session-retries 0 \
  --progress-every 1

