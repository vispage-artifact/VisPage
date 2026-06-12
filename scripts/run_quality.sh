#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <baseline_config_dir> <vispage_config_dir> [base_url]" >&2
  exit 2
fi

BASELINE_DIR="$1"
VISPAGE_DIR="$2"
BASE_URL="${3:-http://127.0.0.1:8000}"

cd "$(dirname "$0")/../vispage"

PYTHONPATH=src python evaluation/quality/judge_outputs.py \
  "$BASELINE_DIR" \
  "$VISPAGE_DIR" \
  --base-url "$BASE_URL" \
  --model qwen3vl-8b \
  --schedule session \
  --max-workers 64 \
  --progress-every 20 \
  --run-id artifact_quality

