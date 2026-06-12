"""Generate 32B server-TTFT quick rerun configs.

This mirrors the 8B quick rerun layout, but keeps the 32B visual-token
budgets from the prior 32B configs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "exp1_32b_server_ttft_quick"


DATASETS: dict[str, dict[str, Any]] = {
    "locomo": {
        "memory_path": "dataset/locomo/processed/memory_units.jsonl",
        "query_path": "dataset/locomo/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/locomo/processed/embeddings/qwen3-embed-4b.json",
        "port": 8010,
        "gpu": 0,
        "scale": 1.0,
    },
    "eventqa": {
        "memory_path": "dataset/eventqa/processed/memory_units.jsonl",
        "query_path": "dataset/eventqa/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/eventqa/processed/embeddings/qwen3-embed-4b.json",
        "port": 8011,
        "gpu": 1,
        "scale": 1.0,
    },
    "perma": {
        "memory_path": "dataset/perma/processed/memory_units.jsonl",
        "query_path": "dataset/perma/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/perma/processed/embeddings/qwen3-embed-4b.json",
        "port": 8012,
        "gpu": 2,
        "scale": 1.0,
        "temporal_memory_mask": True,
    },
}


MODES: dict[str, dict[str, Any]] = {
    "baseline": {"mode": "baseline"},
    "semantic": {"mode": "embedding_ball", "radius_percentile": 100.0, "max_units": 1000},
}


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset in DATASETS.items():
        list_paths: list[Path] = []
        for paper_mode, locality in MODES.items():
            config = make_config(
                dataset_name=dataset_name,
                dataset=dataset,
                paper_mode=paper_mode,
                locality=locality,
            )
            path = CONFIG_DIR / f"{dataset_name}_{paper_mode}_{dataset['port']}.json"
            path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            list_paths.append(path.relative_to(ROOT))

        list_path = CONFIG_DIR / f"{dataset_name}_{dataset['port']}.list"
        list_path.write_text(
            "".join(f"{path.as_posix()}\n" for path in list_paths),
            encoding="utf-8",
        )

    for dataset_name, dataset in DATASETS.items():
        env_path = CONFIG_DIR / f"qwen3vl-32b-gpu{dataset['gpu']}-{dataset['port']}.env"
        env_path.write_text(make_env(dataset_name, dataset), encoding="utf-8")

    (CONFIG_DIR / "README.md").write_text(make_readme(), encoding="utf-8")
    print(f"wrote 32B server-TTFT quick configs to {CONFIG_DIR}")


def make_config(
    *,
    dataset_name: str,
    dataset: dict[str, Any],
    paper_mode: str,
    locality: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(base_config())
    config["dataset"] = {
        "name": dataset_name,
        "memory_path": dataset["memory_path"],
        "query_path": dataset["query_path"],
    }
    if dataset.get("temporal_memory_mask"):
        config["dataset"]["temporal_memory_mask"] = True

    config["retrieval"]["embedding_cache_path"] = dataset["embedding_cache_path"]
    config["locality"] = deepcopy(locality)

    scale_key = format_scale(dataset["scale"])
    config["page"]["key_prefix"] = f"exp1-32b-quick-{dataset_name}-{paper_mode}-s{scale_key}"
    config["renderer"]["output_dir"] = (
        "paper_results/rendered_pages/exp1_32b_server_ttft_quick/"
        f"{dataset_name}_scale{scale_key}_{paper_mode}"
    )
    config["renderer"]["width_scale"] = dataset["scale"]
    config["runtime"]["base_url"] = f"http://127.0.0.1:{dataset['port']}"

    config["metadata"] = {
        "paper_mode": paper_mode,
        "dataset": dataset_name,
        "width_scale": dataset["scale"],
        "post_render_scale": config["renderer"]["post_render_scale"],
        "embedding_model_key": "qwen3-embed-4b",
        "canonical_exp1_32b_server_ttft_quick": True,
        "temporal_memory_mask": bool(dataset.get("temporal_memory_mask")),
        "visual_budget_policy": "budget_aware_candidate_trim",
        "server_ttft_metric": True,
        "runtime_port": dataset["port"],
        "runtime_gpu": dataset["gpu"],
    }
    if paper_mode == "semantic":
        config["metadata"]["semantic_policy"] = "nearest_under_amp"
    return config


def base_config() -> dict[str, Any]:
    return {
        "run_name": "exp1_32b_server_ttft_quick",
        "dataset": {},
        "retrieval": {
            "mode": "embedding_retrieve",
            "topk": 10,
            "embedding_cache_path": None,
        },
        "locality": {},
        "page": {
            "key_prefix": "exp1-32b-quick",
            "max_units": 1000,
            "max_amplification": 5,
            "max_visual_tokens": 18000,
            "max_cache_visual_tokens": 12000,
        },
        "renderer": {
            "output_dir": "paper_results/rendered_pages/exp1_32b_server_ttft_quick",
            "unit_width": 1024,
            "width_scale": 1.0,
            "post_render_scale": 0.7,
            "font_size": 24,
            "line_height": 34,
            "padding": 24,
            "chars_per_line": 72,
            "tile_gap": 0,
        },
        "foreground": {"min_coverage": 0.3},
        "prewarm": {
            "min_candidate_coverage": 0.5,
            "max_new_in_old": 0.5,
            "max_inflight_pages": 1,
        },
        "runtime": {
            "mode": "vllm",
            "base_url": "http://127.0.0.1:8010",
            "model": "qwen3vl-32b",
            "api_key": None,
            "foreground_max_tokens": 64,
            "background_max_tokens": 1,
            "background_chunk_tokens": 512,
        },
    }


def make_env(dataset_name: str, dataset: dict[str, Any]) -> str:
    return f"""CONDA_ENV=vllm-prefetch
VLLM_REPO=<VLLM_REPO>
CUDA_HOME=/usr/local/cuda

SERVICE_NAME=vllm-qwen3vl-32b-{dataset_name}-{dataset['port']}
CUDA_VISIBLE_DEVICES={dataset['gpu']}

MODEL_PATH=<MODEL_ROOT>/qwen3-vl-32b
SERVED_MODEL_NAME=qwen3vl-32b

HOST=0.0.0.0
PORT={dataset['port']}

TENSOR_PARALLEL_SIZE=1
DTYPE=bfloat16
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=32768
TRUST_REMOTE_CODE=1
TASK=
ALLOWED_LOCAL_MEDIA_PATH=<ARTIFACT_DATA_ROOT>

EXTRA_VLLM_ARGS=

VLLM_AM_ENCODER_CACHE_SIZE=32768
VLLM_AM_ENCODER_CACHE_RESERVE=8192
VLLM_AM_ENCODER_CPU_CACHE_GB=16

RUN_DIR=<VLLM_DEPLOY_DIR>/run
LOG_DIR=<VLLM_DEPLOY_DIR>/logs

VLLM_NO_USAGE_STATS=1
"""


def make_readme() -> str:
    return """# Exp1 32B Server-TTFT Quick Rerun

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

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/locomo_8010.list \\
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/locomo_8010 \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

EventQA on GPU1 / 8011:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/eventqa_8011.list \\
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/eventqa_8011 \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

PERMA on GPU2 / 8012:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/perma_8012.list \\
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/perma_8012 \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

Resume by reusing the printed `batch_dir`:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/<dataset>_<port>.list \\
  --resume-batch-dir <batch_dir> \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
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

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/locomo_8010.list \\
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/locomo_8010 \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

Run EventQA on GPU1 / 8011:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/eventqa_8011.list \\
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/eventqa_8011 \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

Run PERMA on GPU2 / 8012:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/perma_8012.list \\
  --output-root paper_results/evaluation/exp1_32b_server_ttft_quick/perma_8012 \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

Resume LoCoMO:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/locomo_8010.list \\
  --resume-batch-dir <batch_dir> \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

Resume EventQA:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/eventqa_8011.list \\
  --resume-batch-dir <batch_dir> \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```

Resume PERMA:

```bash
cd <VISPAGE_ROOT>

PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \\
  --config-list configs/exp1_32b_server_ttft_quick/perma_8012.list \\
  --resume-batch-dir <batch_dir> \\
  --schedule session \\
  --session-limit 5 \\
  --session-retries 1 \\
  --progress-every 1
```
"""


def format_scale(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
