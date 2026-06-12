"""Generate single-GPU 32B Exp1 configs.

This matrix keeps the base workload unchanged (topk=10, amp=5) while applying
a tighter visual-token budget to reusable pages. Temporal/append is omitted.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "exp1_32b_single"
PARALLEL_CONFIG_DIR = ROOT / "configs" / "exp1_32b_parallel_8003"
CONFIG_LIST = ROOT / "evaluation" / "autoeval" / "configs_exp1_32b_single.txt"
LOCOMO_CONFIG_LIST = ROOT / "evaluation" / "autoeval" / "configs_exp1_32b_locomo_8002.txt"
EVENTQA_CONFIG_LIST = ROOT / "evaluation" / "autoeval" / "configs_exp1_32b_eventqa_8003.txt"
PERMA_CONFIG_LIST = ROOT / "evaluation" / "autoeval" / "configs_exp1_32b_perma_8003.txt"
EVENTQA_PERMA_CONFIG_LIST = (
    ROOT / "evaluation" / "autoeval" / "configs_exp1_32b_eventqa_perma_8003.txt"
)


DATASETS: dict[str, dict[str, Any]] = {
    "locomo": {
        "memory_path": "dataset/locomo/processed/memory_units.jsonl",
        "query_path": "dataset/locomo/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/locomo/processed/embeddings/qwen3-embed-4b.json",
        "scales": [1.0],
    },
    "eventqa": {
        "memory_path": "dataset/eventqa/processed/memory_units.jsonl",
        "query_path": "dataset/eventqa/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/eventqa/processed/embeddings/qwen3-embed-4b.json",
        "scales": [1.0],
    },
    "perma": {
        "memory_path": "dataset/perma/processed/memory_units.jsonl",
        "query_path": "dataset/perma/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/perma/processed/embeddings/qwen3-embed-4b.json",
        "scales": [1.0],
        "temporal_memory_mask": True,
    },
}


MODES: dict[str, dict[str, Any]] = {
    "baseline": {"mode": "baseline"},
    "random": {"mode": "random_anchor", "max_units": 1000, "random_seed": 0},
    "semantic": {"mode": "embedding_ball", "radius_percentile": 100.0, "max_units": 1000},
}


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PARALLEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    locomo_paths = []
    eventqa_paths = []
    perma_paths = []
    eventqa_perma_paths = []
    for dataset_name, dataset in DATASETS.items():
        for scale in dataset["scales"]:
            for paper_mode, locality in MODES.items():
                config = make_config(
                    dataset_name=dataset_name,
                    dataset=dataset,
                    paper_mode=paper_mode,
                    locality=locality,
                    width_scale=scale,
                )
                path = CONFIG_DIR / (
                    f"exp1_32b_{dataset_name}_scale{format_scale(scale)}_"
                    f"{paper_mode}_qwen3_embed_4b.json"
                )
                path.write_text(
                    json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
                paths.append(path.relative_to(ROOT))
                if dataset_name == "locomo":
                    locomo_paths.append(path.relative_to(ROOT))
                else:
                    parallel_config = make_parallel_8003_config(
                        config=config,
                        dataset_name=dataset_name,
                        paper_mode=paper_mode,
                        width_scale=scale,
                    )
                    parallel_path = PARALLEL_CONFIG_DIR / (
                        f"exp1_32b_{dataset_name}_scale{format_scale(scale)}_"
                        f"{paper_mode}_qwen3_embed_4b_8003.json"
                    )
                    parallel_path.write_text(
                        json.dumps(
                            parallel_config,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if dataset_name == "eventqa":
                        eventqa_paths.append(parallel_path.relative_to(ROOT))
                    elif dataset_name == "perma":
                        perma_paths.append(parallel_path.relative_to(ROOT))
                    eventqa_perma_paths.append(parallel_path.relative_to(ROOT))
    CONFIG_LIST.write_text(
        "".join(f"{path.as_posix()}\n" for path in paths),
        encoding="utf-8",
    )
    LOCOMO_CONFIG_LIST.write_text(
        "".join(f"{path.as_posix()}\n" for path in locomo_paths),
        encoding="utf-8",
    )
    EVENTQA_CONFIG_LIST.write_text(
        "".join(f"{path.as_posix()}\n" for path in eventqa_paths),
        encoding="utf-8",
    )
    PERMA_CONFIG_LIST.write_text(
        "".join(f"{path.as_posix()}\n" for path in perma_paths),
        encoding="utf-8",
    )
    EVENTQA_PERMA_CONFIG_LIST.write_text(
        "".join(f"{path.as_posix()}\n" for path in eventqa_perma_paths),
        encoding="utf-8",
    )
    print(f"wrote {len(paths)} configs to {CONFIG_DIR}")
    print(f"wrote {len(eventqa_perma_paths)} configs to {PARALLEL_CONFIG_DIR}")
    print(f"wrote {CONFIG_LIST}")
    print(f"wrote {LOCOMO_CONFIG_LIST}")
    print(f"wrote {EVENTQA_CONFIG_LIST}")
    print(f"wrote {PERMA_CONFIG_LIST}")
    print(f"wrote {EVENTQA_PERMA_CONFIG_LIST}")


def make_config(
    *,
    dataset_name: str,
    dataset: dict[str, Any],
    paper_mode: str,
    locality: dict[str, Any],
    width_scale: float,
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
    scale_key = format_scale(width_scale)
    config["page"]["key_prefix"] = f"exp1-32b-{dataset_name}-{paper_mode}-s{scale_key}"
    config["renderer"]["output_dir"] = (
        f"paper_results/rendered_pages/exp1_32b_single/"
        f"{dataset_name}_scale{scale_key}_{paper_mode}"
    )
    config["renderer"]["width_scale"] = width_scale
    config["metadata"] = {
        "paper_mode": paper_mode,
        "dataset": dataset_name,
        "width_scale": width_scale,
        "post_render_scale": config["renderer"]["post_render_scale"],
        "embedding_model_key": "qwen3-embed-4b",
        "canonical_exp1_32b_single": True,
        "temporal_memory_mask": bool(dataset.get("temporal_memory_mask")),
        "visual_budget_policy": "budget_aware_candidate_trim",
    }
    if paper_mode == "semantic":
        config["metadata"]["semantic_policy"] = "nearest_under_amp"
    return config


def make_parallel_8003_config(
    *,
    config: dict[str, Any],
    dataset_name: str,
    paper_mode: str,
    width_scale: float,
) -> dict[str, Any]:
    parallel_config = deepcopy(config)
    scale_key = format_scale(width_scale)
    parallel_config["run_name"] = "exp1_32b_parallel_8003"
    parallel_config["page"]["key_prefix"] = (
        f"exp1-32b-{dataset_name}-{paper_mode}-s{scale_key}-gpu3"
    )
    parallel_config["renderer"]["output_dir"] = (
        f"paper_results/rendered_pages/exp1_32b_parallel_8003/"
        f"{dataset_name}_scale{scale_key}_{paper_mode}"
    )
    parallel_config["runtime"]["base_url"] = "http://127.0.0.1:8003"
    parallel_config["metadata"]["parallel_runtime"] = "gpu3_8003"
    return parallel_config


def base_config() -> dict[str, Any]:
    return {
        "run_name": "exp1_32b_single",
        "dataset": {},
        "retrieval": {
            "mode": "embedding_retrieve",
            "topk": 10,
            "embedding_cache_path": None,
        },
        "locality": {},
        "page": {
            "key_prefix": "exp1-32b",
            "max_units": 1000,
            "max_amplification": 5,
            "max_visual_tokens": 18000,
            "max_cache_visual_tokens": 12000,
        },
        "renderer": {
            "output_dir": "paper_results/rendered_pages/exp1_32b_single",
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
            "base_url": "http://127.0.0.1:8002",
            "model": "qwen3vl-32b",
            "api_key": None,
            "foreground_max_tokens": 64,
            "background_max_tokens": 1,
            "background_chunk_tokens": 1024,
        },
    }


def format_scale(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
