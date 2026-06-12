"""Generate canonical Exp1 config matrix.

The generated JSON files are deliberately explicit so long unattended runs can
be inspected without reverse-engineering which axis value was changed.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "exp1_main"
MAIN_LIST = ROOT / "evaluation" / "autoeval" / "configs_exp1_main.txt"


DATASETS: dict[str, dict[str, Any]] = {
    "locomo": {
        "memory_path": "dataset/locomo/processed/memory_units.jsonl",
        "query_path": "dataset/locomo/processed/query_records.jsonl",
        "embedding_cache_path": "dataset/locomo/processed/embeddings/qwen3-embed-4b.json",
        "scales": [1.0, 4.0],
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
    "temporal": {"mode": "append", "max_units": 1000},
    "semantic": {"mode": "embedding_ball", "radius_percentile": 100.0, "max_units": 1000},
}


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    main_paths = generate_matrix(DATASETS)
    write_list(MAIN_LIST, main_paths)
    print(f"wrote {len(main_paths)} main configs to {CONFIG_DIR}")
    print(f"wrote {MAIN_LIST}")


def generate_matrix(datasets: dict[str, dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for dataset_name, dataset in datasets.items():
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
                    f"exp1_{dataset_name}_scale{format_scale(scale)}_"
                    f"{paper_mode}_qwen3_embed_4b.json"
                )
                path.write_text(
                    json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
                paths.append(path.relative_to(ROOT))
    return paths


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
    config["page"]["key_prefix"] = f"exp1-{dataset_name}-{paper_mode}-s{format_scale(width_scale)}"
    config["renderer"]["output_dir"] = (
        f"paper_results/rendered_pages/exp1_main/"
        f"{dataset_name}_scale{format_scale(width_scale)}_{paper_mode}"
    )
    config["renderer"]["width_scale"] = width_scale
    config["metadata"] = {
        "paper_mode": paper_mode,
        "dataset": dataset_name,
        "width_scale": width_scale,
        "embedding_model_key": "qwen3-embed-4b",
        "canonical_exp1": True,
        "temporal_memory_mask": bool(dataset.get("temporal_memory_mask")),
    }
    if paper_mode == "semantic":
        config["metadata"]["semantic_policy"] = "nearest_under_amp"
    return config


def base_config() -> dict[str, Any]:
    return {
        "run_name": "exp1",
        "dataset": {},
        "retrieval": {
            "mode": "embedding_retrieve",
            "topk": 10,
            "embedding_cache_path": None,
        },
        "locality": {},
        "page": {
            "key_prefix": "exp1",
            "max_units": 1000,
            "max_amplification": 5,
            "max_visual_tokens": 60000,
        },
        "renderer": {
            "output_dir": "paper_results/rendered_pages/exp1_main",
            "unit_width": 1024,
            "width_scale": 1.0,
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
            "base_url": "http://127.0.0.1:8000",
            "model": "qwen3vl-8b",
            "api_key": None,
            "foreground_max_tokens": 64,
            "background_max_tokens": 1,
            "background_chunk_tokens": 1024,
        },
    }


def write_list(path: Path, paths: list[Path]) -> None:
    path.write_text(
        "".join(f"{config_path.as_posix()}\n" for config_path in paths),
        encoding="utf-8",
    )


def format_scale(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
