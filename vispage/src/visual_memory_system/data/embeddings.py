"""Embedding cache loader for actual retrieval and locality estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_embedding_cache(path: str | Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Load canonical embedding cache.

    Expected JSON format:

    ```json
    {
      "memory_embeddings": {"unit_id": [0.1, 0.2]},
      "query_embeddings": {"query_id": [0.1, 0.2]}
    }
    ```
    """

    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{cache_path} must contain a JSON object")
    memory = _parse_embedding_map(payload.get("memory_embeddings"), "memory_embeddings")
    query = _parse_embedding_map(payload.get("query_embeddings"), "query_embeddings")
    return memory, query


def _parse_embedding_map(value: Any, field_name: str) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError(f"embedding cache field {field_name!r} must be an object")
    parsed: dict[str, list[float]] = {}
    for key, embedding in value.items():
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"{field_name}.{key} must be a non-empty list")
        parsed[str(key)] = [float(x) for x in embedding]
    return parsed

