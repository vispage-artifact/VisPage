"""JSON configuration loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visual_memory_system.config import (
    DatasetConfig,
    ForegroundConfig,
    LocalityConfig,
    PageConfig,
    PrewarmConfig,
    RendererConfig,
    RetrievalConfig,
    RunConfig,
    RuntimeConfig,
)


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return parse_run_config(payload)


def parse_run_config(payload: dict[str, Any]) -> RunConfig:
    _require(payload, ["run_name", "dataset", "retrieval", "locality", "renderer", "runtime"])
    return RunConfig(
        run_name=str(payload["run_name"]),
        dataset=DatasetConfig(**_object(payload, "dataset")),
        retrieval=RetrievalConfig(**_object(payload, "retrieval")),
        locality=LocalityConfig(**_object(payload, "locality")),
        renderer=RendererConfig(**_object(payload, "renderer")),
        runtime=RuntimeConfig(**_object(payload, "runtime")),
        page=PageConfig(**payload.get("page", {})),
        foreground=ForegroundConfig(**payload.get("foreground", {"min_coverage": 0.5})),
        prewarm=PrewarmConfig(**payload.get("prewarm", {"min_candidate_coverage": 0.5})),
        metadata=dict(payload.get("metadata", {})),
    )


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config field {key!r} must be an object")
    return value


def _require(payload: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"missing required config fields: {missing}")

