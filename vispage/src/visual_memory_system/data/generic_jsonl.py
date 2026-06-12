"""Generic JSONL adapter for normalized datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visual_memory_system.data.base import DataAdapter
from visual_memory_system.schema import MemoryUnit, QueryRecord


_ALLOWED_ACCESS_TYPES = {"gold", "actual", "profile", "retrieved", "provided"}


class GenericJsonlAdapter(DataAdapter):
    def __init__(self, memory_path: str | Path, query_path: str | Path) -> None:
        self.memory_path = Path(memory_path)
        self.query_path = Path(query_path)

    def load_memory_units(self) -> list[MemoryUnit]:
        rows = _load_jsonl(self.memory_path)
        units: list[MemoryUnit] = []
        for row in rows:
            _require(row, ["unit_id", "text"])
            units.append(
                MemoryUnit(
                    unit_id=str(row["unit_id"]),
                    text=str(row["text"]),
                    task_id=_optional_str(row, "task_id"),
                    source_id=_optional_str(row, "source_id"),
                    session_id=_optional_str(row, "session_id"),
                    turn_id=_optional_str(row, "turn_id"),
                    step_id=row.get("step_id"),
                    timestamp=row.get("timestamp"),
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return units

    def load_queries(self) -> list[QueryRecord]:
        rows = _load_jsonl(self.query_path)
        queries: list[QueryRecord] = []
        for row in rows:
            _require(
                row,
                ["query_id", "task_id", "query_index", "query_text", "access_units", "access_type"],
            )
            if not isinstance(row["access_units"], list):
                raise ValueError(f"query {row['query_id']} access_units must be a list")
            access_type = str(row["access_type"])
            if access_type not in _ALLOWED_ACCESS_TYPES:
                raise ValueError(
                    f"query {row['query_id']} has invalid access_type={access_type!r}; "
                    f"expected one of {sorted(_ALLOWED_ACCESS_TYPES)}"
                )
            access_units = tuple(str(x) for x in row["access_units"])
            queries.append(
                QueryRecord(
                    query_id=str(row["query_id"]),
                    task_id=str(row["task_id"]),
                    query_index=int(row["query_index"]),
                    query_text=str(row["query_text"]),
                    access_units=access_units,
                    access_type=access_type,  # type: ignore[arg-type]
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return queries


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            rows.append(row)
    return rows


def _require(row: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"missing required fields: {missing}")


def _optional_str(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)
