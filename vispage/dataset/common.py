"""Small helpers for dataset cleaning scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def safe_id(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text.strip())
    return text.strip("-") or "unknown"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stringify_dialogue(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        role = turn.get("role") or turn.get("speaker") or "unknown"
        content = turn.get("content") or turn.get("text") or ""
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def chunk_text(text: str, *, max_chars: int = 1800, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(text.rfind(". ", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
