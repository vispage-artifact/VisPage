#!/usr/bin/env python3
"""Clean PERMA into the standard JSONL interface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset.common import safe_id, stringify_dialogue, write_json, write_jsonl


TASK_FILES = {
    "SD": "input_data_s.json",
    "MD": "input_data_multi_s.json",
}


def stage_from_eval_name(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_([123])\.json", name)
    if not match:
        raise ValueError(f"cannot parse PERMA eval filename {name!r}")
    return match.group(1), match.group(2)


def task_kind(task_id: str) -> str:
    return "MD" if task_id.startswith("MD-") else "SD"


def context_key(context_unit: Any) -> str:
    return json.dumps(context_unit, ensure_ascii=False, sort_keys=True)


def load_task_items(user_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in TASK_FILES.values():
        path = user_dir / filename
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8"))["overall"]:
            items[(str(item["task_id"]), str(item.get("type")))] = item
    return items


def clean(input_root: Path, output_dir: Path) -> dict[str, Any]:
    memory_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    task_root = input_root / "tasks"
    eval_root = input_root / "evaluation"
    users = sorted(path.name for path in task_root.glob("user*") if path.is_dir())

    for user_id in users:
        task_id = f"perma:{safe_id(user_id)}"
        task_items = load_task_items(task_root / user_id)
        context_to_unit: dict[str, str] = {}
        context_counter = 0

        for item in task_items.values():
            for local_index, context_unit in enumerate(item.get("context", [])):
                key = context_key(context_unit)
                if key in context_to_unit:
                    continue
                context_counter += 1
                dialogue, date_text = context_unit[0], context_unit[1] if len(context_unit) > 1 else None
                unit_id = f"{task_id}:ctx:{context_counter:05d}"
                context_to_unit[key] = unit_id
                topics = item.get("topic") or []
                memory_rows.append(
                    {
                        "unit_id": unit_id,
                        "text": f"[{date_text}]\n{stringify_dialogue(dialogue)}",
                        "task_id": task_id,
                        "source_id": "PERMA",
                        "session_id": f"context:{context_counter:05d}",
                        "turn_id": str(local_index),
                        "step_id": context_counter,
                        "metadata": {
                            "dataset": "perma",
                            "user_id": user_id,
                            "date": date_text,
                            "topics": topics,
                            "unit_type": "context_dialogue",
                        },
                    }
                )

        eval_files = sorted((eval_root / user_id / "meta" / "overall").glob("*.json"))
        eval_records: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
        for eval_file in eval_files:
            raw_task_id, stage = stage_from_eval_name(eval_file.name)
            item = task_items[(raw_task_id, stage)]
            eval_payload = json.loads(eval_file.read_text(encoding="utf-8"))
            eval_records.append((stage, eval_file, eval_payload, item))

        eval_records.sort(key=lambda row: (str(row[2].get("question_date") or row[3]["task"].get("date")), row[1].name))
        for query_index, (stage, eval_file, eval_payload, item) in enumerate(eval_records, start=1):
            access_units: list[str] = []
            for link in item.get("affinity_links", []):
                idx = link.get("index")
                if isinstance(idx, int) and 0 <= idx < len(item.get("context", [])):
                    unit_id = context_to_unit.get(context_key(item["context"][idx]))
                    if unit_id and unit_id not in access_units:
                        access_units.append(unit_id)
            query_rows.append(
                {
                    "query_id": f"{task_id}:q{query_index:04d}",
                    "task_id": task_id,
                    "query_index": query_index,
                    "query_text": str(eval_payload.get("question", "")),
                    "access_units": access_units,
                    "access_type": "gold",
                    "metadata": {
                        "dataset": "perma",
                        "user_id": user_id,
                        "eval_file": eval_file.name,
                        "raw_task_id": item["task_id"],
                        "stage": stage,
                        "question_date": eval_payload.get("question_date"),
                        "task_date": item["task"].get("date"),
                        "task_kind": task_kind(item["task_id"]),
                        "topics": item.get("topic"),
                        "gold_label": eval_payload.get("gold_label"),
                        "options": eval_payload.get("options"),
                        "preferences": item.get("preferences", []),
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    memory_count = write_jsonl(output_dir / "memory_units.jsonl", memory_rows)
    query_count = write_jsonl(output_dir / "query_records.jsonl", query_rows)
    manifest = {
        "dataset": "perma",
        "input_root": str(input_root),
        "memory_units": memory_count,
        "queries": query_count,
        "notes": "Task context is converted into retrievable memory units. Queries use evaluation questions only; context is not included in query_text.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("dataset/candidate/PERMA"))
    parser.add_argument("--output", type=Path, default=Path("dataset/perma/processed"))
    args = parser.parse_args()
    print(json.dumps(clean(args.input_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
