#!/usr/bin/env python3
"""Clean LoCoMO into the repository standard JSONL interface."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset.common import safe_id, write_json, write_jsonl


def session_sort_key(key: str) -> int:
    match = re.search(r"session_(\d+)$", key)
    return int(match.group(1)) if match else 10**9


def clean(input_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    memory_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for sample in raw:
        sample_id = safe_id(sample["sample_id"])
        task_id = f"locomo:{sample_id}"
        conversation = sample["conversation"]
        dia_to_unit: dict[str, str] = {}
        global_step = 0

        session_keys = sorted(
            [key for key, value in conversation.items() if re.fullmatch(r"session_\d+", key) and isinstance(value, list)],
            key=session_sort_key,
        )
        for session_key in session_keys:
            session_index = session_sort_key(session_key)
            date_text = conversation.get(f"{session_key}_date_time")
            for turn in conversation[session_key]:
                global_step += 1
                dia_id = safe_id(turn["dia_id"])
                unit_id = f"{task_id}:{dia_id}"
                dia_to_unit[turn["dia_id"]] = unit_id
                speaker = turn.get("speaker", "unknown")
                prefix = f"[{date_text}] " if date_text else ""
                memory_rows.append(
                    {
                        "unit_id": unit_id,
                        "text": f"{prefix}{speaker}: {turn.get('text', '')}",
                        "task_id": task_id,
                        "source_id": "locomo",
                        "session_id": session_key,
                        "turn_id": turn["dia_id"],
                        "step_id": global_step,
                        "metadata": {
                            "dataset": "locomo",
                            "sample_id": sample["sample_id"],
                            "speaker": speaker,
                            "dia_id": turn["dia_id"],
                            "session_index": session_index,
                            "session_datetime": date_text,
                        },
                    }
                )

        for index, qa in enumerate(sample.get("qa", []), start=1):
            evidence = [str(x) for x in qa.get("evidence", [])]
            access_units = [dia_to_unit[eid] for eid in evidence if eid in dia_to_unit]
            query_rows.append(
                {
                    "query_id": f"{task_id}:q{index:04d}",
                    "task_id": task_id,
                    "query_index": index,
                    "query_text": str(qa.get("question", "")),
                    "access_units": access_units,
                    "access_type": "gold",
                    "metadata": {
                        "dataset": "locomo",
                        "sample_id": sample["sample_id"],
                        "answer": qa.get("answer"),
                        "category": qa.get("category"),
                        "evidence": evidence,
                        "missing_evidence": [eid for eid in evidence if eid not in dia_to_unit],
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    memory_count = write_jsonl(output_dir / "memory_units.jsonl", memory_rows)
    query_count = write_jsonl(output_dir / "query_records.jsonl", query_rows)
    manifest = {
        "dataset": "locomo",
        "input_path": str(input_path),
        "memory_units": memory_count,
        "queries": query_count,
        "notes": "Memory units are raw utterances. Session/event summaries are intentionally excluded.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/candidate/locomo/locomo10.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/locomo/processed"))
    args = parser.parse_args()
    print(json.dumps(clean(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
