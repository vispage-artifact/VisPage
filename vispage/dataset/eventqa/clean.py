#!/usr/bin/env python3
"""Clean MemoryAgentBench EventQA rows into the standard JSONL interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset.common import chunk_text, safe_id, write_json, write_jsonl


def listish(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    return list(value)


def is_eventqa_row(row: Any) -> bool:
    metadata = row["metadata"] or {}
    previous_events = metadata.get("previous_events")
    if previous_events is not None:
        return True
    questions = row["questions"]
    return bool(len(questions) and "events that have already occurred" in str(questions[0]))


def clean(input_path: Path, output_dir: Path, *, max_chars: int, overlap: int) -> dict[str, Any]:
    df = pd.read_parquet(input_path)
    memory_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    scenario_index = 0

    for row_index, row in df.iterrows():
        if not is_eventqa_row(row):
            continue
        scenario_index += 1
        metadata = row["metadata"] or {}
        qa_ids = listish(metadata.get("qa_pair_ids"))
        task_id = f"eventqa:scenario:{scenario_index:04d}"
        chunks = chunk_text(str(row["context"]), max_chars=max_chars, overlap=overlap)
        for chunk_index, text in enumerate(chunks, start=1):
            memory_rows.append(
                {
                    "unit_id": f"{task_id}:chunk:{chunk_index:05d}",
                    "text": text,
                    "task_id": task_id,
                    "source_id": "MemoryAgentBench:Accurate_Retrieval:EventQA",
                    "session_id": f"row_{row_index}",
                    "step_id": chunk_index,
                    "metadata": {
                        "dataset": "eventqa",
                        "row_index": int(row_index),
                        "chunk_index": chunk_index,
                        "chunk_max_chars": max_chars,
                        "chunk_overlap": overlap,
                    },
                }
            )

        questions = listish(row["questions"])
        answers = listish(row["answers"])
        previous_events = listish(metadata.get("previous_events"))
        for q_index, question in enumerate(questions, start=1):
            answer = answers[q_index - 1] if q_index - 1 < len(answers) else None
            if hasattr(answer, "tolist"):
                answer = answer.tolist()
            query_rows.append(
                {
                    "query_id": f"{task_id}:q{q_index:04d}",
                    "task_id": task_id,
                    "query_index": q_index,
                    "query_text": str(question),
                    "access_units": [],
                    "access_type": "gold",
                    "metadata": {
                        "dataset": "eventqa",
                        "row_index": int(row_index),
                        "qa_pair_id": str(qa_ids[q_index - 1]) if q_index - 1 < len(qa_ids) else None,
                        "answer": answer,
                        "previous_events": str(previous_events[q_index - 1]) if q_index - 1 < len(previous_events) else None,
                        "gold_evidence_status": "unavailable",
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    memory_count = write_jsonl(output_dir / "memory_units.jsonl", memory_rows)
    query_count = write_jsonl(output_dir / "query_records.jsonl", query_rows)
    manifest = {
        "dataset": "eventqa",
        "input_path": str(input_path),
        "memory_units": memory_count,
        "queries": query_count,
        "scenarios": scenario_index,
        "notes": "EventQA has ordered narrative queries but no reliable memory-unit gold evidence; access_units are empty.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/candidate/MemoryAgentBench/Accurate_Retrieval.parquet"))
    parser.add_argument("--output", type=Path, default=Path("dataset/eventqa/processed"))
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()
    print(json.dumps(clean(args.input, args.output, max_chars=args.max_chars, overlap=args.overlap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
