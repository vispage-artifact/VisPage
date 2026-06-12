#!/usr/bin/env python3
"""Build embedding caches for cleaned datasets through a vLLM embedding server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EmbeddingConfig:
    memory_path: Path
    query_path: Path
    output_path: Path
    base_url: str
    model: str
    model_key: str
    batch_size: int = 64
    concurrency: int = 8
    timeout_seconds: float = 120.0
    api_key: str | None = None
    checkpoint_every_batches: int = 20
    max_chars: int | None = None
    memory_limit: int | None = None
    query_limit: int | None = None


@dataclass(frozen=True)
class EmbedRecord:
    kind: str
    record_id: str
    text: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Load inputs and report pending work only")
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_records(config)
    cache = load_existing_cache(config.output_path)
    cache["metadata"] = build_metadata(config)
    pending = [
        record
        for record in records
        if record.record_id
        not in (cache["memory_embeddings"] if record.kind == "memory" else cache["query_embeddings"])
    ]

    print(
        json.dumps(
            {
                "memory_path": str(config.memory_path),
                "query_path": str(config.query_path),
                "output_path": str(config.output_path),
                "model": config.model,
                "model_key": config.model_key,
                "batch_size": config.batch_size,
                "concurrency": config.concurrency,
                "total_records": len(records),
                "pending_records": len(pending),
                "existing_memory_embeddings": len(cache["memory_embeddings"]),
                "existing_query_embeddings": len(cache["query_embeddings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run or not pending:
        return

    endpoint = embedding_endpoint(config.base_url)
    batches = list(batch_records(pending, config.batch_size))
    completed_batches = 0
    started_at = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        future_to_batch = {
            pool.submit(embed_batch, endpoint, config, batch): batch for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            embeddings = future.result()
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"embedding server returned {len(embeddings)} embeddings for "
                    f"{len(batch)} inputs"
                )
            for record, embedding in zip(batch, embeddings):
                target = (
                    cache["memory_embeddings"]
                    if record.kind == "memory"
                    else cache["query_embeddings"]
                )
                target[record.record_id] = embedding

            completed_batches += 1
            if completed_batches % max(1, config.checkpoint_every_batches) == 0:
                save_cache(config.output_path, cache)
                print(progress_line(completed_batches, len(batches), pending, started_at), flush=True)

    save_cache(config.output_path, cache)
    print(progress_line(completed_batches, len(batches), pending, started_at), flush=True)


def load_config(path: Path) -> EmbeddingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["memory_path", "query_path", "base_url", "model"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    model_key = str(payload.get("model_key") or safe_model_key(payload["model"]))
    output_path = payload.get("output_path")
    if output_path is None:
        output_dir = payload.get("output_dir")
        if output_dir is None:
            raise ValueError(f"{path} must set either output_path or output_dir")
        output_path = str(Path(output_dir) / f"{model_key}.json")
    config = EmbeddingConfig(
        memory_path=Path(payload["memory_path"]),
        query_path=Path(payload["query_path"]),
        output_path=Path(output_path),
        base_url=str(payload["base_url"]),
        model=str(payload["model"]),
        model_key=model_key,
        batch_size=int(payload.get("batch_size", 64)),
        concurrency=int(payload.get("concurrency", 8)),
        timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
        api_key=payload.get("api_key"),
        checkpoint_every_batches=int(payload.get("checkpoint_every_batches", 20)),
        max_chars=payload.get("max_chars"),
        memory_limit=payload.get("memory_limit"),
        query_limit=payload.get("query_limit"),
    )
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if config.checkpoint_every_batches <= 0:
        raise ValueError("checkpoint_every_batches must be positive")
    return config


def load_records(config: EmbeddingConfig) -> list[EmbedRecord]:
    records: list[EmbedRecord] = []
    for row in iter_jsonl(config.memory_path, limit=config.memory_limit):
        text = str(row["text"])
        if config.max_chars is not None:
            text = text[: int(config.max_chars)]
        records.append(EmbedRecord("memory", str(row["unit_id"]), text))
    for row in iter_jsonl(config.query_path, limit=config.query_limit):
        text = str(row["query_text"])
        if config.max_chars is not None:
            text = text[: int(config.max_chars)]
        records.append(EmbedRecord("query", str(row["query_id"]), text))
    return records


def iter_jsonl(path: Path, *, limit: int | None = None):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            yield row
            count += 1
            if limit is not None and count >= int(limit):
                break


def load_existing_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"memory_embeddings": {}, "query_embeddings": {}, "metadata": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    memory = payload.get("memory_embeddings", {})
    query = payload.get("query_embeddings", {})
    if not isinstance(memory, dict) or not isinstance(query, dict):
        raise ValueError(f"{path} must contain memory_embeddings and query_embeddings objects")
    return {
        "memory_embeddings": {str(key): [float(x) for x in value] for key, value in memory.items()},
        "query_embeddings": {str(key): [float(x) for x in value] for key, value in query.items()},
        "metadata": dict(payload.get("metadata", {})),
    }


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def safe_model_key(model: Any) -> str:
    text = str(model).strip().replace("/", "__")
    text = "".join(ch if ch.isalnum() or ch in "._:-" else "-" for ch in text)
    return text.strip("-") or "embedding-model"


def build_metadata(config: EmbeddingConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "model_key": config.model_key,
        "base_url": config.base_url,
        "memory_path": str(config.memory_path),
        "query_path": str(config.query_path),
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "max_chars": config.max_chars,
        "created_by": "dataset/build_embeddings.py",
    }


def batch_records(records: list[EmbedRecord], batch_size: int) -> list[list[EmbedRecord]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def embedding_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


def embed_batch(endpoint: str, config: EmbeddingConfig, batch: list[EmbedRecord]) -> list[list[float]]:
    payload = {
        "model": config.model,
        "input": [record.text for record in batch],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embedding request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"embedding request failed: {exc}") from exc

    data = response_payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("embedding response missing data list")
    data.sort(key=lambda item: int(item.get("index", 0)))
    embeddings: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("embedding response contains an invalid embedding")
        embeddings.append([float(x) for x in embedding])
    return embeddings


def progress_line(
    completed_batches: int,
    total_batches: int,
    pending: list[EmbedRecord],
    started_at: float,
) -> str:
    elapsed = max(0.001, time.time() - started_at)
    rate = completed_batches / elapsed
    return json.dumps(
        {
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "pending_records": len(pending),
            "batches_per_second": round(rate, 3),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
