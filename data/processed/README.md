# Processed Data Placeholder

This artifact does not bundle the full processed datasets by default.

Expected layout after running the cleaning and embedding scripts:

```text
data/processed/locomo/
  memory_units.jsonl
  query_records.jsonl
  manifest.json
  embeddings/qwen3-embed-4b.json

data/processed/eventqa/
  memory_units.jsonl
  query_records.jsonl
  manifest.json
  embeddings/qwen3-embed-4b.json

data/processed/perma/
  memory_units.jsonl
  query_records.jsonl
  manifest.json
  embeddings/qwen3-embed-4b.json
```

The `vispage/dataset/*/processed` symlinks point here.
