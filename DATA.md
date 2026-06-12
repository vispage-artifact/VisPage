# Data Notes

The release does not bundle full normalized data by default, because the source
datasets are public third-party datasets and redistribution depends on their
original licenses and hosting terms. Instead, it includes cleaning scripts,
schema examples, and experiment traces.

After data preparation, normalized data should be placed under:

```text
data/processed/locomo/
data/processed/eventqa/
data/processed/perma/
```

Each processed workload should have:

```text
memory_units.jsonl             normalized memory items
query_records.jsonl            workflow queries and evidence metadata
manifest.json                  cleaning metadata
embeddings/qwen3-embed-4b.json embedding cache for retrieval and layout
```

Schema examples are provided under:

```text
data/schema_examples/
```

The cleaning scripts are kept under `vispage/dataset/<workload>/clean.py`.

The experiment traces under `results/traces/` are included so that the reported
latency, reuse-path, session-level speedup, and microbenchmark numbers can be
audited without redistributing the full processed datasets.

PERMA quality in the paper uses an original candidate-selection sanity run for
one sampled session. The artifact includes the resulting trace and quality
summary, but not the temporary derived query file.
