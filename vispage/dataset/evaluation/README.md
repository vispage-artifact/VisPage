# Dataset Locality Evaluation

`profile_locality.py` estimates workload locality offline from the system's
retrieved top-k memory units. It does not call vLLM.

Example:

```bash
PYTHONPATH=src .conda-memoryagent/bin/python dataset/evaluation/profile_locality.py \
  --memory dataset/locomo/processed/memory_units.jsonl \
  --queries dataset/locomo/processed/query_records.jsonl \
  --embeddings dataset/locomo/processed/embeddings/qwen3-embed-4b.json \
  --output-dir paper_results/locality_profiles/locomo_topk10 \
  --dataset-name locomo \
  --topk 10 \
  --bin-size 50
```

Outputs:

```text
summary.json
summary.md
bins.jsonl
per_query.jsonl
```

Main signals:

- `unit_overlap_at_1`: adjacent concrete memory-unit reuse.
- `recent_unit_coverage_at_10`: how much current top-k is covered by the last
  10 queries' retrieved units; this estimates temporal/path locality.
- `recent_unique_ratio_at_10`: recent working-set growth relative to top-k; high
  values indicate append pressure.
- `centroid_sim_at_1` and `centroid_sim_max_at_5`: near semantic locality in the
  retrieved embedding region.
- `high_sem_low_unit_at_1`: adjacent semantic similarity without concrete unit
  reuse; this is an embedding-friendly pattern.
- `far_centroid_sim_gap_<N>` and `cluster_revisit_gap_gt_<N>`: longer-range
  semantic recurrence.
