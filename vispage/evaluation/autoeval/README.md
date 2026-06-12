# AutoEval Session Runner

This is a glue runner for long evaluations. It keeps existing experiment
components unchanged and adds session-level checkpointing.

## Run

```bash
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --config-list evaluation/autoeval/configs_exp1_main.txt \
  --output-root paper_results/evaluation/exp1 \
  --progress-every 20 \
  --session-retries 2 \
  --retry-sleep 10 \
  --max-visual-tokens 60000
```

Configs are executed in the order provided. Each config is split by
`QueryRecord.task_id`; every task/session gets a fresh `SystemRunner`.
Before a config runs, autoeval writes `preflight.json` and checks dataset paths,
embedding coverage, query/session validity, visual budget estimates, and vLLM
reachability. Preflight failures skip that config and continue to the next one.

The runner continues by default after a session/config failure. Use
`--stop-on-failure` only when you want fail-fast behavior. Completed sessions are
checkpointed independently, so a failed session can be retried later without
rerunning already completed sessions.

Use `--fail-on-background-errors` if background prewarm failures should make the
session retry/fail. Without it, background errors are recorded in summaries but
do not invalidate foreground results.

Canonical Exp1 configs can be regenerated with:

```bash
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/generate_exp1_configs.py
```

## Resume

```bash
PYTHONPATH=src .conda-memoryagent/bin/python evaluation/autoeval/run_sessions.py \
  --resume-batch-dir paper_results/evaluation/exp1/autoeval/20260607_230000
```

Completed sessions have `sessions/<task_id>/completed.json` and are skipped.
Incomplete sessions are replayed from the session start, so no vLLM cache state
needs to be restored.

## Outputs

```text
<output-root>/autoeval/<batch-timestamp>/
  batch_manifest.json
  config_000_<config-stem>/
    manifest.json
    run_config.json
    preflight.json
    session_index.jsonl
    trace.jsonl
    pages.jsonl
    aggregate_summary.json
    completed.json
    heartbeat.json
    sessions/<task-id>/
      input/memory_units.jsonl
      input/query_records.jsonl
      attempts/attempt_001/error.json
      trace.jsonl
      pages.jsonl
      summary.json
      completed.json
```

Aggregate `trace.jsonl` adds `task_id`, `session_position`, `local_position`,
`global_position`, `selected_page_source`, `selected_page_root_source`,
`selected_page_submitted_at`, and `selected_page_unit_count` to each query row.
Each query row also includes `output_text` for later quality checks.

Query-row `metadata` includes `cache_select_reason` and
`cache_select_inspected` when vLLM returns cache-select diagnostics. The
inspected list is useful for distinguishing candidate states such as missing,
warming, partial, evicted, and ready.

`heartbeat.json` is updated on progress, session completion, and failed
attempts. It records the current session/query, warm/fallback counts, average
engine TTFT, registered pages, inflight background count, and visual-budget
fallback count.
