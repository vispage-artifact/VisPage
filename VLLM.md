# Modified vLLM Runtime

VisPage requires a modified vLLM runtime with agent-memory page-cache support.
This release provides a single patch:

```text
vllm_patch/vllm_vispage_artifact.patch
```

The patch contains the AM cache registry, cache-select API, encoder cache
offload, timing trace, chunked page-computation scheduling changes, and the
request-arrival-to-first-token TTFT metric used by the final experiments.

## Expected Runtime Features

- Chat completions accept visual page cache metadata.
- `/v1/am/cache/select` returns runtime page-cache readiness for candidate pages.
- Page-computation requests can be submitted as low-priority chunked requests.
- Responses expose TTFT timing metadata used by VisPage traces.

## Example Deployment

Use the env templates under:

```text
vllm_patch/deploy_examples/
```

Important fields to set:

```text
VLLM_REPO=/path/to/vllm-cache-prefetch
MODEL_PATH=/path/to/qwen3-vl-8b-or-32b
ALLOWED_LOCAL_MEDIA_PATH=/path/to/vispage-artifact-release
RUN_DIR=/path/to/vllm-deploy/run
LOG_DIR=/path/to/vllm-deploy/logs
```

The `ALLOWED_LOCAL_MEDIA_PATH` must cover the rendered page image output path.

## Patch Application

Apply the patch to the base vLLM source used for the artifact:

```bash
cd /path/to/vllm-cache-prefetch
git checkout <BASE_VLLM_COMMIT>
git apply /path/to/vispage-artifact-release/vllm_patch/vllm_vispage_artifact.patch
```

The artifact keeps the base commit as a placeholder for anonymization. Fill it
with the released vLLM base commit before public release.
