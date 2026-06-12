"""Actual vLLM runtime client.

This client talks to a vLLM server that exposes the AM cache extensions used by
the visual memory system.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from visual_memory_system.runtime.client import RuntimeClient
from visual_memory_system.schema import CacheSelection, CacheState, ForegroundPlan, RuntimeResponse, VisualPage


class VllmRuntimeClient(RuntimeClient):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        foreground_max_tokens: int = 64,
        background_max_tokens: int = 1,
        background_chunk_tokens: int = 256,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.foreground_max_tokens = foreground_max_tokens
        self.background_max_tokens = background_max_tokens
        self.background_chunk_tokens = background_chunk_tokens

    def select_ready_page(self, page_keys: list[str]) -> CacheSelection:
        if len(page_keys) != len(set(page_keys)):
            raise ValueError("select_ready_page received duplicate page keys")
        if not page_keys:
            return CacheSelection(selected_page_key=None, kv_state=CacheState.MISSING, inspected=())
        candidates = [{"cache_key": page_key} for page_key in page_keys]
        response = self._post_json("/v1/am/cache/select", {"candidates": candidates})
        selected = response.get("selected_cache_key", response.get("selected"))
        kv_state = _parse_cache_state(response.get("kv_state"))
        inspected = tuple(str(x) for x in response.get("inspected", page_keys))
        if selected is not None and str(selected) not in page_keys:
            raise RuntimeError(f"vLLM selected unknown cache key {selected!r}")
        return CacheSelection(
            selected_page_key=None if selected is None else str(selected),
            lease_id=response.get("lease_id"),
            kv_state=kv_state,
            encoder_state=response.get("encoder_state"),
            inspected=inspected,
            metadata=response,
        )

    def send_foreground(self, plan: ForegroundPlan) -> RuntimeResponse:
        if not plan.input_pages:
            raise ValueError(f"foreground plan for {plan.query.query_id} has no input pages")
        for page in plan.input_pages:
            _require_image_path(page)
        if plan.cache_page is not None:
            cache_key = plan.cache_page.page_key
        elif plan.selected_page is not None:
            cache_key = plan.selected_page.page_key
        else:
            cache_key = f"query:{plan.input_pages[0].page_key}:{plan.query.query_id}"
        payload = self._chat_payload(
            pages=plan.input_pages,
            text=plan.query.query_text,
            max_tokens=self.foreground_max_tokens,
            vllm_xargs={
                "am_role": "foreground",
                "am_cache_key": cache_key,
                **({"am_lease_id": plan.lease_id} if plan.lease_id else {}),
            },
        )
        start = time.perf_counter()
        response = self._post_json("/v1/chat/completions", payload)
        client_ttft_ms = (time.perf_counter() - start) * 1000.0
        return _runtime_response(
            request_id=f"fg:{plan.query.query_id}",
            response=response,
            client_ttft_ms=client_ttft_ms,
        )

    def send_background(self, page: VisualPage) -> RuntimeResponse:
        _require_image_path(page)
        payload = self._chat_payload(
            pages=(page,),
            text="",
            max_tokens=self.background_max_tokens,
            vllm_xargs={
                "am_role": "background",
                "am_cache_key": page.page_key,
                "am_prefill_only": 1,
                "am_chunk_tokens": self.background_chunk_tokens,
                "am_decode_overlap": 0,
                "am_pause_on_foreground": 1,
                "am_evictable": 1,
            },
        )
        start = time.perf_counter()
        response = self._post_json("/v1/chat/completions", payload)
        client_ttft_ms = (time.perf_counter() - start) * 1000.0
        return _runtime_response(
            request_id=f"bg:{page.page_key}",
            response=response,
            client_ttft_ms=client_ttft_ms,
        )

    def _chat_payload(
        self,
        *,
        pages: tuple[VisualPage, ...],
        text: str,
        max_tokens: int,
        vllm_xargs: dict[str, Any],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for page in pages:
            image_path = _require_image_path(page)
            content.append({"type": "image_url", "image_url": {"url": image_path.as_uri()}})
        content.append({"type": "text", "text": text})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "vllm_xargs": vllm_xargs,
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {exc.code} for {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to reach vLLM endpoint {self.base_url}{path}: {exc}") from exc
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise RuntimeError(f"vLLM response for {path} is not a JSON object")
        return decoded

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _runtime_response(*, request_id: str, response: dict[str, Any], client_ttft_ms: float) -> RuntimeResponse:
    metrics = response.get("vllm_internal_metrics")
    if not isinstance(metrics, dict) or "engine_queue_to_first_token_ms" not in metrics:
        raise RuntimeError(f"vLLM response lacks engine TTFT metric for {request_id}")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    server_ttft_ms = metrics.get("server_request_to_first_token_ms")
    pre_scheduler_delay_ms = metrics.get("pre_scheduler_delay_ms")
    return RuntimeResponse(
        request_id=request_id,
        engine_ttft_ms=float(metrics["engine_queue_to_first_token_ms"]),
        client_ttft_ms=client_ttft_ms,
        server_ttft_ms=None if server_ttft_ms is None else float(server_ttft_ms),
        pre_scheduler_delay_ms=(
            None if pre_scheduler_delay_ms is None else float(pre_scheduler_delay_ms)
        ),
        prompt_tokens=usage.get("prompt_tokens"),
        output_text=_extract_output_text(response),
        metadata=response,
    )


def _extract_output_text(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return None if content is None else str(content)
    return None


def _require_image_path(page: VisualPage) -> Path:
    if not page.image_path:
        raise ValueError(f"page {page.page_key} is not rendered")
    path = Path(page.image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _parse_cache_state(value: Any) -> CacheState | None:
    if value is None:
        return None
    try:
        return CacheState(str(value))
    except ValueError as exc:
        raise RuntimeError(f"unknown vLLM cache state {value!r}") from exc
