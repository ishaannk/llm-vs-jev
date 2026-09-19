"""Anthropic backend, through the same `system-one-adapter`.

Same spec object, same adapter, same probability mode as the OpenAI side. The
provider subclass adds three things the stock one does not:

- the `anthropic-workspace-id` header, required by org-scoped keys;
- `thinking: disabled`, so this is a System-One-shaped call rather than a chain
  of thought billed at output rates — the same choice made for `reasoning_effort`
  on the OpenAI side, and the only fair way to compare a per-message guardrail;
- `cache_control` on the constant system prefix, which makes the warm-cache
  column a *measured* number here rather than a modelled one.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
from system_one_adapter import RetryPolicy, SystemOneAdapterClient
from system_one_adapter.providers.anthropic import AnthropicProvider
from system_one_adapter.providers.base import Message, ProviderResult, record_request, record_response, translating
from typesafe_sdk import TypeSafeError

from src import spec
from src.backend import BackendResult
from src.backends.common import RETRY, empty_answers, normalise

_last_cache = {"read": 0, "write": 0}


class _CachingProvider(AnthropicProvider):
    def __init__(self, model_name: str, *, api_key: str, workspace_id: str | None, max_tokens: int = 2048):
        super().__init__(model_name, max_tokens=max_tokens)
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0, default_headers=headers)

    def request(self, messages: list[Message], *, schema: dict[str, Any], structured: bool) -> ProviderResult:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            # One cache breakpoint at the end of the constant prefix. Everything
            # after it (the document) differs per message and is never cached.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": m.role, "content": m.content} for m in messages if m.role != "system"],
            "thinking": {"type": "disabled"},
        }
        if structured:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        record_request(kwargs, api="messages")
        with translating(self.translate_error):
            response = self._client.messages.create(**kwargs)
        record_response(response, finish_reason=response.stop_reason)
        if response.stop_reason == "max_tokens":
            raise TypeSafeError("Anthropic response truncated at the output token limit.")
        usage = response.usage
        _last_cache["read"] = getattr(usage, "cache_read_input_tokens", 0) or 0
        _last_cache["write"] = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return ProviderResult(
            text="".join(b.text for b in response.content if b.type == "text"),
            # Anthropic reports cached tokens outside `input_tokens`; add them
            # back so the token column means the same thing as OpenAI's.
            input_tokens=usage.input_tokens + _last_cache["read"] + _last_cache["write"],
            output_tokens=usage.output_tokens,
        )


class AnthropicBackend:
    def __init__(self, api_key: str, model: str, *, workspace_id: str | None = None) -> None:
        self.name = f"anthropic-{model}"
        self.model = model
        self._provider = _CachingProvider(model, api_key=api_key, workspace_id=workspace_id)
        self._client = SystemOneAdapterClient(
            structured_outputs=True,
            llm_answer_mode="probabilities",
            normalize_probabilities=True,
            n_retry_malformed_structure=RETRY["repair_retries"],
            retry=RetryPolicy(max_retries=RETRY["transport_retries"], timeout=180.0),
        )

    def ask(self, text: str) -> BackendResult:
        state = spec.build_state(text)
        _last_cache.update(read=0, write=0)
        started = time.perf_counter()
        try:
            response = self._client.system_one(state, spec.questions(), model=self._provider)
        except TypeSafeError as error:
            return BackendResult(
                answers=empty_answers(), input_tokens=0, cached_input_tokens=0, output_tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000, invalid=True,
                repair_attempts=getattr(error, "debug", {}).get("n_retries_malformed_structure", 0),
                model=self.model, spec_hash=spec.SPEC_HASH,
                raw={"state": state, "error": f"{type(error).__name__}: {error}",
                     "attempts": getattr(error, "debug", {}).get("llm_attempts", [])},
            )
        latency_ms = (time.perf_counter() - started) * 1000
        answers, invalid = normalise(response.answers)
        usage = response.usage
        return BackendResult(
            answers=answers,
            input_tokens=usage.input_tokens_total,
            cached_input_tokens=_last_cache["read"],
            output_tokens=usage.output_tokens_total,
            latency_ms=latency_ms,
            invalid=invalid,
            repair_attempts=usage.n_retries_malformed_structure,
            model=response.model,
            spec_hash=spec.SPEC_HASH,
            raw={"state": state, "response": response.model_dump(mode="json"),
                 "cache": dict(_last_cache)},
        )
