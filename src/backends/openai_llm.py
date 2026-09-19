"""LLM backend, through TypeSafe's own `system-one-adapter`.

Using the vendor's published adapter (github.com/typesafe-ai/system-one-adapter-python)
is deliberate: they state it is the most accurate way they have found to get
decisions out of an LLM, so nobody can say the LLM was sandbagged. It renders
the *same* `spec.QUESTIONS` object into a prompt, asks for a full probability
distribution per question, and enforces the answer with OpenAI's native strict
JSON-schema structured outputs.

The only thing this module adds is a provider subclass that (a) pins reasoning
effort so the comparison is a System-One-shaped call rather than a chain of
thought billed at output rates, and (b) reads back the cached-token count that
the stock provider drops, because the warm-cache cost column needs it.
"""

from __future__ import annotations

import time
from typing import Any

from system_one_adapter import RetryPolicy, SystemOneAdapterClient
from system_one_adapter.providers.openai import OpenAIProvider
from system_one_adapter.providers.base import (
    Message,
    ProviderResult,
    record_request,
    record_response,
    render_messages,
    translating,
)
from typesafe_sdk import TypeSafeError

from src import spec
from src.backend import BackendResult
from src.backends.common import RETRY, empty_answers, normalise

#: Reported alongside every number: how many input tokens the provider served
#: from its prompt cache on the last call.
_last_cached_tokens = {"value": 0}


class _CachedTokenProvider(OpenAIProvider):
    """OpenAI Responses provider that pins reasoning effort and reports cache hits."""

    def __init__(self, model_name: str, *, api_key: str, reasoning_effort: str | None):
        super().__init__(model_name, api_key=api_key, api="responses")
        self.reasoning_effort = reasoning_effort

    def request(self, messages: list[Message], *, schema: dict[str, Any], structured: bool) -> ProviderResult:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "instructions": "\n\n".join(m.content for m in messages if m.role == "system"),
            "input": render_messages([m for m in messages if m.role != "system"]),
            "text": {"format": {"type": "json_schema", "name": "evaluation", "schema": schema, "strict": True}},
            "store": False,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        record_request(kwargs, api="responses")
        with translating(self.translate_error):
            response = self._client.responses.create(**kwargs)
        record_response(response, finish_reason=response.status)
        if response.status != "completed":
            raise TypeSafeError(f"OpenAI response did not complete: {response.status}")
        details = getattr(response.usage, "input_tokens_details", None)
        _last_cached_tokens["value"] = getattr(details, "cached_tokens", 0) or 0
        return ProviderResult(
            text=response.output_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIBackend:
    def __init__(self, api_key: str, model: str, *, reasoning_effort: str | None = None) -> None:
        self.name = f"openai-{model}"
        self.model = model
        self._provider = _CachedTokenProvider(model, api_key=api_key, reasoning_effort=reasoning_effort)
        self._client = SystemOneAdapterClient(
            structured_outputs=True,       # native strict JSON schema
            llm_answer_mode="probabilities",  # comparable to Jev's distributions
            normalize_probabilities=True,
            n_retry_malformed_structure=RETRY["repair_retries"],
            retry=RetryPolicy(max_retries=RETRY["transport_retries"], timeout=180.0),
        )

    def ask(self, text: str) -> BackendResult:
        state = spec.build_state(text)
        _last_cached_tokens["value"] = 0
        started = time.perf_counter()
        try:
            response = self._client.system_one(state, spec.questions(), model=self._provider)
        except TypeSafeError as error:
            return BackendResult(
                answers=empty_answers(),
                input_tokens=0,
                cached_input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                invalid=True,
                repair_attempts=getattr(error, "debug", {}).get("n_retries_malformed_structure", 0),
                model=self.model,
                spec_hash=spec.SPEC_HASH,
                raw={"state": state, "error": f"{type(error).__name__}: {error}",
                     "attempts": getattr(error, "debug", {}).get("llm_attempts", [])},
            )
        latency_ms = (time.perf_counter() - started) * 1000
        answers, invalid = normalise(response.answers)
        usage = response.usage
        return BackendResult(
            answers=answers,
            input_tokens=usage.input_tokens_total,
            cached_input_tokens=_last_cached_tokens["value"],
            output_tokens=usage.output_tokens_total,
            latency_ms=latency_ms,
            invalid=invalid,
            repair_attempts=usage.n_retries_malformed_structure,
            model=response.model,
            spec_hash=spec.SPEC_HASH,
            raw={"state": state, "response": response.model_dump(mode="json")},
        )
