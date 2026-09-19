"""Jev backend. Same spec object, one request per message."""

from __future__ import annotations

import time

from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

from src import spec
from src.backend import BackendResult
from src.backends.common import RETRY, empty_answers, normalise


class JevBackend:
    def __init__(self, api_key: str, model: str = "jev-latest") -> None:
        self.name = "jev"
        self.model = model
        self._client = TypeSafeClient(
            api_key=api_key,
            model=model,
            # Identical to the LLM side: transport retries only, no repair loop.
            retry=RetryPolicy(max_retries=RETRY["transport_retries"], timeout=120.0),
        )

    def ask(self, text: str) -> BackendResult:
        state = spec.build_state(text)
        started = time.perf_counter()
        try:
            response = self._client.system_one(state, spec.questions())
        except TypeSafeError as error:
            return BackendResult(
                answers=empty_answers(),
                input_tokens=0,
                cached_input_tokens=0,
                output_tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                invalid=True,
                repair_attempts=0,
                model=self.model,
                spec_hash=spec.SPEC_HASH,
                raw={"state": state, "error": f"{type(error).__name__}: {error}"},
            )
        latency_ms = (time.perf_counter() - started) * 1000
        answers, invalid = normalise(response.answers)
        return BackendResult(
            answers=answers,
            input_tokens=response.usage.input_tokens or 0,
            cached_input_tokens=0,  # TypeSafe publishes no cached-input rate
            output_tokens=response.usage.output_tokens or 0,
            latency_ms=latency_ms,
            invalid=invalid,
            # Structural: the API returns the spec's own types, so there is no
            # schema to violate and nothing to repair. Reported as 0, not measured.
            repair_attempts=0,
            model=response.model,
            spec_hash=spec.SPEC_HASH,
            raw={"state": state, "response": response.model_dump(mode="json")},
        )
