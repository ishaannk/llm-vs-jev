"""The one shape every backend returns, and the prices used to cost it."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class BackendResult(TypedDict):
    answers: dict[str, Any]  # per question: value + probabilities + confidence
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    latency_ms: float
    invalid: bool  # unparseable / schema-violating / off-spec label / API failure
    repair_attempts: int  # retries needed to get valid structure
    model: str  # the version string that actually answered
    spec_hash: str  # what this backend was handed; the harness asserts parity
    raw: dict[str, Any]  # request + response, written to raw/ untouched


class Backend(Protocol):
    name: str
    model: str

    def ask(self, text: str) -> BackendResult: ...


class Price(TypedDict):
    input: float  # USD per 1M input tokens
    cached_input: float  # USD per 1M cached input tokens
    output: float  # USD per 1M output tokens


#: OpenAI: developers.openai.com/api/docs/pricing, read 2026-09-20.
#: Anthropic: platform.claude.com/docs/en/about-claude/pricing, read 2026-09-20.
#: TypeSafe: docs.typesafe.ai/models — $42/Btok input, output free.
#: `cached_input` is the cache-*read* rate. Cache *writes* cost 1.25x base on
#: Anthropic and are billed here at the base rate instead; with one 900-token
#: prefix over hundreds of calls that understates the warm column by <$0.002.
PRICES: dict[str, Price] = {
    "jev-1.13.0": {"input": 0.042, "cached_input": 0.042, "output": 0.0},
    "jev-latest": {"input": 0.042, "cached_input": 0.042, "output": 0.0},
    "gpt-6-astra": {"input": 10.00, "cached_input": 1.00, "output": 50.00},
    "gpt-5.6-sol": {"input": 4.00, "cached_input": 0.40, "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "claude-opus-5": {"input": 5.00, "cached_input": 0.50, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "cached_input": 0.20, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "cached_input": 0.10, "output": 5.00},
    "claude-fable-5-1": {"input": 10.00, "cached_input": 0.25, "output": 50.00},
}


def price_for(model: str) -> Price:
    for key, price in PRICES.items():
        if model.startswith(key):
            return price
    raise KeyError(f"no published price for {model!r}; add it to PRICES")


def cost_usd(result: BackendResult, *, warm_cache: bool) -> float:
    """Cost of one call.

    Cold: every input token billed at the full rate, which is what the first
    caller of a fresh prompt prefix pays. Warm: the tokens the provider actually
    reported as cached are billed at the cached rate. Both are reported, because
    caching is the single biggest lever on the LLM's apparent cost.
    """
    price = price_for(result["model"])
    cached = result["cached_input_tokens"] if warm_cache else 0
    uncached = result["input_tokens"] - cached
    return (
        uncached * price["input"]
        + cached * price["cached_input"]
        + result["output_tokens"] * price["output"]
    ) / 1_000_000
