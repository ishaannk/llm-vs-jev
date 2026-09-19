"""How does cost scale with the number of judgments per message?

This is the structural claim worth measuring: Jev ingests `state` once and
evaluates every question against it in parallel, so a sixth question costs
almost nothing extra. An LLM pays for a longer schema on input and a longer
answer on output. One spec, prefixes of its question list, same items.

Separate from the main benchmark: it varies the question set on purpose, so it
is never mixed into the accuracy numbers.

    uv run eval/cost_curve.py --items 20
"""

from __future__ import annotations

import argparse
import json
import time

from src import spec
from src.backend import price_for

from eval.harness import BACKENDS, RESULTS, load_items

QUESTION_ORDER = ["category", "severity", "instruction_to_model",
                  "requests_disallowed_content", "contains_pii"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--backends", default="jev,openai,openai-mini")
    args = parser.parse_args()

    items = load_items(args.items)
    full = dict(spec.QUESTIONS)
    original = spec.QUESTIONS
    curve: dict[str, list[dict]] = {}
    started = time.perf_counter()

    for name in (n.strip() for n in args.backends.split(",") if n.strip()):
        backend = BACKENDS[name]()
        curve[name] = []
        for k in range(1, len(QUESTION_ORDER) + 1):
            subset = {qid: full[qid] for qid in QUESTION_ORDER[:k]}
            spec.QUESTIONS = subset  # type: ignore[assignment]
            try:
                results = [backend.ask(item["text"]) for item in items]
            finally:
                spec.QUESTIONS = original  # type: ignore[assignment]
            model = results[0]["model"]
            price = price_for(model)
            n = len(results)
            total_in = sum(r["input_tokens"] for r in results)
            total_out = sum(r["output_tokens"] for r in results)
            curve[name].append(
                {
                    "n_questions": k,
                    "model": model,
                    "mean_input_tokens": total_in / n,
                    "mean_output_tokens": total_out / n,
                    "usd_per_1k": (total_in * price["input"] + total_out * price["output"]) / 1e6 / n * 1000,
                    "p50_latency_ms": sorted(r["latency_ms"] for r in results)[n // 2],
                    "invalid": sum(r["invalid"] for r in results),
                }
            )
            point = curve[name][-1]
            print(f"{name:<14} q={k}  ${point['usd_per_1k']:.4f}/1k  "
                  f"in {point['mean_input_tokens']:.0f} out {point['mean_output_tokens']:.0f}  "
                  f"p50 {point['p50_latency_ms']:.0f}ms", flush=True)

    (RESULTS / "cost_curve.json").write_text(
        json.dumps({"n_items": len(items), "question_order": QUESTION_ORDER, "curve": curve}, indent=2),
        encoding="utf-8",
    )
    print(f"\n{time.perf_counter() - started:.0f}s -> eval/results/cost_curve.json")


if __name__ == "__main__":
    main()
