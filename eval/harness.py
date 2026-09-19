"""Interleaved multi-seed runner.

Every backend runs in the same session, over the same items, in the same order,
against the same spec object. Each backend owns one worker thread and issues its
calls strictly one at a time, so the recorded latencies are real per-call
latencies rather than throughput-under-load. The three threads run concurrently
only so the whole pass fits in one hour on one machine: same machine, same
hour, one interleaved session.

    uv run eval/harness.py --seeds 3 --backends jev,openai,openai-mini
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src import policy, spec
from src.backend import BackendResult, cost_usd
from src.backends.anthropic_llm import AnthropicBackend
from src.backends.jev import JevBackend
from src.backends.openai_llm import OpenAIBackend
from src.env import key, optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
CORPUS = ROOT / "data" / "corpus.jsonl"

BACKENDS = {
    "jev": lambda: JevBackend(key("TYPESAFE_API_KEY"), model="jev-latest"),
    "openai": lambda: OpenAIBackend(key("OPENAI_API_KEY"), "gpt-5.1", reasoning_effort="none"),
    "openai-mini": lambda: OpenAIBackend(key("OPENAI_API_KEY"), "gpt-5-mini", reasoning_effort="minimal"),
    # gpt-6-astra rejects effort "none"; "low" is the floor the API allows.
    "openai-astra": lambda: OpenAIBackend(key("OPENAI_API_KEY"), "gpt-6-astra", reasoning_effort="low"),
    "anthropic-opus": lambda: AnthropicBackend(
        key("ANTHROPIC_API_KEY"), "claude-opus-5", workspace_id=optional("ANTHROPIC_WORKSPACE_ID")
    ),
    "anthropic-haiku": lambda: AnthropicBackend(
        key("ANTHROPIC_API_KEY"), "claude-haiku-4-5-20251001", workspace_id=optional("ANTHROPIC_WORKSPACE_ID")
    ),
}

SAMPLE_SEED = 20260920  # fixed before any backend ran; see RESULTS.md
BASE_ITEMS = 240  # the set the main table ran on; smaller sets nest inside it


def load_items(limit: int | None, path: Path = CORPUS, base: int = BASE_ITEMS) -> list[dict]:
    """Stratified subsample, and *nested*: the 120-item set is a subset of the
    240-item set, so a backend that could only afford half the corpus is still
    comparable to the others on exactly the items it saw."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is None or limit >= len(rows):
        return rows
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    rng = random.Random(SAMPLE_SEED)
    per_class = max(limit, base) // len(by_label)
    sample: list[dict] = []
    for label in sorted(by_label):
        pool = sorted(by_label[label], key=lambda r: r["id"])
        drawn = rng.sample(pool, min(per_class, len(pool)))
        if limit < base:  # nest: keep the first limit/4 of the base draw, by id
            drawn = sorted(drawn, key=lambda r: r["id"])[: limit // len(by_label)]
        sample.extend(drawn)
    sample.sort(key=lambda r: r["id"])
    return sample


def run_backend(name: str, items: list[dict], seeds: int, progress: dict, resume: bool = False) -> None:  # noqa: C901
    backend = BACKENDS[name]()
    # One probe with a one-character document measures the constant part of the
    # prompt: the spec, the schema, and the adapter's framing. That is the prefix
    # a provider cache could serve, and metrics.py uses it for the modelled
    # warm-cache column when the provider reports no cache hits of its own.
    meta_path = RESULTS / name / "meta.json"
    if resume and meta_path.exists():
        probe = None  # keep the recorded prefix rather than paying for it twice
    else:
        probe = backend.ask("x")
    (RESULTS / name).mkdir(parents=True, exist_ok=True)
    if probe is not None:
        meta_path.write_text(
            json.dumps(
                {
                    "model": probe["model"],
                    "prefix_tokens": probe["input_tokens"],
                    "spec_hash": probe["spec_hash"],
                    "seeds": seeds,
                    "n_items": len(items),
                    "item_ids": [i["id"] for i in items],
                    "retry": {"transport_retries": 3, "repair_retries": 0},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    for seed in range(1, seeds + 1):
        out_dir = RESULTS / name / str(seed)
        (out_dir / "raw").mkdir(parents=True, exist_ok=True)
        answers_path = out_dir / "answers.jsonl"
        done: set[str] = set()
        if resume and answers_path.exists():
            done = {json.loads(line)["id"]
                    for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        with answers_path.open("a" if done else "w", encoding="utf-8") as handle:
            for index, item in enumerate(items):
                if item["id"] in done:
                    continue
                result: BackendResult = backend.ask(item["text"])
                if result["spec_hash"] != spec.SPEC_HASH:
                    raise SystemExit(
                        f"SPEC PARITY FAILURE: {name} saw {result['spec_hash']}, "
                        f"expected {spec.SPEC_HASH}. Run invalid."
                    )
                action = policy.decide(result["answers"])
                handle.write(
                    json.dumps(
                        {
                            "id": item["id"],
                            "label": item["label"],
                            "source": item["source"],
                            "action": action,
                            "answers": result["answers"],
                            "input_tokens": result["input_tokens"],
                            "cached_input_tokens": result["cached_input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "latency_ms": result["latency_ms"],
                            "invalid": result["invalid"],
                            "repair_attempts": result["repair_attempts"],
                            "model": result["model"],
                            "cost_cold_usd": cost_usd(result, warm_cache=False),
                            "cost_warm_usd": cost_usd(result, warm_cache=True),
                        }
                    )
                    + "\n"
                )
                handle.flush()
                (out_dir / "raw" / f"{item['id']}.json").write_text(
                    json.dumps(result["raw"], ensure_ascii=False), encoding="utf-8"
                )
                progress[name] = f"seed {seed}/{seeds} item {index + 1}/{len(items)}"
    progress[name] = "done"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--backends", default="jev,openai,openai-mini")
    parser.add_argument("--items", type=int, default=240, help="stratified subsample size; 0 = all")
    parser.add_argument("--resume", action="store_true",
                        help="skip items already recorded in answers.jsonl (an interrupted pass)")
    args = parser.parse_args()

    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    unknown = set(names) - set(BACKENDS)
    if unknown:
        raise SystemExit(f"unknown backends: {sorted(unknown)}")

    items = load_items(args.items or None)
    print(f"spec {spec.SPEC_HASH} | {len(items)} items | {args.seeds} seeds | {names}")

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    progress: dict[str, str] = {n: "starting" for n in names}
    errors: dict[str, str] = {}

    def target(name: str) -> None:
        try:
            run_backend(name, items, args.seeds, progress, resume=args.resume)
        except BaseException as error:  # noqa: BLE001 - recorded, then re-reported
            errors[name] = f"{type(error).__name__}: {error}"
            progress[name] = "FAILED"

    threads = [threading.Thread(target=target, args=(n,), name=n) for n in names]
    for thread in threads:
        thread.start()
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        print(f"  [{time.perf_counter() - clock:7.0f}s] " + " | ".join(f"{n}: {progress[n]}" for n in names), flush=True)
    for thread in threads:
        thread.join()

    finished = datetime.now(timezone.utc)
    run_path = RESULTS / "run.json"
    previous = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    if previous and previous.get("spec_hash") != spec.SPEC_HASH:
        raise SystemExit(
            f"existing results were produced against spec {previous['spec_hash']}, "
            f"this code is {spec.SPEC_HASH}. Delete eval/results/ and rerun."
        )
    windows = previous.get("windows", [])
    if previous and "started_utc" in previous and not windows:
        windows = [{"backends": previous["backends"], "started_utc": previous["started_utc"],
                    "finished_utc": previous["finished_utc"], "wall_clock_s": previous["wall_clock_s"],
                    "n_items": previous["n_items"], "seeds": previous["seeds"]}]
    # A backend re-run replaces its earlier window; the old data is gone.
    windows = [w for w in windows if not set(w["backends"]) <= set(names)]
    windows.append({"backends": names, "started_utc": started.isoformat(),
                    "finished_utc": finished.isoformat(),
                    "wall_clock_s": round(time.perf_counter() - clock, 1),
                    "n_items": len(items), "seeds": args.seeds})
    merged = [n for n in previous.get("backends", []) if n not in names] + names
    run_path.write_text(
        json.dumps(
            {
                "spec_hash": spec.SPEC_HASH,
                "backends": merged,
                "seeds": args.seeds,
                "n_items": len(items),
                "sample_seed": SAMPLE_SEED,
                "windows": windows,
                "started_utc": min(w["started_utc"] for w in windows),
                "finished_utc": max(w["finished_utc"] for w in windows),
                "wall_clock_s": sum(w["wall_clock_s"] for w in windows),
                "machine": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                },
                "policy": {
                    "review_threshold": policy.REVIEW_THRESHOLD,
                    "action_threshold": policy.ACTION_THRESHOLD,
                    "severity_block": policy.SEVERITY_BLOCK,
                },
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwall clock {time.perf_counter() - clock:.0f}s")
    if errors:
        raise SystemExit(f"backend failures: {errors}")


if __name__ == "__main__":
    main()
