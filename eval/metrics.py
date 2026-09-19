"""Everything in RESULTS.md comes from here. Nothing is hand-typed.

    uv run eval/metrics.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from src import policy
from src.backend import price_for

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"
GATES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
ECE_BINS = 10
REVIEW_BUDGET = 0.05  # share of traffic an ops team will look at by hand


def load(backend: str, seed: int) -> list[dict]:
    path = RESULTS / backend / str(seed) / "answers.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation, no mean, ever."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return ordered[rank - 1]


def _correct(row: dict, expected: dict[str, frozenset[str]]) -> bool:
    return row["action"] in expected[row["label"]]


def strict_accuracy(rows: list[dict], gate: float, expected) -> float:
    """An item is an error if the action is wrong OR confidence is below the gate."""
    return sum(
        _correct(row, expected) and row["answers"]["category"]["confidence"] >= gate
        for row in rows
    ) / len(rows)


def expected_calibration_error(rows: list[dict], bins: int = ECE_BINS) -> tuple[float, list[dict]]:
    """ECE over the Choice: p = probability of the picked category, correct =
    that category's implied action is acceptable for the item's true label."""
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row in rows:
        answer = row["answers"]["category"]
        picked = answer["choice"]
        if picked is None:
            continue
        p = answer["probabilities"][picked]
        implied = policy.CATEGORY_ACTION[picked]
        hit = int(implied in policy.EXPECTED_ACTIONS[row["label"]])
        buckets[min(bins - 1, int(p * bins))].append((p, hit))
    total = sum(len(b) for b in buckets)
    ece, diagram = 0.0, []
    for index, bucket in enumerate(buckets):
        if not bucket:
            diagram.append({"bin": index / bins, "n": 0, "confidence": None, "accuracy": None})
            continue
        confidence = statistics.fmean(p for p, _ in bucket)
        accuracy = statistics.fmean(h for _, h in bucket)
        ece += len(bucket) / total * abs(accuracy - confidence)
        diagram.append(
            {"bin": index / bins, "n": len(bucket), "confidence": confidence, "accuracy": accuracy}
        )
    return ece, diagram


def escalation(rows: list[dict], budget: float, expected) -> dict:
    """Send the least-confident `budget` share to a human who is always right.
    How much of the residual error does that buy?"""
    errors_before = sum(not _correct(row, expected) for row in rows)
    n_escalated = int(len(rows) * budget)
    ranked = sorted(rows, key=lambda r: r["answers"]["category"]["confidence"])
    caught = sum(not _correct(row, expected) for row in ranked[:n_escalated])
    return {
        "budget": budget,
        "n_escalated": n_escalated,
        "errors_before": errors_before,
        "errors_caught": caught,
        "residual_error_caught": caught / errors_before if errors_before else None,
        "accuracy_after": (len(rows) - (errors_before - caught)) / len(rows),
    }


def confusion(rows: list[dict]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        matrix.setdefault(row["label"], {a: 0 for a in ("PASS", "REVIEW", "BLOCK")})
        matrix[row["label"]][row["action"]] += 1
    return matrix


def warm_cost(rows: list[dict], prefix_tokens: int, model: str) -> dict:
    """Cost per 1000 screens, three ways.

    - cold: nothing cached, every input token at the full rate.
    - warm_measured: the cached tokens the provider actually reported.
    - warm_modelled: the whole constant prompt prefix served from cache. This is
      the best case an LLM guardrail can reach, and it is the number to argue
      against, not the cold one.
    """
    price = price_for(model)
    n = len(rows)
    out = sum(r["output_tokens"] for r in rows)
    total_in = sum(r["input_tokens"] for r in rows)
    measured_cached = sum(r["cached_input_tokens"] for r in rows)
    modelled_cached = sum(min(prefix_tokens, r["input_tokens"]) for r in rows)

    def usd(cached: int) -> float:
        return ((total_in - cached) * price["input"] + cached * price["cached_input"]
                + out * price["output"]) / 1_000_000 / n * 1000

    return {
        "cold_usd_per_1k": usd(0),
        "warm_measured_usd_per_1k": usd(measured_cached),
        "warm_modelled_usd_per_1k": usd(modelled_cached),
        "cached_tokens_reported": measured_cached,
        "prefix_tokens": prefix_tokens,
        "mean_input_tokens": total_in / n,
        "mean_output_tokens": out / n,
    }


def summarise_seed(rows: list[dict], prefix_tokens: int) -> dict:
    latencies = [r["latency_ms"] for r in rows]
    model = rows[0]["model"]
    ece, diagram = expected_calibration_error(rows)
    return {
        "model": model,
        "n": len(rows),
        "accuracy": strict_accuracy(rows, 0.0, policy.EXPECTED_ACTIONS),
        "accuracy_strict_borderline": strict_accuracy(rows, 0.0, policy.EXPECTED_ACTIONS_STRICT),
        "gate_sweep": {str(g): strict_accuracy(rows, g, policy.EXPECTED_ACTIONS) for g in GATES},
        "invalid_rate": sum(r["invalid"] for r in rows) / len(rows),
        "mean_repair_attempts": statistics.fmean(r["repair_attempts"] for r in rows),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "cost": warm_cost(rows, prefix_tokens, model),
        "ece": ece,
        "reliability": diagram,
        "escalation": escalation(rows, REVIEW_BUDGET, policy.EXPECTED_ACTIONS),
        "confusion": confusion(rows),
    }


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _get(summary: dict, path: str) -> float:
    node: Any = summary
    for part in path.split("."):
        node = node[part]
    return 0.0 if node is None else node  # a seed with zero errors reports None


AGGREGATE = [
    "accuracy", "accuracy_strict_borderline", "invalid_rate", "mean_repair_attempts",
    "latency_ms.p50", "latency_ms.p95", "latency_ms.p99",
    "cost.cold_usd_per_1k", "cost.warm_measured_usd_per_1k", "cost.warm_modelled_usd_per_1k",
    "cost.mean_input_tokens", "cost.mean_output_tokens",
    "ece", "escalation.residual_error_caught", "escalation.accuracy_after",
]


def disagreements(per_backend: dict[str, list[list[dict]]]) -> list[dict]:
    """Every item where any backend got it wrong, with its runner-up."""
    names = list(per_backend)
    by_id: dict[str, dict] = {}
    for name in names:
        for rows in per_backend[name]:
            for row in rows:
                entry = by_id.setdefault(
                    row["id"], {"id": row["id"], "label": row["label"], "source": row["source"], "backends": {}}
                )
                probs = row["answers"]["category"]["probabilities"]
                ranked = sorted(probs.items(), key=lambda kv: -kv[1])
                entry["backends"].setdefault(name, []).append(
                    {
                        "action": row["action"],
                        "correct": _correct(row, policy.EXPECTED_ACTIONS),
                        "category": row["answers"]["category"]["choice"],
                        "p": ranked[0][1],
                        "runner_up": ranked[1][0],
                        "p_runner_up": ranked[1][1],
                        "confidence": row["answers"]["category"]["confidence"],
                        "severity": row["answers"]["severity"]["score"],
                    }
                )
    out = []
    for entry in by_id.values():
        wrong = {n: sum(not s["correct"] for s in seeds) for n, seeds in entry["backends"].items()}
        if not any(wrong.values()):
            continue
        entry["n_wrong_of_seeds"] = wrong
        entry["n_seeds"] = {n: len(seeds) for n, seeds in entry["backends"].items()}
        out.append(entry)
    out.sort(key=lambda e: (-sum(e["n_wrong_of_seeds"].values()), e["id"]))
    return out


def main() -> None:
    run = json.loads((RESULTS / "run.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {"run": run, "backends": {}}
    per_backend: dict[str, list[list[dict]]] = {}

    for name in run["backends"]:
        meta = json.loads((RESULTS / name / "meta.json").read_text(encoding="utf-8"))
        # Count the seed directories that actually exist rather than trusting a
        # run-level number: passes ran with different seed counts.
        n_seeds = len([d for d in (RESULTS / name).iterdir()
                       if d.is_dir() and d.name.isdigit() and (d / "answers.jsonl").exists()])
        seeds = [load(name, s) for s in range(1, n_seeds + 1)]
        per_backend[name] = seeds
        summaries = [summarise_seed(rows, meta["prefix_tokens"]) for rows in seeds]
        for seed_index, summary in enumerate(summaries, start=1):
            (RESULTS / name / str(seed_index) / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        report["backends"][name] = {
            "model": summaries[0]["model"],
            "n_seeds": n_seeds,
            "n_items": meta.get("n_items", len(seeds[0])),
            "meta": {k: v for k, v in meta.items() if k != "item_ids"},
            "per_seed": summaries,
            "aggregate": {path: _mean_std([_get(s, path) for s in summaries]) for path in AGGREGATE},
            "gate_sweep": {
                g: _mean_std([s["gate_sweep"][g] for s in summaries]) for g in map(str, GATES)
            },
            "per_label_accuracy": {
                label: _mean_std(
                    [
                        sum(_correct(r, policy.EXPECTED_ACTIONS) for r in rows if r["label"] == label)
                        / max(1, sum(r["label"] == label for r in rows))
                        for rows in seeds
                    ]
                )
                for label in sorted(policy.EXPECTED_ACTIONS)
            },
        }

    # Backends that could only afford part of the corpus still get a like-for-like
    # row: every backend re-scored on exactly the item ids the smallest run saw.
    common = set.intersection(*(set(r["id"] for r in seeds[0]) for seeds in per_backend.values()))
    report["common_subset"] = {
        "n_items": len(common),
        "backends": {
            name: {
                "accuracy": _mean_std([
                    strict_accuracy([r for r in rows if r["id"] in common], 0.0, policy.EXPECTED_ACTIONS)
                    for rows in seeds
                ]),
                "ece": _mean_std([
                    expected_calibration_error([r for r in rows if r["id"] in common])[0] for rows in seeds
                ]),
            }
            for name, seeds in per_backend.items()
        },
    }

    report["disagreements"] = disagreements(per_backend)
    adversarial = RESULTS / "adversarial.json"
    if adversarial.exists():
        report["adversarial"] = json.loads(adversarial.read_text(encoding="utf-8"))

    (RESULTS / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = f"{'backend':<14}{'acc':>16}{'ECE':>14}{'p50 ms':>10}{'p99 ms':>10}{'$/1k cold':>12}{'$/1k warm':>12}"
    print(header)
    print("-" * len(header))
    for name, data in report["backends"].items():
        agg = data["aggregate"]
        print(
            f"{name:<14}"
            f"{agg['accuracy']['mean']:>10.1%} ±{agg['accuracy']['std']:.1%}"
            f"{agg['ece']['mean']:>9.3f} ±{agg['ece']['std']:.3f}"
            f"{agg['latency_ms.p50']['mean']:>10.0f}"
            f"{agg['latency_ms.p99']['mean']:>10.0f}"
            f"{agg['cost.cold_usd_per_1k']['mean']:>12.4f}"
            f"{agg['cost.warm_modelled_usd_per_1k']['mean']:>12.4f}"
        )
    print(f"\n{len(report['disagreements'])} items with at least one error -> summary.json")


if __name__ == "__main__":
    main()
