"""Shared across backends: the retry policy, and one answer shape.

Kept here so the retry policy is provably identical on both sides and the
policy layer never has to know which backend produced an answer.
"""

from __future__ import annotations

from typing import Any

from src import spec

#: Identical for every backend, and logged into every summary.
#: `transport_retries` covers 429/5xx/timeouts only. `repair_retries` is the
#: number of corrective re-prompts allowed after a schema-invalid response; it
#: is zero so invalid output is counted rather than silently fixed.
RETRY = {"transport_retries": 3, "repair_retries": 0}


def empty_answers() -> dict[str, Any]:
    """The neutral assessment used when a call fails outright.

    Maximum uncertainty: flat over the categories, flat over severity, 0.5 on
    each Noul. An invalid call is scored as an error either way; this just keeps
    downstream arithmetic total.
    """
    cats = spec.CATEGORIES
    flat = {c: 1 / len(cats) for c in cats}
    return {
        "category": {"choice": None, "probabilities": flat, "confidence": 0.0},
        "severity": {
            "score": 1.5,
            "probabilities": {str(i): 0.25 for i in range(4)},
            "confidence": 0.0,
        },
        **{nid: {"noul": 0.5} for nid in spec.NOUL_IDS},
    }


def normalise(answers: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """SDK answers -> plain dicts. Returns (answers, invalid).

    `invalid` is True when a question is missing or a Choice came back with a
    label that is not in the spec. Both are off-spec output, and both are
    counted rather than repaired: a silent retry would hide the thing we are
    trying to measure.
    """
    out: dict[str, Any] = {}
    invalid = False
    for qid in spec.QUESTIONS:
        answer = answers.get(qid)
        if answer is None:
            invalid = True
            continue
        if qid == "category":
            probabilities = dict(answer.probabilities or {})
            if answer.choice not in spec.CATEGORIES or set(probabilities) != set(
                spec.CATEGORIES
            ):
                # Off-spec label or distribution: count it and fall back to the
                # neutral prior rather than feeding a broken dict to the policy.
                invalid = True
                continue
            out[qid] = {
                "choice": answer.choice,
                "probabilities": probabilities,
                "confidence": answer.confidence,
            }
        elif qid == "severity":
            out[qid] = {
                "score": answer.score,
                "probabilities": {str(k): v for k, v in (answer.probabilities or {}).items()},
                "confidence": answer.confidence,
            }
        else:
            out[qid] = {"noul": answer.noul}
    if invalid:
        neutral = empty_answers()
        return {**neutral, **out}, True
    return out, False
