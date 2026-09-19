"""Every backend must have seen an identical spec. Run: uv run tests/test_spec_parity.py"""

from __future__ import annotations

import json
from pathlib import Path

from src import policy, spec

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"


def test_spec_is_frozen() -> None:
    try:
        spec.QUESTIONS["category"] = None  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("spec.QUESTIONS is mutable")


def test_backends_import_the_same_objects() -> None:
    """The backends must not own a copy of the spec or the policy."""
    from src.backends import jev, openai_llm

    assert jev.spec is spec and openai_llm.spec is spec
    assert jev.spec.QUESTIONS is openai_llm.spec.QUESTIONS
    src = (Path(__file__).resolve().parent.parent / "src" / "backends").glob("*.py")
    for path in src:
        text = path.read_text(encoding="utf-8")
        assert "instructions=" not in text, f"{path.name} defines its own question text"


def test_hash_parity_across_recorded_runs() -> None:
    run = RESULTS / "run.json"
    if not run.exists():
        print("  (no run yet; parity of recorded runs not checked)")
        return
    recorded = json.loads(run.read_text(encoding="utf-8"))
    assert recorded["spec_hash"] == spec.SPEC_HASH, (
        f"run was made against spec {recorded['spec_hash']}, code is now "
        f"{spec.SPEC_HASH}. The run is invalid against this spec."
    )
    for backend in recorded["backends"]:
        for seed in range(1, recorded["seeds"] + 1):
            path = RESULTS / backend / str(seed) / "answers.jsonl"
            assert path.exists(), f"missing {path}"


def test_policy_thresholds_are_shared() -> None:
    text = (Path(__file__).resolve().parent.parent / "src" / "backends").glob("*.py")
    for path in text:
        body = path.read_text(encoding="utf-8")
        for name in ("REVIEW_THRESHOLD", "ACTION_THRESHOLD", "SEVERITY_BLOCK"):
            assert name not in body, f"{path.name} redefines {name}"
    assert 0 < policy.REVIEW_THRESHOLD < policy.ACTION_THRESHOLD < 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print(f"spec hash {spec.SPEC_HASH}")
