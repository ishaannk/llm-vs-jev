"""Self-check for the non-obvious arithmetic in metrics.py."""

from __future__ import annotations

from eval.metrics import escalation, expected_calibration_error, percentile, strict_accuracy
from src import policy


def row(label: str, action: str, picked: str, p: float, confidence: float) -> dict:
    rest = (1 - p) / 3
    probs = {c: rest for c in ("benign", "prompt_injection", "jailbreak", "disallowed_request")}
    probs[picked] = p
    return {
        "label": label,
        "action": action,
        "answers": {"category": {"choice": picked, "probabilities": probs, "confidence": confidence}},
    }


def test_percentile_is_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 0.50) == 50
    assert percentile(values, 0.95) == 95
    assert percentile(values, 0.99) == 99
    assert percentile([7.0], 0.99) == 7.0


def test_strict_accuracy_gates_on_confidence() -> None:
    rows = [row("benign", "PASS", "benign", 0.9, 0.9), row("benign", "PASS", "benign", 0.6, 0.6)]
    assert strict_accuracy(rows, 0.0, policy.EXPECTED_ACTIONS) == 1.0
    assert strict_accuracy(rows, 0.8, policy.EXPECTED_ACTIONS) == 0.5
    wrong = [row("injection", "PASS", "benign", 0.9, 0.9)]
    assert strict_accuracy(wrong, 0.0, policy.EXPECTED_ACTIONS) == 0.0


def test_borderline_strict_reading_is_harsher() -> None:
    rows = [row("borderline", "PASS", "benign", 0.9, 0.9)]
    assert strict_accuracy(rows, 0.0, policy.EXPECTED_ACTIONS) == 1.0
    assert strict_accuracy(rows, 0.0, policy.EXPECTED_ACTIONS_STRICT) == 0.0


def test_ece_is_zero_when_probability_matches_outcome() -> None:
    # 10 items in the 0.9 bin, 9 of them right: perfectly calibrated there.
    rows = [row("benign", "PASS", "benign", 0.9, 0.9) for _ in range(9)]
    rows += [row("injection", "PASS", "benign", 0.9, 0.9)]
    ece, _ = expected_calibration_error(rows)
    assert ece < 1e-9, ece
    # Same confidence, all wrong: ECE is the full gap.
    allwrong = [row("injection", "PASS", "benign", 0.9, 0.9) for _ in range(10)]
    ece2, _ = expected_calibration_error(allwrong)
    assert abs(ece2 - 0.9) < 1e-9, ece2


def test_escalation_catches_the_least_confident_errors() -> None:
    rows = [row("benign", "PASS", "benign", 0.99, 0.99) for _ in range(90)]
    rows += [row("injection", "PASS", "benign", 0.5, 0.1) for _ in range(10)]  # wrong, unconfident
    result = escalation(rows, 0.05, policy.EXPECTED_ACTIONS)
    assert result["errors_before"] == 10
    assert result["errors_caught"] == 5
    assert result["accuracy_after"] == 0.95


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
