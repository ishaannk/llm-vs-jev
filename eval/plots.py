"""Reliability diagram, gate sweep, cost-vs-questions. Reads summary.json only.

    uv run eval/plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"
PLOTS = RESULTS / "plots"
COLOURS = {"jev": "#1b7f5f", "openai": "#3b5bdb", "openai-mini": "#c2410c"}


def _colour(name: str) -> str:
    return COLOURS.get(name, "#555555")


def reliability(report: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", c="#999", lw=1, label="perfect calibration")
    for name, data in report["backends"].items():
        points = data["per_seed"][0]["reliability"]
        xs = [p["confidence"] for p in points if p["n"]]
        ys = [p["accuracy"] for p in points if p["n"]]
        sizes = [12 + 3 * p["n"] ** 0.7 for p in points if p["n"]]
        ece = data["aggregate"]["ece"]["mean"]
        ax.plot(xs, ys, "-o", c=_colour(name), ms=4, lw=1.6, label=f"{name}  ECE={ece:.3f}")
        ax.scatter(xs, ys, s=sizes, c=_colour(name), alpha=0.25, edgecolors="none")
    ax.set_xlabel("stated probability of the chosen category")
    ax.set_ylabel("share actually acceptable")
    ax.set_title("Reliability (marker area = items in bin)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "reliability.png", dpi=160)
    plt.close(fig)


def gate_sweep(report: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for name, data in report["backends"].items():
        gates = sorted(float(g) for g in data["gate_sweep"])
        means = [data["gate_sweep"][str(g)]["mean"] for g in gates]
        stds = [data["gate_sweep"][str(g)]["std"] for g in gates]
        ax.errorbar(gates, means, yerr=stds, fmt="-o", ms=4, capsize=3, c=_colour(name), label=name)
    ax.set_xlabel("confidence gate (below it, the item counts as an error)")
    ax.set_ylabel("strict accuracy")
    ax.set_title("Strict accuracy vs confidence gate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "gate_sweep.png", dpi=160)
    plt.close(fig)


def cost_curve() -> None:
    path = RESULTS / "cost_curve.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for name, points in data["curve"].items():
        ks = [p["n_questions"] for p in points]
        ax.plot(ks, [p["usd_per_1k"] for p in points], "-o", ms=4, c=_colour(name), label=name)
        bx.plot(ks, [p["p50_latency_ms"] for p in points], "-o", ms=4, c=_colour(name), label=name)
    ax.set_yscale("log")
    ax.set_xlabel("judgments per message")
    ax.set_ylabel("USD per 1000 screens (log)")
    ax.set_title("Cost vs number of judgments")
    bx.set_xlabel("judgments per message")
    bx.set_ylabel("p50 latency (ms)")
    bx.set_title("Latency vs number of judgments")
    for axis in (ax, bx):
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "cost_curve.png", dpi=160)
    plt.close(fig)


def confusion(report: dict) -> None:
    names = list(report["backends"])
    labels = ["benign", "borderline", "injection", "jailbreak"]
    actions = ["PASS", "REVIEW", "BLOCK"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4))
    axes = axes if len(names) > 1 else [axes]
    for ax, name in zip(axes, names):
        matrix = report["backends"][name]["per_seed"][0]["confusion"]
        grid = [[matrix.get(label, {}).get(action, 0) for action in actions] for label in labels]
        total = max(1, max(max(row) for row in grid))
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=total)
        ax.set_xticks(range(len(actions)), actions, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(name, fontsize=9)
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                ax.text(j, i, value, ha="center", va="center", fontsize=8,
                        color="white" if value > total * 0.6 else "black")
    fig.suptitle("Confusion: true class vs action taken (seed 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion.png", dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    report = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    reliability(report)
    gate_sweep(report)
    confusion(report)
    cost_curve()
    print(f"plots -> {PLOTS}")


if __name__ == "__main__":
    main()
