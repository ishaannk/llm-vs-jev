"""Download, normalise and pin the public corpus. Nothing here is hand-written.

Four classes, four public datasets, every revision pinned by commit hash.
Run: `uv run data/build_corpus.py --per-class 200`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "corpus.jsonl"
PROVENANCE = Path(__file__).parent / "provenance.json"

SOURCES = {
    "deepset/prompt-injections": {
        "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "licence": "apache-2.0",
        "url": "https://huggingface.co/datasets/deepset/prompt-injections",
    },
    "jackhhao/jailbreak-classification": {
        "revision": "2f2ceeb39658696fd3f462403562b6eea5306287",
        "licence": "apache-2.0",
        "url": "https://huggingface.co/datasets/jackhhao/jailbreak-classification",
    },
    "natolambert/xstest-v2-copy": {
        "revision": "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
        "licence": "cc-by-4.0",
        "url": "https://huggingface.co/datasets/natolambert/xstest-v2-copy",
    },
    "bench-llm/or-bench": {
        "revision": "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
        "licence": "cc-by-4.0",
        "url": "https://huggingface.co/datasets/bench-llm/or-bench",
    },
}

#: XSTest's `contrast_*` types are genuinely unsafe prompts; everything else in
#: the set is a *safe* prompt written to look unsafe, which is exactly the
#: borderline class.
XSTEST_SAFE_PREFIXES = ("contrast_",)

MIN_CHARS = 20


def _rows(name: str, config: str | None, split: str):
    source = SOURCES[name]
    return load_dataset(name, config, split=split, revision=source["revision"])


def collect() -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {k: [] for k in ("benign", "injection", "jailbreak", "borderline")}

    for split in ("train", "test"):
        for row in _rows("deepset/prompt-injections", None, split):
            label = "injection" if row["label"] == 1 else "benign"
            buckets[label].append(
                {"text": row["text"], "label": label, "source": f"deepset/prompt-injections@{split}"}
            )

    for split in ("train", "test"):
        for row in _rows("jackhhao/jailbreak-classification", None, split):
            label = "jailbreak" if row["type"] == "jailbreak" else "benign"
            buckets[label].append(
                {"text": row["prompt"], "label": label, "source": f"jackhhao/jailbreak-classification@{split}"}
            )

    for row in _rows("natolambert/xstest-v2-copy", "default", "prompts"):
        if row["type"].startswith(XSTEST_SAFE_PREFIXES):
            continue
        buckets["borderline"].append(
            {"text": row["prompt"], "label": "borderline", "source": f"natolambert/xstest-v2-copy@{row['type']}"}
        )

    for row in _rows("bench-llm/or-bench", "or-bench-hard-1k", "train"):
        buckets["borderline"].append(
            {"text": row["prompt"], "label": "borderline", "source": f"bench-llm/or-bench@hard-1k/{row['category']}"}
        )

    return buckets


def build(per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    corpus, seen = [], set()
    for label, rows in collect().items():
        pool = []
        for row in rows:
            text = " ".join(row["text"].split())
            digest = hashlib.sha1(text.lower().encode()).hexdigest()
            if len(text) < MIN_CHARS or digest in seen:
                continue
            seen.add(digest)
            pool.append({**row, "text": text, "id": f"{label}-{digest[:10]}"})
        rng.shuffle(pool)
        if len(pool) < per_class:
            print(f"  ! {label}: only {len(pool)} available, wanted {per_class}")
        corpus.extend(pool[:per_class])
    corpus.sort(key=lambda r: r["id"])
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    corpus = build(args.per_class, args.seed)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in corpus:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    chars: dict[str, list[int]] = {}
    for row in corpus:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
        chars.setdefault(row["label"], []).append(len(row["text"]))
    PROVENANCE.write_text(
        json.dumps(
            {
                "sources": SOURCES,
                "sample_seed": args.seed,
                "per_class": args.per_class,
                "counts": counts,
                "median_chars": {k: sorted(v)[len(v) // 2] for k, v in chars.items()},
                "corpus_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()[:16],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(corpus)} items -> {OUT}")
    for label, count in sorted(counts.items()):
        print(f"  {label:<11}{count:>5}   median {sorted(chars[label])[len(chars[label]) // 2]} chars")


if __name__ == "__main__":
    main()
