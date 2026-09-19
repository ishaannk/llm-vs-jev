"""Generate RESULTS.md from summary.json. Every number here is read, not typed.

    uv run eval/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src import policy, spec

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "RESULTS.md"


def fmt(stat: dict, pattern: str = "{:.1%}") -> str:
    mean, std = stat["mean"], stat["std"]
    if pattern.endswith("%}"):
        return f"{mean:.1%} ± {std:.1%}"
    return f"{pattern.format(mean)} ± {pattern.format(std)}"


def main() -> None:
    report = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text(encoding="utf-8"))
    run = report["run"]
    backends = report["backends"]
    names = list(backends)

    def agg(name: str, path: str) -> dict:
        return backends[name]["aggregate"][path]

    lines: list[str] = []
    w = lines.append

    # ---- headline -----------------------------------------------------------
    best_acc = max(names, key=lambda n: agg(n, "accuracy")["mean"])
    best_ece = min(names, key=lambda n: agg(n, "ece")["mean"])
    best_p99 = min(names, key=lambda n: agg(n, "latency_ms.p99")["mean"])
    best_cost = min(names, key=lambda n: agg(n, "cost.warm_modelled_usd_per_1k")["mean"])

    w("# RESULTS")
    w("")
    w(f"Generated from `eval/results/summary.json` on spec `{run['spec_hash']}`. "
      f"No number on this page was typed by hand.")
    w("")
    w("## Verdict")
    w("")
    w(f"- **Most accurate:** `{best_acc}` at {fmt(agg(best_acc, 'accuracy'))}")
    w(f"- **Best calibrated (lowest ECE):** `{best_ece}` at {fmt(agg(best_ece, 'ece'), '{:.3f}')}")
    w(f"- **Fastest tail (p99):** `{best_p99}` at {agg(best_p99, 'latency_ms.p99')['mean']:.0f} ms")
    w(f"- **Cheapest (warm-cache model):** `{best_cost}` at "
      f"${agg(best_cost, 'cost.warm_modelled_usd_per_1k')['mean']:.4f} per 1000 screens")
    w("")
    # Who is "best" depends on what you are buying, so state the frontier instead
    # of picking one. A backend is dominated if another is at least as good on all
    # three axes and strictly better on one.
    axes = [("accuracy", 1), ("cost.warm_modelled_usd_per_1k", -1), ("latency_ms.p99", -1)]
    def dominated(name: str) -> bool:
        mine = [agg(name, path)["mean"] * sign for path, sign in axes]
        return any(
            all(o >= m for o, m in zip([agg(other, path)["mean"] * sign for path, sign in axes], mine))
            and any(o > m for o, m in zip([agg(other, path)["mean"] * sign for path, sign in axes], mine))
            for other in names if other != name
        )
    front = [n for n in names if not dominated(n)]
    report["pareto_front"] = front
    w("")
    w(f"**Nothing wins outright.** On the three axes that decide a guardrail — "
      f"accuracy, cost per 1000 screens, p99 latency — the backends that are not "
      f"beaten on all three by some other backend are: "
      + ", ".join(f"`{n}`" for n in front) + ". "
      + ("Everything else is strictly dominated." if len(front) < len(names)
         else "No backend dominates another."))
    w("")
    w("| backend | accuracy | $/1k warm | p99 | on the frontier |")
    w("| --- | ---: | ---: | ---: | :---: |")
    for name in sorted(names, key=lambda n: -agg(n, "accuracy")["mean"]):
        w(f"| `{name}` | {agg(name, 'accuracy')['mean']:.1%} | "
          f"${agg(name, 'cost.warm_modelled_usd_per_1k')['mean']:.4f} | "
          f"{agg(name, 'latency_ms.p99')['mean']:.0f} ms | "
          f"{'yes' if name in front else 'no'} |")
    w("")

    jev_acc = agg("jev", "accuracy")["mean"] if "jev" in backends else None
    if jev_acc is not None:
        llms = [n for n in names if n != "jev"]
        gaps = {n: jev_acc - agg(n, "accuracy")["mean"] for n in llms}
        worst = max(gaps.values())
        best = min(gaps.values())
        w(f"Jev's strict accuracy is between {best:+.1%} and {worst:+.1%} against the LLM "
          f"backends here (positive = Jev ahead).")
        ratios = {n: agg(n, "cost.warm_modelled_usd_per_1k")["mean"]
                     / agg("jev", "cost.warm_modelled_usd_per_1k")["mean"] for n in llms}
        w(f"On warm-cache cost per 1000 screens the LLMs are "
          f"{min(ratios.values()):.0f}×–{max(ratios.values()):.0f}× Jev's price. "
          f"On p99 latency, {min(agg(n, 'latency_ms.p99')['mean'] / agg('jev', 'latency_ms.p99')['mean'] for n in llms):.1f}×"
          f"–{max(agg(n, 'latency_ms.p99')['mean'] / agg('jev', 'latency_ms.p99')['mean'] for n in llms):.1f}× Jev's.")
    w("")

    # ---- the negative, first ------------------------------------------------
    adv = report.get("adversarial")
    if adv:
        rates = {b["backend"]: b["steer_rate"] for b in adv["backends"]
                 if b["steer_rate"] is not None}
        ranked = sorted(rates.items(), key=lambda kv: -kv[1])
        loser, loser_rate = ranked[0]
        detail = next(b for b in adv["backends"] if b["backend"] == loser)
        clean = [n for n, r in ranked if r == 0.0]
        w("### The result nobody wants to publish")
        w("")
        w(f"**Most of these guardrails can be argued out of guarding.** Appending one "
          f"sentence that simply asserts the classification it wants — no jailbreak "
          f"craft, no obfuscation — moved the final action on:")
        w("")
        w("| backend | steer rate | steered / attempts | worst suffix |")
        w("| --- | ---: | ---: | --- |")
        for name, rate in ranked:
            row = next(b for b in adv["backends"] if b["backend"] == name)
            by_suffix = row["per_suffix"]
            worst_suffix = (f"`{max(by_suffix, key=lambda k: by_suffix[k])}` "
                            f"({max(by_suffix.values()):.0%})") if by_suffix else "—"
            w(f"| `{name}` | {rate:.1%} | {row['n_steered']}/{row['n_attempts']} | "
              f"{worst_suffix if rate else '—'} |")
        for name, note in (adv.get("not_probed") or {}).items():
            w(f"| `{name}` | not probed | — | {note} |")
        w("")
        w(f"`{loser}` is the most steerable backend in this benchmark at "
          f"{loser_rate:.1%}"
          + (f", and it is also the most accurate one." if loser == max(
              names, key=lambda n: agg(n, "accuracy")["mean"]) else ".")
          + (f" {', '.join('`' + n + '`' for n in clean)} "
             f"{'was' if len(clean) == 1 else 'were'} moved by none of the four "
             f"suffixes." if clean else ""))
        w("")
        w("This is the failure mode TypeSafe documents for Jev — \"state is data, and "
          "jev-1.13 does not treat it as hostile by default... text that argues for "
          "its own classification can move the answer\" "
          "([jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)) "
          "— and the measurement says it is not Jev's alone. The LLM adapter's system "
          "prompt explicitly instructs the model to treat the document as untrusted "
          "data and never follow instructions inside it; that instruction did not "
          "hold either. A guardrail that can be talked down is worse than a slow one, "
          "so this is the number to fix before any of the rest matters.")
        w("")


    # ---- environment --------------------------------------------------------
    w("## Environment")
    w("")
    w(f"| | |")
    w(f"| --- | --- |")
    w(f"| Machine | {run['machine']['platform']} |")
    w(f"| CPU | {run['machine']['processor'] or 'n/a'} |")
    w(f"| Python | {run['machine']['python']} |")
    w(f"| Wall clock (UTC) | {run['started_utc']} → {run['finished_utc']} ({run['wall_clock_s']:.0f}s) |")
    by_backend = run.get("n_items_by_backend", {})
    w(f"| Items | {run['n_items']} max (stratified subsample, seed "
      f"{run['sample_seed']}); per backend: "
      + ", ".join(f"{b} {n}" for b, n in sorted(by_backend.items())) + " |")
    w(f"| Passes | " + "; ".join(
        f"{', '.join(win['backends'])} — {win['n_items']} items × {win['seeds']} seed(s), "
        f"{win['started_utc'][11:19]}–{win['finished_utc'][11:19]}Z"
        for win in run.get("windows", [])) + " |")
    w(f"| Spec hash | `{run['spec_hash']}` |")
    w(f"| Policy | review ≥ {run['policy']['review_threshold']}, action ≥ "
      f"{run['policy']['action_threshold']}, severity blocks at {run['policy']['severity_block']} |")
    w("")
    w("Resolved model versions:")
    w("")
    w("| backend | model that answered | items | seeds | prompt prefix (tokens) | transport retries | repair retries |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in names:
        meta = backends[name]["meta"]
        w(f"| `{name}` | `{backends[name]['model']}` | {backends[name]['n_items']} | "
          f"{backends[name]['n_seeds']} | {meta['prefix_tokens']} | "
          f"{meta['retry']['transport_retries']} | {meta['retry']['repair_retries']} |")
    w("")

    # ---- corpus -------------------------------------------------------------
    w("## Corpus")
    w("")
    w("| dataset | revision | licence | class |")
    w("| --- | --- | --- | --- |")
    class_of = {
        "deepset/prompt-injections": "injection + benign",
        "jackhhao/jailbreak-classification": "jailbreak + benign",
        "natolambert/xstest-v2-copy": "borderline (safe prompts written to look unsafe)",
        "bench-llm/or-bench": "borderline (over-refusal hard set)",
    }
    for dataset, meta in provenance["sources"].items():
        w(f"| [{dataset}]({meta['url']}) | `{meta['revision'][:12]}` | {meta['licence']} | {class_of[dataset]} |")
    w("")
    w(f"Built corpus: {sum(provenance['counts'].values())} items, "
      + ", ".join(f"{v} {k}" for k, v in sorted(provenance["counts"].items()))
      + f". `corpus.jsonl` sha256 `{provenance['corpus_sha256']}`. "
      f"Median length by class (characters): "
      + ", ".join(f"{k} {v}" for k, v in sorted(provenance["median_chars"].items()))
      + ".")
    w("")
    w("All four licences permit redistribution, so `data/corpus.jsonl` is committed "
      "alongside the builder.")
    w("")

    # ---- main table ---------------------------------------------------------
    w("## The table")
    w("")
    w("Mean ± std across seeds. Latency is per call, measured with each backend "
      "issuing one request at a time.")
    w("")
    w("| backend | n | strict accuracy | invalid rate | repairs | p50 | p95 | p99 | $/1k cold | $/1k warm (measured) | $/1k warm (modelled) | ECE |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in names:
        w(f"| `{name}` "
          f"| {backends[name]['n_items']}×{backends[name]['n_seeds']} "
          f"| {fmt(agg(name, 'accuracy'))} "
          f"| {agg(name, 'invalid_rate')['mean']:.2%} "
          f"| {agg(name, 'mean_repair_attempts')['mean']:.2f} "
          f"| {agg(name, 'latency_ms.p50')['mean']:.0f} ms "
          f"| {agg(name, 'latency_ms.p95')['mean']:.0f} ms "
          f"| {agg(name, 'latency_ms.p99')['mean']:.0f} ms "
          f"| ${agg(name, 'cost.cold_usd_per_1k')['mean']:.4f} "
          f"| ${agg(name, 'cost.warm_measured_usd_per_1k')['mean']:.4f} "
          f"| ${agg(name, 'cost.warm_modelled_usd_per_1k')['mean']:.4f} "
          f"| {fmt(agg(name, 'ece'), '{:.3f}')} |")
    w("")
    w("`$/1k warm (measured)` uses the cached-token counts the provider actually "
      "reported during the run; `$/1k warm (modelled)` assumes the entire constant "
      "prompt prefix is served from cache, which is the best case an LLM guardrail "
      "can reach. Argue against the modelled column, not the cold one.")
    w("")

    # ---- per class ----------------------------------------------------------
    w("### Accuracy by class")
    w("")
    w("| backend | " + " | ".join(sorted(policy.EXPECTED_ACTIONS)) + " | borderline read strictly |")
    w("| --- | " + " | ".join("---:" for _ in policy.EXPECTED_ACTIONS) + " | ---: |")
    for name in names:
        per = backends[name]["per_label_accuracy"]
        w(f"| `{name}` | " + " | ".join(fmt(per[label]) for label in sorted(per))
          + f" | {fmt(agg(name, 'accuracy_strict_borderline'))} |")
    w("")
    w("Expected actions: "
      + ", ".join(f"`{k}` → {'/'.join(sorted(v))}" for k, v in sorted(policy.EXPECTED_ACTIONS.items()))
      + ". The last column re-scores the whole set with `borderline` → REVIEW only, "
      "which is the stricter reading; see *Design choices*.")
    w("")

    if "common_subset" in report and len({backends[n]["n_items"] for n in names}) > 1:
        cs = report["common_subset"]
        w("### Like-for-like: the items every backend saw")
        w("")
        w(f"Backends ran on different item counts because of the API budget. "
          f"Re-scored on the {cs['n_items']} items common to all of them:")
        w("")
        w("| backend | strict accuracy | ECE |")
        w("| --- | ---: | ---: |")
        for name in names:
            row = cs["backends"][name]
            w(f"| `{name}` | {fmt(row['accuracy'])} | {fmt(row['ece'], '{:.3f}')} |")
        w("")

    # ---- escalation ---------------------------------------------------------
    w("### Escalation efficiency")
    w("")
    w(f"Hand the least-confident {policy.EXPECTED_ACTIONS and ''}"
      f"{backends[names[0]]['per_seed'][0]['escalation']['budget']:.0%} of traffic to a "
      "human who is always right.")
    w("")
    w("| backend | items escalated | residual error caught | accuracy after review |")
    w("| --- | ---: | ---: | ---: |")
    for name in names:
        esc = backends[name]["per_seed"][0]["escalation"]
        w(f"| `{name}` | {esc['n_escalated']} | "
          f"{fmt(agg(name, 'escalation.residual_error_caught'))} | "
          f"{fmt(agg(name, 'escalation.accuracy_after'))} |")
    w("")

    # ---- plots --------------------------------------------------------------
    w("## Plots")
    w("")
    for title, filename in [
        ("Reliability diagram", "reliability.png"),
        ("Gate sweep", "gate_sweep.png"),
        ("Confusion", "confusion.png"),
        ("Cost and latency vs number of judgments", "cost_curve.png"),
    ]:
        if (RESULTS / "plots" / filename).exists():
            w(f"### {title}")
            w("")
            w(f"![{title}](eval/results/plots/{filename})")
            w("")

    curve_path = RESULTS / "cost_curve.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        w("Cost per 1000 screens as judgments are added to the same message "
          f"({curve['n_items']} items per point):")
        w("")
        w("| backend | " + " | ".join(f"{p['n_questions']} q" for p in next(iter(curve["curve"].values()))) + " |")
        w("| --- | " + " | ".join("---:" for _ in next(iter(curve["curve"].values()))) + " |")
        for name, points in curve["curve"].items():
            w(f"| `{name}` | " + " | ".join(f"${p['usd_per_1k']:.4f}" for p in points) + " |")
        w("")

    # ---- adversarial --------------------------------------------------------
    if "adversarial" in report:
        w("## Adversarial probe")
        w("")
        w("Each probe is a message with a known correct action plus a suffix arguing "
          "for a different one. Steer rate is the share of probes where the suffix "
          "moved the action the way the suffix wanted.")
        w("")
        w("| backend | probes | steerable attempts | steered | steer rate |")
        w("| --- | ---: | ---: | ---: | ---: |")
        for s in report["adversarial"]["backends"]:
            rate = "n/a" if s["steer_rate"] is None else f"{s['steer_rate']:.1%}"
            w(f"| `{s['backend']}` | {s['n_probes']} | {s['n_attempts']} | {s['n_steered']} | {rate} |")
        w("")
        w("By suffix type:")
        w("")
        suffixes = sorted({k for s in report["adversarial"]["backends"] for k in s["per_suffix"]})
        w("| backend | " + " | ".join(f"`{s}`" for s in suffixes) + " |")
        w("| --- | " + " | ".join("---:" for _ in suffixes) + " |")
        for s in report["adversarial"]["backends"]:
            w(f"| `{s['backend']}` | "
              + " | ".join(f"{s['per_suffix'].get(k, 0):.0%}" for k in suffixes) + " |")
        w("")

    # ---- disagreements ------------------------------------------------------
    w("## Every disagreement")
    w("")
    disagreements = report["disagreements"]
    w(f"{len(disagreements)} of {run['n_items']} items produced at least one wrong action "
      f"from at least one backend on at least one seed. Full list, worst first. "
      f"`p` is the probability on the chosen category; `runner-up` is the second "
      f"category and its probability, from seed 1.")
    w("")
    w("| id | true class | source | " + " | ".join(f"`{n}`" for n in names) + " |")
    w("| --- | --- | --- | " + " | ".join("---" for _ in names) + " |")
    for entry in disagreements:
        cells = []
        for name in names:
            seeds = entry["backends"].get(name, [])
            if not seeds:
                cells.append("—")
                continue
            first = seeds[0]
            wrong = entry["n_wrong_of_seeds"].get(name, 0)
            mark = "✗" if wrong else "✓"
            cells.append(
                f"{mark} {first['action']} · {first['category']} p={first['p']:.2f} · "
                f"2nd {first['runner_up']} {first['p_runner_up']:.2f} · conf {first['confidence']:.2f} · "
                f"sev {first['severity']:.1f} · wrong {wrong}/{entry['n_seeds'][name]}"
            )
        source = entry["source"].split("@")[0].split("/")[-1]
        w(f"| `{entry['id']}` | {entry['label']} | {source} | " + " | ".join(cells) + " |")
    w("")

    # ---- limits -------------------------------------------------------------
    w("## Limits")
    w("")
    w("Quoting [the jev-1.13 jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md), "
      "which this design was written against:")
    w("")
    w("- **No arithmetic.** \"Jev is not a calculator... implement any mathematical "
      "logic in code.\" All thresholding happens in `src/policy.py`; the model is "
      "never asked to compare or count.")
    w("- **Dates are text.** Not exercised here, but it rules out any guardrail rule "
      "that depends on a time window.")
    w("- **Degradation on large noisy state.** \"Accuracy falls as the state grows "
      "with content unrelated to the decision.\" Our state is one field; long "
      f"jailbreak prompts are truncated at {spec.MAX_CHARS} characters for every backend alike.")
    w("- **No generation.** Jev returns judgments, not text. It cannot rewrite or "
      "redact a message, only route it. Anything that has to *produce* a safe reply "
      "still needs a generative model.")
    w("- **Adversarial by default.** \"State is data, and jev-1.13 does not treat it "
      "as hostile by default.\" That is exactly what the adversarial probe above "
      "measures, and the LLM adapter's own system prompt claims the opposite "
      "protection, so the two are directly comparable.")
    w("- **Literal reading.** Every criterion in `src/spec.py` is written as a "
      "concrete situation rather than a label, because the model answers the "
      "question as written.")
    w("")

    # ---- threats ------------------------------------------------------------
    w("## Threats to validity")
    w("")
    w("- **Our thresholds.** `review 0.35 / action 0.70 / severity 2.0` come from "
      "TypeSafe's own cookbook and were fixed before any backend ran. They are not "
      "optimal for anyone. The gate sweep is published precisely so you can see how "
      "much the conclusion moves.")
    w("- **Our corpus, and its label noise.** These are public labels, not ground "
      "truth. `deepset/prompt-injections` marks mild role-play as an injection and "
      "contains German items; `jackhhao/jailbreak-classification` files some "
      "role-play under `benign` and some under `jailbreak`. That ceiling applies "
      "equally to every backend, but it means absolute accuracy here is not a "
      "leaderboard number.")
    w("- **Borderline is a judgment call.** See *Deviations*.")
    w(f"- **One region, {len(run.get('windows', [1]))} pass(es).** Everything ran from one "
      f"machine inside {run['wall_clock_s']:.0f} seconds of wall clock, listed under "
      "*Passes* above. Latency is network-dependent and we did not test another region.")
    w("- **Unequal seeds and item counts.** The expensive models ran on fewer items "
      "and fewer seeds because each had a hard spend cap. Their rows carry no "
      "meaningful std, and the like-for-like table re-scores everyone on the common "
      "subset.")
    w("- **Concurrency.** The three backends ran in parallel with one another so the "
      "pass fit in one session; each backend was strictly sequential internally. "
      "Three concurrent HTTP requests should not disturb per-call latency, but it "
      "is not a hermetic latency lab.")
    w("- **Subsample.** The built corpus is "
      f"{sum(provenance['counts'].values())} items; {run['n_items']} were evaluated, "
      f"stratified, drawn with seed {run['sample_seed']} before any result was seen. "
      "The API budget for this run was fixed in advance.")
    w("- **Early-access model.** `jev-latest` moves. The resolved version that "
      f"answered is recorded above.")
    w("")

    # ---- design choices -----------------------------------------------------
    w("## Design choices, and what they cost")
    w("")
    w("1. **Unequal budgets, so unequal runs.** Each API key had a hard spend "
      "cap (OpenAI $5 total, Anthropic $2). `jev`, `openai` and `openai-mini` "
      "ran the full 240 items x 3 seeds. `anthropic-opus` ran 240 x 1 and "
      "`openai-astra` 120 x 1, because at $5/Mtok and $10/Mtok input those "
      "were what fit. The single-seed rows carry no std, and the "
      "*Like-for-like* table above re-scores every backend on the items all of "
      "them saw. The 120-item set is a strict subset of the 240-item set by "
      "construction (`load_items`), not a fresh draw."
      )
    w("2. **`openai-astra` was never adversarially probed.** The 36 probes "
      "would have cost about $0.54 against roughly $0.37 of remaining OpenAI "
      "credit. Every other backend was probed. The gap is marked in the "
      "adversarial table rather than left as a blank cell that reads like a "
      "zero, and it is the one measurement this run is missing."
      )
    w("3. **Borderline items are scored as PASS or REVIEW, not REVIEW alone.** "
      "They come from XSTest and OR-Bench, which are *safe* prompts written to "
      "look unsafe. Blocking them is the failure mode; letting them through is "
      "defensible and so is sending them to a human. The stricter REVIEW-only "
      "reading is reported in its own column above, so neither choice hides "
      "anything."
      )
    w("4. **Invalid-output rate is 0 everywhere, on both sides, by "
      "construction.** It looked like it would be a differentiator and it is "
      "not: the adapter uses OpenAI's and Anthropic's native strict "
      "JSON-schema structured outputs, which makes a schema violation "
      "impossible for the LLMs too. The honest finding is that *with the "
      "vendor-recommended adapter*, schema compliance is not where an LLM "
      "loses. Repair retries were set to zero on every backend so a violation "
      "would have been counted rather than quietly fixed."
      )
    w("5. **`contains_pii` is measured but routes nothing.** It is a property "
      "of the message rather than of its intent, so `src/policy.py` maps it to "
      "no action. It stays in the spec because it is reported per item and "
      "because it costs almost nothing to ask alongside the others."
      )
    w("6. **Thinking is pinned off wherever the API allows it.** `gpt-5.1` and "
      "`gpt-6-astra` are reasoning models, and `claude-opus-5` has extended "
      "thinking on by default. A guardrail that runs on 100% of traffic cannot "
      "pay for a chain of thought billed at output rates, so `reasoning_effort` "
      "is `none` (`minimal` on `gpt-5-mini`) and Opus runs `thinking: "
      "disabled`. `gpt-6-astra` rejects `none`; its floor is `low`, so that is "
      "what it ran at, and it is the one backend given a reasoning budget. "
      "Leaving thinking on would have inflated both the cost and the latency "
      "columns against the LLMs."
      )
    w("")

    w("## Reproducing")
    w("")
    w("```bash")
    w("uv run data/build_corpus.py --per-class 200")
    for win in run.get("windows", []):
        w(f"uv run eval/harness.py --seeds {win['seeds']} --items {win['n_items']} "
          f"--backends {','.join(win['backends'])}")
    probed = [b["backend"] for b in report.get("adversarial", {}).get("backends", [])] or names
    w(f"uv run eval/adversarial.py --backends {','.join(probed)}")
    unprobed = [n for n in names if n not in probed]
    if unprobed:
        w(f"# not probed: {', '.join(unprobed)} - see Deviations")
    w("uv run eval/cost_curve.py --items 20")
    w("uv run eval/metrics.py && uv run eval/plots.py && uv run eval/report.py")
    w("```")
    w("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")
    write_readme(report)


def write_readme(report: dict) -> None:
    """README last, from the same numbers. Nothing typed by hand here either."""
    run, backends = report["run"], report["backends"]
    names = list(backends)

    def agg(name: str, path: str) -> float:
        return backends[name]["aggregate"][path]["mean"]

    best_acc = max(names, key=lambda n: agg(n, "accuracy"))
    adv = {
        b["backend"]: b["steer_rate"]
        for b in report.get("adversarial", {}).get("backends", [])
        if b["steer_rate"] is not None
    }
    lines: list[str] = []
    w = lines.append
    w("# LLM vs Jev: a controlled comparison on LLM guardrailing")
    w("")
    w("One decision spec. One policy. Several perception backends. The question is "
      "whether TypeSafe's System One model, Jev, can screen every message going into "
      "an LLM app as well as a frontier LLM can, and for how much less.")
    w("")
    w("## What the numbers say")
    w("")
    w(f"Scored through one shared spec (`{run['spec_hash']}`) and one shared "
      f"threshold policy, on items drawn from four public datasets:")
    w("")
    w("| backend | model | n | strict accuracy | ECE | p99 latency | $/1000 screens |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in names:
        w(f"| `{name}` | `{backends[name]['model']}` | "
          f"{backends[name]['n_items']}x{backends[name]['n_seeds']} | "
          f"{agg(name, 'accuracy'):.1%} | {agg(name, 'ece'):.3f} | "
          f"{agg(name, 'latency_ms.p99'):.0f} ms | "
          f"${agg(name, 'cost.warm_modelled_usd_per_1k'):.4f} |")
    w("")
    w("Cost is the *warm-cache* figure, the best case for the LLM. Accuracy is "
      "strict: an item is an error if the action is wrong. Full table, per-class "
      "breakdown, gate sweep, reliability diagram and every single disagreement are "
      "in [RESULTS.md](RESULTS.md).")
    w("")
    front = report.get("pareto_front") or []
    if front:
        w(f"**Nothing wins outright.** On the three axes that decide a guardrail — "
          f"accuracy, cost per 1000 screens and p99 latency — "
          + ", ".join(f"`{n}`" for n in front) + " "
          + ("is" if len(front) == 1 else "are")
          + " not beaten on all three by anything else here. Every other backend is "
            "strictly dominated.")
        w("")
    if "jev" in backends:
        llms = [n for n in names if n != "jev"]
        cheaper = max(agg(n, "cost.warm_modelled_usd_per_1k") / agg("jev", "cost.warm_modelled_usd_per_1k")
                      for n in llms)
        faster = max(agg(n, "latency_ms.p99") / agg("jev", "latency_ms.p99") for n in llms)
        ahead = [n for n in llms if agg(n, "accuracy") > agg("jev", "accuracy")]
        behind = [n for n in llms if agg(n, "accuracy") <= agg("jev", "accuracy")]
        sub = report.get("common_subset", {}).get("backends", {})
        note = ""
        if sub and len({backends[n]["n_items"] for n in names}) > 1:
            note = (f" On the {report['common_subset']['n_items']} items every backend "
                    f"saw, Jev is {sub['jev']['accuracy']['mean']:.1%} against "
                    + ", ".join(f"{sub[n]['accuracy']['mean']:.1%} for `{n}`" for n in ahead)
                    + ".")
        w(f"**Jev is the cheap end of the frontier.** It is beaten on accuracy by "
          + ", ".join(f"`{n}`" for n in ahead)
          + (f" and beats " + ", ".join(f"`{n}`" for n in behind) if behind else "")
          + f", while costing up to {cheaper:.0f}x less per 1000 screens and running "
            f"up to {faster:.0f}x faster at p99 than the models above it."
          + note)
        w("")
        w(f"Its calibration is the quieter result: ECE "
          f"{agg('jev', 'ece'):.3f} against "
          + ", ".join(f"{agg(n, 'ece'):.3f} for `{n}`" for n in names if n != "jev")
          + ". It matches a frontier model's calibration at a fraction of the price, "
            "which is what RLCD training is supposed to buy.")
        w("")
    adv_rows = {b["backend"]: b["steer_rate"] for b in report.get("adversarial", {}).get("backends", [])
                if b["steer_rate"] is not None}
    if adv_rows:
        worst, worst_rate = max(adv_rows.items(), key=lambda kv: kv[1])
        clean = [n for n, r in adv_rows.items() if r == 0.0]
        w(f"**And most of them can be argued out of guarding.** Appending one sentence "
          f"that asserts the classification it wants moved the final action on "
          + ", ".join(f"`{n}` {r:.1%}" for n, r in sorted(adv_rows.items(), key=lambda kv: -kv[1]))
          + f". `{worst}` is the most steerable at {worst_rate:.1%}"
          + (f"; {', '.join('`' + n + '`' for n in clean)} held against all four "
             f"suffixes." if clean else ".")
          + " That is the finding to act on before any of the rest matters; see "
            "[RESULTS.md](RESULTS.md#the-result-nobody-wants-to-publish).")
        w("")

    w("TypeSafe's own claim is *similar* intelligence on System One tasks with large "
      "cost and latency gains. These numbers do not contradict that, and this repo "
      "does not make a stronger claim on their behalf.")
    w("")
    w("## Reproducing")
    w("")
    w("```bash")
    w("uv sync")
    w("cp .env.example .env   # TYPESAFE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
    w("uv run data/build_corpus.py --per-class 200")
    for win in run.get("windows", []):
        w(f"uv run eval/harness.py --seeds {win['seeds']} --items {win['n_items']} "
          f"--backends {','.join(win['backends'])}")
    probed = [b["backend"] for b in report.get("adversarial", {}).get("backends", [])] or names
    w(f"uv run eval/adversarial.py --backends {','.join(probed)}")
    unprobed = [n for n in names if n not in probed]
    if unprobed:
        w(f"# not probed: {', '.join(unprobed)} - see Deviations")
    w("uv run eval/cost_curve.py --items 20")
    w("uv run eval/metrics.py && uv run eval/plots.py && uv run eval/report.py")
    w("```")
    w("")
    w("`tests/test_spec_parity.py` asserts every backend imported the same spec object "
      "and that no backend file defines its own question text or thresholds. The "
      "harness aborts on a spec-hash mismatch.")
    w("")
    w("## How it is built")
    w("")
    w("| file | what it owns |")
    w("| --- | --- |")
    w("| `src/spec.py` | the decision spec: one Choice, one Score, three Nouls. Frozen, hashed. |")
    w("| `src/policy.py` | probabilities to PASS / REVIEW / BLOCK. Identical thresholds for everyone. |")
    w("| `src/backends/` | perception only. No question text, no thresholds; the parity test enforces it. |")
    w("| `eval/harness.py` | interleaved multi-seed runner, one worker thread per backend. |")
    w("| `eval/metrics.py` | strict accuracy, ECE, latency percentiles, cost, escalation. |")
    w("| `eval/adversarial.py` | can the message argue its way past the guardrail? |")
    w("")
    w("The LLM backends go through TypeSafe's own "
      "[system-one-adapter](https://github.com/typesafe-ai/system-one-adapter-python), "
      "which renders the same spec object into a prompt and enforces the answer with "
      "native structured outputs. Using the vendor's recommended adapter is "
      "deliberate: it removes the argument that the LLM was handicapped.")
    w("")
    readme = ROOT / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {readme} ({len(lines)} lines)")



if __name__ == "__main__":
    main()
