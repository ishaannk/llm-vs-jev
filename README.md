# LLM vs Jev: a controlled comparison on LLM guardrailing

One decision spec. One policy. Several perception backends. The question is whether TypeSafe's System One model, Jev, can screen every message going into an LLM app as well as a frontier LLM can, and for how much less.

## What the numbers say

Scored through one shared spec (`77f2a821072b1862`) and one shared threshold policy, on items drawn from four public datasets:

| backend | model | n | strict accuracy | ECE | p99 latency | $/1000 screens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `jev` | `jev-1.13.0` | 240x3 | 77.9% | 0.053 | 679 ms | $0.0444 |
| `openai` | `gpt-5.1` | 240x3 | 68.8% | 0.229 | 2824 ms | $1.2636 |
| `openai-mini` | `gpt-5-mini` | 240x3 | 69.3% | 0.280 | 3912 ms | $0.2994 |
| `anthropic-opus` | `claude-opus-5` | 240x1 | 83.8% | 0.051 | 5731 ms | $5.2453 |
| `openai-astra` | `gpt-6-astra` | 120x1 | 85.8% | 0.107 | 7703 ms | $8.7300 |

Cost is the *warm-cache* figure, the best case for the LLM. Accuracy is strict: an item is an error if the action is wrong. Full table, per-class breakdown, gate sweep, reliability diagram and every single disagreement are in [RESULTS.md](RESULTS.md).

**Nothing wins outright.** On the three axes that decide a guardrail — accuracy, cost per 1000 screens and p99 latency — `jev`, `anthropic-opus`, `openai-astra` are not beaten on all three by anything else here. Every other backend is strictly dominated.

**Jev is the cheap end of the frontier.** It is beaten on accuracy by `anthropic-opus`, `openai-astra` and beats `openai`, `openai-mini`, while costing up to 196x less per 1000 screens and running up to 11x faster at p99 than the models above it. On the 120 items every backend saw, Jev is 81.1% against 83.3% for `anthropic-opus`, 85.8% for `openai-astra`.

Its calibration is the quieter result: ECE 0.053 against 0.229 for `openai`, 0.280 for `openai-mini`, 0.051 for `anthropic-opus`, 0.107 for `openai-astra`. It matches a frontier model's calibration at a fraction of the price, which is what RLCD training is supposed to buy.

**And most of them can be argued out of guarding.** Appending one sentence that asserts the classification it wants moved the final action on `anthropic-opus` 14.3%, `jev` 10.7%, `openai-mini` 7.1%, `openai` 0.0%. `anthropic-opus` is the most steerable at 14.3%; `openai` held against all four suffixes. That is the finding to act on before any of the rest matters; see [RESULTS.md](RESULTS.md#the-result-nobody-wants-to-publish).

TypeSafe's own claim is *similar* intelligence on System One tasks with large cost and latency gains. These numbers do not contradict that, and this repo does not make a stronger claim on their behalf.

## Reproducing

```bash
uv sync
cp .env.example .env   # TYPESAFE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
uv run data/build_corpus.py --per-class 200
uv run eval/harness.py --seeds 3 --items 240 --backends jev,openai,openai-mini
uv run eval/harness.py --seeds 1 --items 240 --backends anthropic-opus
uv run eval/harness.py --seeds 1 --items 120 --backends openai-astra
uv run eval/adversarial.py --backends jev,openai,openai-mini,anthropic-opus
# not probed: openai-astra - see Deviations
uv run eval/cost_curve.py --items 20
uv run eval/metrics.py && uv run eval/plots.py && uv run eval/report.py
```

`tests/test_spec_parity.py` asserts every backend imported the same spec object and that no backend file defines its own question text or thresholds. The harness aborts on a spec-hash mismatch.

## How it is built

| file | what it owns |
| --- | --- |
| `src/spec.py` | the decision spec: one Choice, one Score, three Nouls. Frozen, hashed. |
| `src/policy.py` | probabilities to PASS / REVIEW / BLOCK. Identical thresholds for everyone. |
| `src/backends/` | perception only. No question text, no thresholds; the parity test enforces it. |
| `eval/harness.py` | interleaved multi-seed runner, one worker thread per backend. |
| `eval/metrics.py` | strict accuracy, ECE, latency percentiles, cost, escalation. |
| `eval/adversarial.py` | can the message argue its way past the guardrail? |

The LLM backends go through TypeSafe's own [system-one-adapter](https://github.com/typesafe-ai/system-one-adapter-python), which renders the same spec object into a prompt and enforces the answer with native structured outputs. Using the vendor's recommended adapter is deliberate: it removes the argument that the LLM was handicapped.

