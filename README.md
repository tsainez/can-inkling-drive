# can-inkling-drive

An evaluation harness for measuring what accuracy *costs* on autonomous-driving
reasoning tasks — accuracy per thinking token and accuracy per dollar, not
accuracy alone.

**Scope limit.** This evaluates the slow reasoning layer only: scene
understanding, edge-case interpretation, decision explanation. It says nothing
about closed-loop low-level control, and no claim here should be read as
support for putting a language model in a control loop.

## Status

- **Step 1 — done.** Harness built, 111 tests passing offline with no API key.
- **Step 2 — done.** Provider selected: Baseten. `reasoning_tokens` confirmed
  available, so RQ2 does not need a proxy metric. See
  [docs/provider-selection.md](docs/provider-selection.md).
- **Step 3 — ready to run.** DriveLM v1.1 **train** split (val ships without
  gold answers). See [docs/runbook.md](docs/runbook.md).
- **Steps 4–5** — pilot and scale-up, not started.

```bash
pip install -e ".[dev]"
pytest                      # 111 tests, no network, no key
```

## Research questions

| | Question | Status |
|---|---|---|
| RQ1 | Structured driving QA vs. comparable open-weights models and a frontier closed reference | table stakes |
| RQ2 | Accuracy per thinking token and per dollar; does token efficiency transfer from coding to visual driving reasoning | the hook |
| RQ3 | How much driving-QA accuracy is visually grounded vs. recoverable from language priors | load-bearing |
| RQ4 | Does the accuracy/efficiency ranking survive sensor degradation | load-bearing |

## Design decisions

**DriveLM is the primary benchmark.** Roughly a quarter of DriveLM questions are
already multiple choice with a single-letter gold answer, which gives objective
scoring with no LLM judge and no rubric. Distractors are human-annotated, so
they do not carry the linguistic fingerprints that VLM-generated distractors
leave behind. AutoDrive-QA is cited for its distractor taxonomy and as a
reference point only; nothing here is built on it.

**Three conditions per model, plus a fourth for grounding.** `clean`,
`blind_tags`, `blind_notags`, `corrupt`.

The two blind conditions exist because DriveLM question text embeds object
references like `<c1,CAM_FRONT,1088.3,497.5>`. Those name a camera and a pixel
location, so a text-only run that leaves them in is not actually blind. Running
both decomposes the grounding gap three ways:

```
total gap          = clean − blind_notags
image contribution = clean − blind_tags
tag leakage        = blind_tags − blind_notags
```

A small total gap has two possible causes — the model is not grounded, or the
benchmark is textually leaky. This split tells them apart instead of leaving the
result ambiguous.

**Tokens are the durable axis; dollars are a dated snapshot.** Accuracy per
token is a property of the model. Accuracy per dollar is a property of whoever
is hosting it this month, and open-weights pricing moves with promotions. Prices
are recorded per-record with a quote date and reported in an appendix, never as
the headline.

**Thinking effort is pinned, not swept.** Inkling exposes a thinking-effort
control. All results are collected at one documented setting; sweeping it is
future work. This is stated as a limitation rather than left implicit.

## Harness properties

- **Collection and scoring never share a process.** A scoring bug costs zero
  dollars to fix — delete the scored output, fix the code, rerun.
- **Content-addressed JSONL cache.** The key covers model, condition, question,
  prompt hash, seed, decode params, thinking effort, and image manifest. Change
  any of them and the cache forks rather than silently serving stale records.
- **Resumable.** Append-only with fsync per record; a killed process loses at
  most the in-flight call, and a truncated trailing line is skipped on reload.
- **Errors are classified.** Retryable (429, 5xx, timeout) are not cached and
  retry. Terminal (malformed request, refusal, context overflow) are cached as
  terminal with a reason, because retrying them forever burns budget on calls
  that can never succeed.
- **Full raw responses are stored.** Metrics nobody anticipated stay recoverable
  without spending money again. This is what rescues RQ2 if a provider turns out
  not to report `usage.completion_tokens_details.reasoning_tokens`.
- **Invalid output is a reported metric.** Unparseable answers count as wrong
  *and* are reported separately, along with how every answer was extracted.
- **Originals are never modified.** Corruptions happen in memory; a test asserts
  the source file hash is unchanged after a full corruption pass.

## Usage

```bash
# Inspect a benchmark: MCQ share, category distribution, chance accuracy
python -m idq.cli inspect --adapter drivelm --data data/drivelm/train.json

# Free end-to-end verification with the mock provider
python -m idq.cli collect --provider mock --condition blind_tags --n-fixture 400
python -m idq.cli score   --cache results/cache.jsonl --out results/scored.jsonl
python -m idq.cli analyze --scored results/scored.jsonl

# Dry run against a real provider: renders prompts, makes no calls
python -m idq.cli collect --adapter drivelm --data data/drivelm/train.json \
  --provider openai_compat --base-url https://api.example.com/v1 \
  --api-key-env IDQ_API_KEY --model-string <exact-model-string> \
  --condition blind_tags --sample 20 --dry-run
```

## The smoke test

`tests/test_smoke.py` runs the whole pipeline offline and asserts, among other
things, that a uniformly random mock model scores at chance — where chance is
`mean(1/n_options)`, not a hardcoded 0.25, because the question set mixes 2-, 3-
and 4-option questions. Hardcoding 0.25 would make the sanity check itself
wrong. If scoring were misaligned, or the parser favoured one option position,
or unparseable output were scored as correct, this test fails. Every other
number the harness produces is only trustworthy because this one holds.

## Two bugs deliberately guarded against

1. **PIL's `ImageFilter.Kernel` accepts only 3×3 and 5×5.** Motion blur is
   implemented in numpy as a normalized line kernel, so severity 5 can use a
   27-pixel streak.
2. **A naive option regex drops options whose text starts with A–D.** "C. Back
   up" vanishes when `[^A-D]` cannot match the B in "Back". Option boundaries
   are detected positionally instead, and a test asserts "Back up" survives.

## IP boundary

Public models and public datasets only. No data, telemetry, logs, imagery, or
observations from any client program enter this repository. Domain knowledge
informs which scenarios are worth slicing on; domain data does not. No employer
hardware or compute is used. This statement belongs in the paper's methods
section verbatim.

Credentials are read from environment variables only. No key is written to the
cache, to a config file, or to a log line — there is a test for that.

## Licence

Code: MIT. DriveLM is redistributed under its own terms; this repository ships
no benchmark data.
