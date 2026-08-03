# Handoff — state of the project as of 2026-08-01

Read this before touching anything. Several things in this repo contradict what
you would reasonably assume, and two other docs contain claims that are now
known to be false (flagged below).

---

## What this project is

Measuring what accuracy *costs* on driving-QA reasoning — accuracy per thinking
token and per dollar — for Inkling (Thinking Machines, 975B/41B MoE, Apache 2.0,
released 2026-07-15) against comparable models.

Deliverables: a public GitHub repo (exists) and a workshop paper (methods draft
exists, no results yet).

**Hard scope limit, repeat it in any writeup:** this evaluates the slow
reasoning layer only — scene understanding, edge-case interpretation, decision
explanation. It says nothing about closed-loop low-level control. Do not let any
claim drift toward "LLM in the control loop."

### Research questions

- **RQ1** structured driving QA vs. comparable models — table stakes
- **RQ2** accuracy per thinking token and per dollar — the hook, but cost/accuracy
  frontier analysis is established in general and only underexplored for driving
  VLMs. Do not oversell it; a reviewer will call that.
- **RQ3** how much accuracy is visually grounded vs. language priors — load-bearing
- **RQ4** does the ranking survive sensor degradation — load-bearing

---

## Non-negotiable constraints

- **IP boundary.** The owner is a contractor on a commercial AV program. Public
  models and public datasets only. No data, telemetry, logs, imagery, or
  observations from any client work. No employer hardware or compute. Domain
  knowledge informs which scenarios to slice on; domain data never enters the
  repo. This is stated in the paper's methods and must stay there.
- **Never hardcode, echo, print, or commit an API key.** Environment variables
  only. A key was accidentally exposed once already and had to be rotated. Use
  `read -rs VAR && export VAR` so it never enters shell history.
- **Do not run git write commands from a sandboxed/containerised environment**
  if it cannot delete files in the working tree. It leaves a stale
  `.git/index.lock` and blocks the user. This happened twice. Read-only git
  inspection is fine. Let the user run `git add/commit/push`.
- **`data/` and `results/` are gitignored** and must stay that way by default.

---

## Current state

Repo: `~/Developer/can-inkling-drive`, public on GitHub, CI green.
177 tests pass offline with no API key and no network.

| step | state |
|---|---|
| 1. Harness | done |
| 2. Provider selection | done — Baseten |
| 3. DriveLM data | done, downloaded, **but see the surprise below** |
| 4. 20-call pilot | **done — results below** |
| 5. Scale-up | in progress — Inkling blind conditions collected and scored |

### Layout

```
src/idq/
  adapters/      base.py (Question, option parsing), drivelm.py, fixture.py
  providers/     base.py, openai_compat.py, mock.py
  convert.py     free-form DriveLM -> balanced MCQ   <- the important one
  cohort.py      frozen balanced cohort + source/provenance validation
  probe.py       raw-data diagnostics
  cache.py       content-addressed resumable JSONL cache
  collect.py     network only
  score.py       offline only
  analyze.py     stats
  tables.py      paper-ready Markdown tables
  pilot.py       the 20-call pilot
  cli.py         inspect | collect | probe | pilot | score | analyze
```

Collection and scoring never share a process. A scoring bug costs zero dollars.

---

## Things you will get wrong if you don't read this

### 1. DriveLM train has ZERO multiple-choice questions

The original plan assumed ~26% of DriveLM questions were already MCQ. Verified
false. Across all **377,956** QA pairs in `v1_1_train_nus.json`, the string
"select the correct answer" appears **zero** times.

- **train** — has answers, no MCQs (free-form text)
- **val** (`v1_1_val_nus_q_only.json`) — 1,543 MCQs, but **every answer is `""`**;
  ground truth is withheld because val backs a public leaderboard

Use `--adapter drivelm_converted`, never `--adapter drivelm`, on train. The
plain `drivelm` adapter returns 0 questions from this file. It is retained only
because it correctly parses inline-MCQ files such as val.

### 2. What the converter does, and why it is defensible

`src/idq/convert.py` builds MCQs from train's free-form answers. This mirrors
DriveLM's own `extract_data.py` / `convert_data.py`, which generate the val
multiple-choice format from the same annotations — we do the equivalent on the
split where answers are public.

Three properties, each asserted by tests in `tests/test_convert.py`:

- **No generated text.** Every option is a verbatim human-written answer given
  to that same question template. We select options, never write them.
- **Balanced gold classes.** Raw DriveLM answers are extreme (89% "No", 92%
  "Going ahead", 99.8% "Low"). Equal sampling per class makes chance exactly
  1/k. Without it, a blind model scores ~90% on priors and the grounding gap
  collapses for reasons unrelated to grounding.
- **Counterbalanced option positions.** Gold is rotated across positions so a
  model that always answers "A" scores exactly chance.

Only **two templates** survived filtering, yielding **28,506** questions:

| template | category | options | n |
|---|---|---|---|
| "Is `<obj>` an object the ego vehicle should consider?" | planning | Yes / No | 15,294 |
| "What is the observed status of `<obj>`?" | perception | Moving / Stationary | 13,212 |

**Chance is 0.50, not 0.25.** Both are binary.

**This is a real scope reduction** and the paper must claim only what it
measures: two balanced binary driving-QA tasks, in two categories. No
prediction, no behavior. Do not describe it as "driving QA" generally.

### 3. Four conditions, and why there are two blind ones

`clean`, `blind_tags`, `blind_notags`, `corrupt`.

DriveLM question text embeds object references like
`<c1,CAM_FRONT,1088.3,497.5>` naming a camera and a pixel coordinate. A
text-only run that leaves those in is **not blind**. Running both variants
decomposes the gap three ways:

```
total gap          = acc(clean) - acc(blind_notags)
image contribution = acc(clean) - acc(blind_tags)
annotation leakage = acc(blind_tags) - acc(blind_notags)
```

This is the project's main methodological contribution. Descendant of
hypothesis-only baselines in NLI (Gururangan et al., NAACL 2018).

### 4. Provider details

```
base_url      https://inference.baseten.co/v1
model_string  thinkingmachines/inkling
api_key_env   BASETEN_API_KEY
pricing       $1.00 / 1M input, $0.17 cached, $4.05 / 1M output  (quoted 2026-07-31)
rate limits   15 requests/min, 100K tokens/min  (Preview tier)
```

Together lists Inkling as "coming soon to Serverless" (not available).
Fireworks serves it on-demand GPU only, not per-token.

`reasoning_content` arrives as a **sibling field alongside a populated
`content`**, not as a replacement. Do not concatenate it into the answer text —
the model's deliberation mentions multiple option letters and would poison the
answer parser. `openai_compat.py` handles this; `tests/test_provider_payloads.py`
pins it.

**Do not use build.nvidia.com** even though it hosts Inkling. The owner is an
NVIDIA contractor and running personal research on the client's platform muddies
the "no employer compute" boundary.

---

## Pilot results (2026-07-31, git_sha 29bdab8)

20 calls, `blind_tags`, `--thinking-effort medium`, temperature 0, seed 0.
Cache at `results/pilot.jsonl`.

| metric | value | reading |
|---|---|---|
| successes | 20/20 | payload shape works |
| extraction method | `answer_tag` 20/20 | Inkling obeys "Answer: X" perfectly |
| invalid rate | 0.000 | nothing unparseable |
| `reasoning_tokens` reported | 20/20 | **RQ2 is safe, no proxy needed** |
| prompt tokens | mean 110 | |
| completion tokens | mean 52, median 54 | far shorter than assumed |
| reasoning tokens | mean 42, median 44 | |
| latency | median 559 ms, p95 1167 ms | |
| measured cost | **$0.000322 / call** | |
| accuracy | 0.500, CI [0.300, 0.700] | n=20; uninformative, ignore |
| determinism | **4 of 5 identical** | see below |

### Two consequences

**(a) The provider does NOT disclose quantization.** Every response reports
`"model": "thinkingmachines/inkling"`. Baseten's published docs example showed
`inferact/inkling-nvfp4`, and `docs/provider-selection.md` and
`docs/paper-methods.md` §4 both currently assert NVFP4 on that basis. **That
claim is not supported by the actual responses.** Record quantization as
`unknown` and correct both documents. This is exactly the kind of unverified
detail a reviewer catches.

**(b) Output is NOT deterministic.** 4 of 5 repeated identical requests came
back byte-identical at temperature 0 with a fixed seed; one did not. So
run-to-run variance is real and must be reported as *provider nondeterminism*,
not as seed variance. Multiple runs measure the provider, not the seed. Decide
deliberately whether to run one seed and report the variance qualitatively, or
run three and characterise it — the latter triples cost.

### Budget, from measured numbers

At $0.000322/call, 600 questions × 2 blind conditions × 5 models = 6,000 calls
≈ **$1.93**. All four conditions ≈ $3.90, though image inputs will raise the
input-token count substantially. A payment card is now attached to the Baseten
account, but every collection still needs an explicit small spending ceiling.

---

## Previously stale documents — fixed 2026-07-31

- `docs/paper-methods.md` now documents the converter, two-task scope, 0.50
  chance, and unknown Inkling quantization.
- `docs/provider-selection.md` now separates Baseten's documentation example
  from the actual pilot response strings and records quantization as unknown.
- `docs/runbook.md` and `README.md` now use `drivelm_converted` for train and
  report the actual v1.1 counts.

---

## Decisions locked before scale collection (2026-07-31)

1. **Frozen cohort.** 600 questions, equal task weight and exact joint balance.
   Manifest: `study/cohorts/drivelm-balanced-600.json`; cohort ID begins
   `d0c80605189e`.
2. **Models.** Inkling; Qwen3-VL-235B-A22B-Thinking; GLM-4.5V; Llama 4
   Maverick; GPT-5.6 Sol. Exact serving snapshot: `study/model-slate.json`.
3. **Primary comparison.** Inkling vs. Qwen3-VL-235B-A22B-Thinking.
4. **Runs.** One primary seed-0 pass plus one identical pass on the balanced
   40-question repeat subset to measure provider nondeterminism.
5. **RQ4.** Motion blur, severity 3, corruption seed 0.

Venue remains open; it does not affect collection.

---

## Inkling blind-condition results (2026-08-01)

Collection ran from clean commit `12a2bde` with the frozen cohort and explicit
spending ceilings. `blind_tags` stopped at 558/600 when its ceiling was reached;
`blind_notags` completed 600/600. All 1,158 calls succeeded, all answers were
valid answer-tag extractions, and all rows reported reasoning tokens. Total
measured spend was **$0.387850**.

The paired comparison uses the 558 shared question IDs. Tagged accuracy was
301/558 = 0.5394; no-tag accuracy was 293/558 = 0.5251. Annotation leakage was
+0.0143, paired bootstrap 95% CI [-0.0323, +0.0609], exact McNemar p=0.6040.
This is inconclusive and should not be described as evidence that the true
effect is zero. Neither blind condition substitutes for the uncollected clean
image condition.

Publication-oriented details, limitations, and artifact hashes are in
`docs/results-log.md`. The machine-readable aggregate receipt is
`study/runs/2026-08-01-inkling-blind.json`; raw and scored rows remain under the
gitignored `results/` directory.

## Immediate next step

Do not spend more merely to fill the 42 missing tagged rows; the matched
558-question comparison is the defensible budget-bounded analysis. First review
and commit the aggregate receipt and results log. The official DriveLM training
image archive is about 450 MB, but its unauthenticated endpoint returned HTTP
401 on 2026-08-01 and this machine has no Hugging Face login tooling configured.
Complete the maintainers' access step and download the official archive; do not
use an unofficial mirror to bypass the gate. Then verify the clean/corrupt image
path entirely offline before proposing a single explicitly capped smoke call or
any larger RQ3/RQ4 image collection.

Every record carries the cohort ID, and every invocation appends a
credential-free receipt to `results/run-log.jsonl`.

Images (`drivelm_nus_imgs_train.zip` from the DriveLM HuggingFace repo) are
needed only for `clean` and `corrupt`. Do **not** use nuScenes-mini; DriveLM
publishes its own image subset covering exactly the frames it annotates.

Run `python -m idq.cli --help` for the CLI. `pytest` should show all green
before and after any change.
