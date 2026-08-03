# Runbook — steps 3 to 5

## Step 3: DriveLM annotations

**Use the v1.1 train split. There is no alternative.**

The val split ships as `v1_1_val_nus_q_only.json` — questions only. The repo
states plainly that ground-truth answers for val will **not** be released,
because val backs the leaderboard. Any evaluation with objective scoring has to
run on train. Say so in the paper's methods in one line; it is normal practice
and a reviewer only objects if you hide it.

Annotations, no images, small enough for hotel WiFi:

```
https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/v1_1_train_nus.json
```

The maintainers ask you to fill in their Google Form before downloading. Do it —
it costs a minute and it is their citation-tracking mechanism.

Save to `data/drivelm/v1_1_train_nus.json` (gitignored — never commit benchmark
data), then:

```bash
python -m idq.cli inspect --adapter drivelm_converted --data data/drivelm/v1_1_train_nus.json
```

The train file contains **377,956 free-form QA pairs and zero native
multiple-choice questions**. The plain `drivelm` adapter therefore returns zero
questions from this file; it is retained for inline-MCQ files such as val and
must never be used on train.

The `drivelm_converted` adapter mirrors DriveLM's own conversion approach on
the split whose answers are public. It selects only verbatim human-written
answers from the same question template, balances gold classes, and
counterbalances option positions. The inspection should report:

- `n_questions: 28506` and `questions_built: 28506`;
- exactly two used templates: 15,294 planning Yes/No questions and 13,212
  perception Moving/Stationary questions;
- `expected_chance: 0.5`, because both tasks are binary;
- exactly balanced gold classes, plus near-balanced full-pool gold positions
  (`A: 14254`, `B: 14252`; the two-item difference comes from odd-sized class
  buckets).

If these values change, stop and explain the source-data or converter change
before collecting. Claims are limited to these two balanced binary tasks—not
DriveLM or driving QA generally, and not prediction or behavior.

Reproduce and verify the publication cohort (this must return cohort ID
`d0c80605189e33dc2b84884c3a5a30a7018f5669230dbdc1b6a4222082c71dc6`):

```bash
python -m idq.cli cohort \
  --adapter drivelm_converted --data data/drivelm/v1_1_train_nus.json \
  --convert-seed 20260731 --n-per-template 300 \
  --out study/cohorts/drivelm-balanced-600.json
```

## Step 3b: images, when you get to them

Do **not** download nuScenes-mini. I flagged earlier that mini's 10 scenes might
not overlap your sampled questions; that turned out to be the wrong thing to
worry about, because DriveLM publishes its own image subset covering exactly the
frames it annotates:

```
https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/drivelm_nus_imgs_train.zip
```

Unzip so paths resolve as `<image_root>/samples/CAM_FRONT/<file>.jpg`, then pass
`--image-root`. Blind conditions need none of this.

As checked on 2026-08-01, an unauthenticated request to the official image-file
endpoint returns HTTP 401. Complete the maintainers' access step and download
while authenticated. Do not work around the gate with an unofficial mirror;
the public-data boundary includes honoring the dataset's access process.

## Step 4: the 20-call pilot

### Pre-flight — do these before the first paid call

Both of these fail silently. Nothing errors; the numbers just turn out to be
unreportable afterwards, and the only fix is to collect again and pay again.

- [ ] **Commit the repository.** `git_sha` comes from `git rev-parse --short HEAD`
      and goes into every cache record. Live collection now refuses a dirty
      tree because a commit SHA alone cannot reproduce uncommitted changes.
- [ ] **Write the primary comparison down, here, before collecting.** It is
      exempt from Holm correction, which is only defensible if it was chosen in
      advance. Picking it after seeing the results is p-hacking with extra steps.

      ```
      primary comparison: inkling vs qwen3-vl-235b-thinking   # decided 2026-07-31, pre-scale
      ```

- [ ] **Decide the thinking-effort value and never change it.** It is in the
      cache key, so a change forces re-collection rather than silently mixing
      settings.

`idq.cli analyze` warns about the first two, but it warns after the money is
gone. That is why they are a checklist and not a validation step.

### Dry run first — renders prompts, makes zero calls, costs nothing:

```bash
read -rs BASETEN_API_KEY && export BASETEN_API_KEY

python -m idq.cli collect \
  --adapter drivelm_converted --data data/drivelm/v1_1_train_nus.json \
  --provider openai_compat \
  --base-url https://inference.baseten.co/v1 \
  --api-key-env BASETEN_API_KEY \
  --model-string thinkingmachines/inkling \
  --model-label inkling --quantization unknown \
  --condition blind_tags --sample 20 \
  --thinking-effort medium --dry-run
```

Then the real 20 calls:

```bash
python -m idq.cli pilot \
  --adapter drivelm_converted --data data/drivelm/v1_1_train_nus.json \
  --provider openai_compat \
  --base-url https://inference.baseten.co/v1 \
  --api-key-env BASETEN_API_KEY \
  --model-string thinkingmachines/inkling \
  --model-label inkling --quantization unknown \
  --condition blind_tags --thinking-effort medium \
  --usd-in 1.00 --usd-out 4.05 --price-date 2026-07-31 \
  --budget 15 --n 20 --n-repeat 5 \
  --cache results/pilot.jsonl
```

### What the pilot output tells you

| field | what to do about it |
|---|---|
| `payload_ok: false` | Stop. Model string, base URL or key. Nothing else matters yet. |
| `reasoning_tokens_reported_frac` | Should be 1.0 on Baseten. If 0.0, RQ2 is on a proxy and the paper must say so. |
| `served_models` | The pilot reported `thinkingmachines/inkling`, which does not disclose quantization. More than one value means the provider swapped builds mid-run—results may not be comparable. |
| `invalid_rate` | Above ~10%, raise `--max-tokens` before scaling. A reasoning model truncated mid-thought never reaches its answer line. |
| `determinism.identical_frac` | **1.0 → run one seed.** Below 1.0 → the variation is provider nondeterminism, and the paper reports it as run-to-run variance, not seed variance. |
| `sizing.binding_constraint` | `budget` means money limits you; `precision` means you already have enough calls for a ±4pp interval and more spending buys little. |
| `sizing.recommended_questions` | Your sample size, from measurement rather than assumption. |

You can rehearse all of this offline first — the mock accepts hypothetical
prices, so you can see the sizing math before spending a cent:

```bash
python -m idq.cli pilot --n 20 --usd-in 1.87 --usd-out 4.68 --budget 15
```

### On thinking effort

Pinned for the whole study. Pick one value, pass `--thinking-effort`, and never
change it — it's in the cache key, so a change forces re-collection rather than
silently mixing settings. Note in the pilot whether `reasoning_tokens` actually
responds to the setting: a provider that ignores the parameter looks identical
to one that honours it, and the only way to tell is to try two values on a
handful of calls and see whether the token counts move.

## Step 5: scaling up

Blind conditions first — they need no images and run on any connection:

**Do not use a plain `--sample 600` for the final study.** With the default
sample seed it produces 316 A versus 284 B answers and also unbalances the four
gold classes; an always-A model would score 52.7%, not 50%. Before any scale
collection, preregister and freeze one question set balanced jointly by
template, gold class, and gold position, then reuse it for every model and
condition. The task-weighting choice (equal templates or approximately
pool-proportional templates) must be explicit. It is now locked to equal
templates in `docs/preregistration.md`.

```bash
for c in blind_tags blind_notags; do
  python -m idq.cli collect \
    --adapter drivelm_converted --data data/drivelm/v1_1_train_nus.json \
    --cohort study/cohorts/drivelm-balanced-600.json \
    --provider openai_compat --base-url https://inference.baseten.co/v1 \
    --api-key-env BASETEN_API_KEY \
    --model-string thinkingmachines/inkling --model-label inkling \
    --quantization unknown --thinking-effort medium --condition $c \
    --usd-in 1.00 --usd-out 4.05 --price-date 2026-07-31 \
    --requests-per-minute 15 --max-usd 0.195 \
    --cache results/cache.jsonl
done

python -m idq.cli score --adapter drivelm_converted --data data/... \
  --cohort study/cohorts/drivelm-balanced-600.json \
  --cache results/cache.jsonl --out results/scored.jsonl

python -m idq.cli analyze --scored results/scored.jsonl \
  --primary "inkling,qwen3-vl-235b-thinking" \
  --markdown results/tables.md --json-out results/report.json
```

Interrupt any of this freely. The cache is append-only with an fsync per
record; rerunning the same command resumes and re-pays for nothing.

`--markdown` writes the paper's tables — main results, all four RQ sections, and
the two appendices — generated from the same report rather than retyped. Scoring
and analysis cost nothing, so rerun them as often as you like.

### What analyze warns about

Warnings print to stderr and lead the Markdown document, because each one makes
some number unquotable until it is resolved:

| warning | what it means |
|---|---|
| `no git_sha on any record` | Collected from an uncommitted tree. Not traceable. |
| `more than one value of served_model` | The provider swapped builds mid-run. Those records are not one condition. |
| `reasoning_tokens never reported` | RQ2 is on a `completion_tokens` proxy and the paper must say so. |
| `priced with no price_quoted_on` | A dollar figure with no quote date. Not reportable. |
| `invalid-output rate > 10%` | Raise `--max-tokens`; a truncated reasoning chain never reaches its answer line. |
| `no primary comparison designated` | Every pair is being Holm-corrected, including the one you care about. |

### Reading the RQ4 output

`robustness` gives the per-model accuracy loss. Two fields do work the raw delta
cannot:

- `above_chance_retention` — the share of *above-chance* accuracy that survives.
  0.90 → 0.50 and 0.50 → 0.30 are both −20pp, but the first keeps 38% of its
  signal and the second keeps 8%. Blank when baseline accuracy is at or below
  chance, because then there was no signal to retain.
- `invalid_rate_delta` — corruption can make a model stop answering rather than
  answer wrongly. An accuracy-only comparison cannot see the difference.

`ranking_stability` answers the actual RQ4 question. `kendall_tau_b` says how far
the ordering moved; **no p-value is attached**, because a rank test on five
models has almost no power and a non-significant tau would say nothing. The
load-bearing field is `n_supported_inversions`: an order flip only counts if the
pair was significantly ordered one way clean and significantly the other way
corrupted. Two models half a point apart that swap places were never
distinguishable, and `ranking_preserved` ignores them.
