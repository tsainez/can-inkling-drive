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
python -m idq.cli inspect --adapter drivelm --data data/drivelm/v1_1_train_nus.json
```

That prints the MCQ share, the per-category distribution of the MCQ subset
versus all QA, and the expected chance accuracy. **Read those three numbers
before doing anything else.** Specifically:

- If `mcq_share_of_all_qa` is far from your assumed ~26%, the sampling plan
  changes.
- If `mcq_by_category` is concentrated in one or two categories, your
  conclusions are about that slice, and the paper says so.
- If `expected_chance` isn't ~0.25, the MCQ subset has non-4-option questions
  and every "above chance" claim needs the measured value.
- `unresolvable_gold` counts questions whose answer the parser could not map to
  a letter. A large number means the gold format varies and the adapter needs
  another case, not that the questions are bad.

Reference scale from the DriveLM docs: the training set is **29,448 QA pairs
across 4,072 frames**. At ~26% MCQ that is roughly 7,600 usable questions —
comfortably more than the 2,000 the full study needs.

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

## Step 4: the 20-call pilot

Dry run first — renders prompts, makes zero calls, costs nothing:

```bash
export BASETEN_API_KEY=...            # environment only, never in a file

python -m idq.cli collect \
  --adapter drivelm --data data/drivelm/v1_1_train_nus.json \
  --provider openai_compat \
  --base-url https://inference.baseten.co/v1 \
  --api-key-env BASETEN_API_KEY \
  --model-string thinkingmachines/inkling \
  --model-label inkling --quantization nvfp4 \
  --condition blind_tags --sample 20 --dry-run
```

Then the real 20 calls:

```bash
python -m idq.cli pilot \
  --adapter drivelm --data data/drivelm/v1_1_train_nus.json \
  --provider openai_compat \
  --base-url https://inference.baseten.co/v1 \
  --api-key-env BASETEN_API_KEY \
  --model-string thinkingmachines/inkling \
  --model-label inkling --quantization nvfp4 \
  --condition blind_tags \
  --usd-in <from console> --usd-out <from console> --price-date 2026-07-24 \
  --budget 15 --n 20 --n-repeat 5 \
  --cache results/pilot.jsonl
```

### What the pilot output tells you

| field | what to do about it |
|---|---|
| `payload_ok: false` | Stop. Model string, base URL or key. Nothing else matters yet. |
| `reasoning_tokens_reported_frac` | Should be 1.0 on Baseten. If 0.0, RQ2 is on a proxy and the paper must say so. |
| `served_models` | Expect `inferact/inkling-nvfp4`. More than one value means the provider swapped builds mid-run — results aren't comparable. |
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

```bash
for c in blind_tags blind_notags; do
  python -m idq.cli collect ... --condition $c --sample <n from pilot> \
    --cache results/cache.jsonl
done

python -m idq.cli score   --adapter drivelm --data data/... --cache results/cache.jsonl --out results/scored.jsonl
python -m idq.cli analyze --scored results/scored.jsonl
```

Interrupt any of this freely. The cache is append-only with an fsync per
record; rerunning the same command resumes and re-pays for nothing.
