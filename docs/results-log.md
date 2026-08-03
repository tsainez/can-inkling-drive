# Study results log

This file records completed study runs in a form suitable for later paper
drafting. Raw responses, scored rows, and provider receipts remain under the
gitignored `results/` directory. Aggregate records here contain no credentials,
client data, or private imagery.

## 2026-08-01 — Inkling blind-condition collection

### Scope and provenance

- Research question: RQ3 annotation leakage,
  `accuracy(blind_tags) - accuracy(blind_notags)`.
- Public source only: DriveLM v1.1 train annotations converted with
  `--adapter drivelm_converted`. No client or employer data, imagery, logs,
  telemetry, hardware, or compute was used.
- Model requested and served: `thinkingmachines/inkling` through Baseten.
- Quantization: unknown; the response did not disclose it.
- Code commit: `12a2bde` on `main`, with a clean tree during collection.
- Frozen cohort ID:
  `d0c80605189e33dc2b84884c3a5a30a7018f5669230dbdc1b6a4222082c71dc6`.
- Decode: temperature 0, top-p 1, seed 0, maximum 2,048 tokens, medium
  thinking effort. The seed does not imply deterministic provider output.
- Price stamp: 2026-07-31, $1.00 per million input tokens and $4.05 per
  million output tokens.

### Collection outcome

| condition | successful rows | invalid rows | cost (USD) | stop reason |
|---|---:|---:|---:|---|
| `blind_tags` | 558/600 | 0 | 0.194938 | spending ceiling reached |
| `blind_notags` | 600/600 | 0 | 0.192912 | cohort complete |
| **total** | **1,158** | **0** | **0.387850** | |

The tagged condition consists of one smoke call ($0.000303) followed by 557
new successful calls ($0.194635). The collector stopped between calls when its
measured continuation spend crossed the $0.194500 ceiling. This is a planned
safety-stop behavior, not provider attrition. There were no terminal or
retryable failures in either condition.

All 1,158 rows used the expected cohort ID and served-model string. Every
response was extracted through the answer tag, and every response reported
reasoning-token usage.

### Accuracy and paired comparison

Unmatched condition summaries are included for completeness:

| condition | correct / n | accuracy | bootstrap 95% CI |
|---|---:|---:|---:|
| `blind_tags` | 301 / 558 | 0.5394 | [0.4982, 0.5806] |
| `blind_notags` | 312 / 600 | 0.5200 | [0.4800, 0.5600] |

The confirmatory annotation-leakage comparison uses only the 558 question IDs
present in both conditions:

| paired metric | value |
|---|---:|
| tagged accuracy | 301 / 558 = 0.5394 |
| no-tag accuracy | 293 / 558 = 0.5251 |
| tag effect | +0.0143 (+1.43 percentage points) |
| paired bootstrap 95% CI | [-0.0323, +0.0609] |
| exact McNemar p-value | 0.6040 |
| correct in both | 206 |
| incorrect in both | 170 |
| tagged only correct | 95 |
| no-tag only correct | 87 |

The matched prefix remains exactly balanced across tasks (279 perception and
279 planning) but is not exactly balanced across gold letters (276 A and 282
B), because the budget stop occurred before the full jointly balanced cohort.

Interpretation: this run finds no detectable annotation-leakage effect. It does
not establish that the effect is zero; the interval still permits effects in
either direction. Both blind accuracies are close to the 0.50 chance level.
Without the clean image condition, these results cannot show that images are or
are not useful and cannot estimate the total grounding gap.

### Efficiency

| condition | mean reasoning tokens | median | p95 | USD / correct |
|---|---:|---:|---:|---:|
| `blind_tags` | 48.82 | 45 | 87.3 | 0.000648 |
| `blind_notags` | 45.07 | 36 | 83.0 | 0.000618 |

These are provider-reported reasoning tokens, not a text-length proxy.

### Local artifacts and integrity hashes

| artifact | SHA-256 |
|---|---|
| `results/inkling-blind-cache.jsonl` | `b6f315cc0c576548b191b3ba419349b2cc9b014f58150f8485276d35e4a802ee` |
| `results/inkling-blind-scored.jsonl` | `3f4292ce1805854adf70dfc64efd7325378107af37be58657626bca0d9e4b6d1` |
| `results/inkling-blind-report.json` | `54aca789496b189e57e3c4a1ff28bc749059b79d0de6928bfee19ba6d8e0358f` |
| `results/inkling-blind-tables.md` | `ca5954f9d1933da35958ddcaf3ea485203287220b46afbb20a9f248080834b1c` |
| `results/run-log.jsonl` | `35505f7da082369612f5fc89449ffd8eee030dc7366462b8e3f579bac3d8792d` |
| `study/cohorts/drivelm-balanced-600.json` | `d419165a4b850e66ede64cd5928f570a206127647518979cb3bf0bff99c0c3e8` |

The machine-readable aggregate receipt is
`study/runs/2026-08-01-inkling-blind.json`.

## 2026-08-03 — Image-path preflight (offline, zero cost)

This was a local readiness check, not a model evaluation and not a paid
collection.

- Official Hugging Face repository: `OpenDriveLab/DriveLM`.
- Cached snapshot: `ae973fbd4e8d4684af4ab234d504bd6c5e946868`.
- Train-image archive: 3,483,205,396 bytes; SHA-256
  `2cb95fafac00ca058c04735c339a183657d60962afee12d782b410c602eab936`.
- The full ZIP passed CRC validation.
- Correct local layout: annotations under `data/drivelm/`, images under
  `data/nuscenes/samples/`, and collection with `--image-root data/drivelm`.
- All 600 frozen cohort questions resolved the camera named in their object tag
  with zero missing paths; these references cover 582 unique images. The camera
  distribution is CAM_FRONT 230, CAM_BACK 164, CAM_FRONT_LEFT 81,
  CAM_FRONT_RIGHT 70, CAM_BACK_RIGHT 32, and CAM_BACK_LEFT 23.
- The collector now hashes and validates all required images before its first
  provider call, preventing a late missing view from consuming paid calls.
- A representative source was 1600x900. Clean encoding preserved that native
  resolution. Motion blur at preregistered severity 3 produced different bytes,
  while the source SHA-256 was identical before and after encoding.
- Provider calls: 0. Cost: $0.00.

The rehearsal ran at committed base `373ae67` with uncommitted image-pipeline
changes, so its hashes are diagnostic rather than a formal study receipt. A
paid image smoke must wait for a reviewed clean commit. Actual image-token cost
is still unknown; estimate it with one call under a $0.002 ceiling before any
larger clean or corrupt condition.

## 2026-08-03 — Inkling clean-image smoke

One paid call was made solely to validate Baseten's native-resolution image
payload and measure billing. Collection ran from clean commit `3f532d6` after
all 188 offline tests passed.

| field | value |
|---|---:|
| calls / successes | 1 / 1 |
| camera | `CAM_FRONT` |
| source resolution | 1600x900 |
| prompt tokens | 1,614 |
| completion tokens | 79 |
| reasoning tokens | 69, provider-reported |
| latency | 2,087.7 ms |
| measured cost | **$0.00193395** |
| answer validity | valid, `answer_tag` |

The served-model string was `thinkingmachines/inkling`, the cohort ID matched,
and there were no terminal or retryable failures. The answer was incorrect, but
correctness at n=1 is deliberately not interpreted. This call establishes only
that the image payload works and that a representative clean call cost about
$0.00193.

A straight 600-call extrapolation is approximately $1.16037 for Inkling clean,
assuming similar usage. This is a planning estimate, not a budget authorization
or an accuracy result; output length and image billing can vary. Cumulative
measured spend for the two blind conditions plus this smoke is **$0.389784**.

Artifacts:

| artifact | SHA-256 |
|---|---|
| `results/inkling-clean-smoke.jsonl` | `4e9874a2231f922f3453f3207607bf670d9ee21b7c979d467929cebb8e98187d` |
| `results/inkling-clean-smoke-scored.jsonl` | `85cbda2e645943d11e39373fa6099ae5aa5f91bcd8e8ed820251f6db3744af30` |
| `results/run-log.jsonl` after the smoke | `f8de9da0bd196c9ef8468ce8da738c80a1e86e758c97d61b920ae454502633eb` |

The machine-readable receipt is
`study/runs/2026-08-03-inkling-clean-smoke.json`. The earlier run-log hash in
the blind-condition receipt identifies its historical append-only prefix; it
is not expected to equal the hash after this later receipt was appended.

## 2026-08-03 — Baseten credit extension

The owner authorized use of the remaining Baseten credit under explicit
collector ceilings. Collection ran from clean commit `a6eac60`, after 188
offline tests passed. Public DriveLM data and the public Inkling model were the
only research inputs; no client or employer data, imagery, logs, telemetry,
hardware, or compute was used. Train annotations were loaded only through
`--adapter drivelm_converted`.

First, 42 calls completed the previously budget-stopped `blind_tags`
condition. All 42 succeeded and cost $0.0143708. Both blind conditions now
contain all 600 frozen-cohort questions:

| condition | correct / n | accuracy | bootstrap 95% CI |
|---|---:|---:|---:|
| `blind_tags` | 326 / 600 | 0.5433 | [0.5033, 0.5833] |
| `blind_notags` | 312 / 600 | 0.5200 | [0.4800, 0.5600] |

The complete paired annotation-leakage estimate is +0.0233, with bootstrap
95% CI [-0.0233, +0.0700] and exact McNemar p=0.3556. It remains
inconclusive; the interval permits effects in both directions.

The remaining authorized ceiling was applied to `clean`. The collector made
238 successful calls with zero errors or invalid answers, then stopped before
call 239 because measured spend had reached $0.50175085. All answers used the
expected answer-tag extraction, and every row reported reasoning tokens.

| clean partial metric | value |
|---|---:|
| correct / n | 143 / 238 |
| accuracy | 0.6008 |
| bootstrap 95% CI | [0.5378, 0.6639] |
| prompt tokens | 383,665 |
| completion tokens | 29,157 |
| reasoning tokens | 26,777 |

This is a budget-stopped prefix, not the preregistered full clean condition. It
contains 128 perception and 110 planning questions, so the following paired
comparisons are **interim exploratory evidence only**:

| paired comparison on the same 238 IDs | effect | bootstrap 95% CI | exact McNemar p |
|---|---:|---:|---:|
| clean - `blind_tags` (image contribution) | +0.0420 | [-0.0294, +0.1134] | 0.3082 |
| clean - `blind_notags` (total gap) | +0.1134 | [+0.0294, +0.1975] | 0.0132 |

The second row is interesting, but treating it as the paper's confirmatory
answer would be premature: the stopping point was imposed by budget, the
prefix is not task-balanced, and 362 clean questions remain uncollected.

This extension cost $0.51612165. Cumulative measured project spend, including
the blind runs and clean smoke, is $0.90590565. No further paid call was made
after the ceiling stop.

### Public question-level artifact

`study/public-results/inkling-through-2026-08-03.jsonl` contains 1,438
deterministically sorted, publication-safe rows: 600 `blind_tags`, 600
`blind_notags`, and 238 clean. Its SHA-256 is
`541411e301386344e1a4777856cffcf754f3f1c86bfde1bd718c499bce0afdbb`.

The exporter uses a fixed allowlist and excludes prompts, source annotations,
response and reasoning text, images and local paths, scene and frame IDs,
provider payloads, cache keys, and credentials. Raw caches and richer scored
rows remain gitignored under `results/`. The machine-readable aggregate receipt
is `study/runs/2026-08-03-inkling-credit-extension.json`.

Current local artifact hashes:

| artifact | SHA-256 |
|---|---|
| `results/inkling-blind-cache.jsonl` | `c652cb2c422e21c6d35742829b4a32a443fa3e9d83a550c7458412753e81e8a7` |
| `results/inkling-blind-scored.jsonl` | `4216a66a1dd94096a7bb2ea5671e6c024a04adba4d7682ded0f68450d5d32859` |
| `results/inkling-clean-cache.jsonl` | `1e84c1cdc86d4ee91075dd25311f0c6a34b1839590dc576ffad7d9a254622f9f` |
| `results/inkling-clean-scored.jsonl` | `b7a90c3a47bbf6ae04954fc9960d02e9ae97522149b85c54c126822e3a934138` |
| `results/run-log.jsonl` | `eea1cfcf21cdfe5f869d17a48d7035feb6605e039cc30035c967d558370de94b` |
