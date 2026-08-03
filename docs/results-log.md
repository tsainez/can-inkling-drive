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
