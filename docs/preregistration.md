# Preregistered study decisions

**Locked 2026-07-31, before scale collection.** The completed 20-call Inkling
pilot was an operational payload/cost check; its n=20 accuracy is ignored for
model selection and inferential claims.

## Scope and IP boundary

This study evaluates only the slow reasoning layer—offline scene
interpretation and explanation—not closed-loop control. It uses public model
weights or publicly accessible commercial models and public DriveLM data only.
No client data, telemetry, logs, imagery, observations, hardware, or compute
may enter the study.

## Frozen cohort

- Manifest: `study/cohorts/drivelm-balanced-600.json`
- Cohort ID: `d0c80605189e33dc2b84884c3a5a30a7018f5669230dbdc1b6a4222082c71dc6`
- Source: DriveLM-nuScenes v1.1 train; source SHA-256 is stored in the manifest.
- Converter: `drivelm_converted`, seed `20260731`.
- Weighting: equal task weight, 300 planning and 300 perception questions.
- Balance: 150 per answer class; 300 A and 300 B; 75 in every template ×
  answer-class × option-position cell.
- Chance accuracy: 0.50.

The same ordered question IDs must be used for every model and condition. A
plain random `--sample` is prohibited for final collection.

## Model slate and primary comparison

The exact serving snapshot is in `study/model-slate.json`.

1. Inkling, served by Baseten; quantization unknown.
2. **Qwen3-VL-235B-A22B-Thinking**, public Apache-2.0 weights, pinned to
   OpenRouter's Novita BF16 endpoint.
3. GLM-4.5V, public MIT-licensed weights, pinned to Novita FP8.
4. Llama 4 Maverick, public weights under the Llama 4 Community License,
   pinned to Novita FP8.
5. GPT-5.6 Sol, closed frontier reference, pinned to an available Azure route;
   quantization unknown.

**Primary comparison: Inkling versus Qwen3-VL-235B-A22B-Thinking.** This pair is
exempt from Holm correction. Qwen is the closest available public-weight,
reasoning-oriented VLM with per-token serving, although its 235B total / 22B
active parameters remain smaller than Inkling's 975B / 41B. The size mismatch
is reported rather than obscured.

If a pinned endpoint is unavailable, collection stops. No provider, model,
quantization, or fallback route may be substituted after results are visible.

## Conditions and decoding

- `clean`, `blind_tags`, `blind_notags`, and `corrupt` for every model.
- RQ4 corruption: deterministic `motion_blur`, severity 3, corruption seed 0.
- Temperature 0 and top-p 1 where the endpoint supports them; unsupported
  sampling fields are omitted rather than emulated.
- Reasoning effort `medium` where controllable. Mandatory-reasoning models use
  their served thinking mode. Each request records the exact request profile.
- Primary generation seed 0. Provider variation is not called seed variance.
- One primary pass per question. A second identical pass on the manifest's
  balanced 40-question repeat subset estimates provider nondeterminism without
  tripling the entire study.

## Outcomes and analysis

- RQ1: exact accuracy, invalid-output rate, bootstrap CI, paired exact McNemar;
  Holm correction except for the named primary comparison.
- RQ2: accuracy versus reasoning tokens when uniformly reported; otherwise all
  models use completion tokens. Dollar results are a dated serving snapshot.
- RQ3: clean-minus-blind decomposition into image contribution, annotation
  leakage, and total grounding gap.
- RQ4: paired clean-to-motion-blur loss and supported rank inversions.

Every paid call must carry the cohort ID, commit SHA, prompt hash, full raw
response, served model/provider, decode configuration, token usage, dated
pricing, and append-only run receipt. Live collection is refused from a dirty
working tree. Inkling blind-condition collection is paced at 15 requests/minute
and capped at $0.195 per 600-question condition ($0.39 total target, with at
most one-call overshoot per condition).
