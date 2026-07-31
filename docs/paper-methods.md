# Methods — working draft

Written before collection on purpose. If the methods cannot be stated clearly
now, the design has a hole, and it is much cheaper to find it here than after
90,000 API calls. Numbers are placeholders marked `[TBD]`.

Venue-agnostic. Trim to fit once a target is chosen (IEEE IV and ITSC run
~6 pages; CVPR/ICCV/NeurIPS workshops vary from 4 to 8).

---

## 1. Scope and what this does not claim

This work evaluates the **slow reasoning layer** of vision-language models on
driving scenarios: scene understanding, edge-case interpretation, and decision
explanation, all in an offline question-answering setting.

It makes **no claim about closed-loop control.** No model in this study
produces actuator commands, is evaluated in simulation, or is placed in a
control loop. Results should not be read as evidence that a language model can
or should drive a vehicle. The question asked here is narrower and, we argue,
answerable: when a model reasons about a road scene, what does its accuracy
cost in thinking tokens and in dollars?

## 2. Ethics, data provenance, and conflicts

All models are publicly available or commercially available via public API. All
data is publicly released benchmark data. **No data, telemetry, logs, imagery,
or observations from any commercial autonomous-vehicle program contributed to
this work.** No employer hardware or compute was used; all inference was
purchased at personal expense from a commercial provider.

The author works in autonomous-vehicle field testing. That domain experience
informed *which* scenario categories are worth analysing separately — it
contributed no data. We state this explicitly because the distinction between
domain knowledge and domain data matters and is easy for a reader to assume
away.

## 3. Benchmark

We evaluate on **DriveLM-nuScenes v1.1** (Sima et al., ECCV 2024), restricted
to the subset of questions that are already formatted as multiple choice with a
single-letter ground-truth answer.

**Why this subset.** Multiple choice with a known correct letter permits exact
scoring. No LLM judge, no rubric, no BLEU or CIDEr against a reference answer.
This removes the largest source of methodological noise in driving-VQA
evaluation, at the cost of covering a narrower slice of the benchmark.

**Why human-annotated distractors.** Recent work (2026) finds that
VLM-generated multiple-choice distractors in driving benchmarks carry
linguistic regularities a model can exploit without consulting the image.
DriveLM's options were written by human annotators and therefore do not carry a
generator model's fingerprints. This matters directly for RQ3, where the entire
question is how much accuracy survives without the image.

**Split.** We use the **train** split. DriveLM's val split is released
question-only; ground-truth answers are withheld to support the public
leaderboard. Objective offline scoring is therefore only possible on train.

**Subset composition.** The MCQ subset is not a uniform sample of DriveLM. We
report its per-category distribution against the full QA distribution
(Table [TBD]) and restrict claims accordingly. Chance accuracy is computed as
the mean of 1/*k* over questions with *k* options rather than assumed to be
0.25, because option counts vary.

**Not used.** AutoDrive-QA (arXiv 2503.15778) is cited for its distractor
taxonomy, which informed our error-slice design, and its reported GPT-4V
figures are noted as context. We build nothing on it: its public repository
contains no code or data, and its reported size is inconsistent across
versions. Its numbers are on a different benchmark and are never tabulated
alongside ours.

## 4. Models and serving configuration

[TBD: five models — Inkling, three comparable open-weights models, one frontier
closed reference.]

Open-weights models are served by a commercial inference provider rather than
self-hosted; Inkling at 975B parameters requires over 2 TB of accelerator
memory at 16-bit precision, which is out of reach for an independent
researcher.

**Serving configuration is part of the result.** We report, for every model:
the exact model string, the provider, the date of collection, and the
quantization the provider reports serving. Inkling was served at **NVFP4**, as
disclosed by the provider in the `model` field of every response
(`inferact/inkling-nvfp4`). Quantization affects accuracy, and open-weights
numbers are not comparable across providers without it. Where a provider does
not disclose quantization we record it as unknown rather than assuming 16-bit.

**Decoding.** Temperature 0, top-p 1.0, fixed seed, max tokens [TBD].
Inkling exposes a controllable thinking-effort parameter; **all Inkling results
are collected at a single documented setting ([TBD])**. Sweeping thinking
effort to trace a within-model accuracy/token curve is the most promising
extension of this work and is left to future work. Consequently our
cross-model efficiency comparison places each model at one operating point,
not on its own frontier — a limitation we return to in §8.

**Determinism.** We verified empirically whether identical requests at
temperature 0 with a fixed seed return identical output [TBD result]. Batched
mixture-of-experts inference is not guaranteed deterministic, and several
providers ignore the seed parameter. Where output varies, we report it as
run-to-run provider variance and do not describe it as seed variation.

## 5. Conditions

Four conditions per model.

| condition | image | object tags in question text |
|---|---|---|
| `clean` | shown | present |
| `blind_tags` | withheld | present |
| `blind_notags` | withheld | replaced |
| `corrupt` | degraded | present |

**Why two blind conditions.** DriveLM question text embeds object references of
the form `<c1,CAM_FRONT,1088.3,497.5>`, naming a camera and a pixel coordinate.
A text-only condition that leaves these in place is not blind: the annotation
itself supplies scene-specific spatial information. Running both variants
decomposes the grounding gap:

```
total gap           = acc(clean) − acc(blind_notags)
image contribution  = acc(clean) − acc(blind_tags)
annotation leakage  = acc(blind_tags) − acc(blind_notags)
```

In `blind_notags`, tags are **replaced** with a neutral noun phrase rather than
deleted, so sentences remain grammatical and the contrast measures information
rather than fluency.

This decomposition addresses an ambiguity that a single blind condition cannot.
A small clean-versus-blind gap admits two readings — the model is not visually
grounded, or the benchmark is textually leaky — and the standard two-condition
design cannot separate them. Our approach is a direct descendant of
hypothesis-only baselines in natural language inference (Gururangan et al.,
2018), which revealed that annotation artifacts allowed models to classify
sentence pairs without seeing the premise.

**Corruption.** [TBD: one corruption] applied in memory at ImageNet-C severity
[TBD] (Hendrycks & Dietterich, ICLR 2019). Source images are never modified.
Motion blur is implemented as a normalized line-kernel convolution in NumPy
rather than via PIL, whose kernel filter is restricted to 3×3 and 5×5 and
cannot express a realistic streak length.

## 6. Answer extraction and scoring

Model output is parsed to a single option letter. The parser records **how**
each answer was extracted — explicit answer tag, sole letter, boxed
expression, final line, verbatim option text, or loose fallback — and flags
responses containing multiple irreconcilable candidates as ambiguous.

Where a provider returns chain-of-thought in a separate field, that text is
excluded from parsing. A letter mentioned mid-deliberation is not an answer,
and including it would inflate the ambiguity rate with the model's own
reasoning.

**Unparseable output is scored as incorrect and reported separately.** The
invalid-output rate is a property of the model's instruction-following and
belongs in the results table, not hidden in the denominator.

Collection and scoring are separate processes communicating only through an
append-only cache of full raw responses. No scoring change can require
re-collection.

## 7. Statistics

- **Accuracy** with 95% percentile bootstrap confidence intervals over
  questions (10,000 resamples).
- **Paired model comparison** by exact McNemar test on discordant pairs.
  We use the exact binomial form rather than the chi-square approximation
  because discordant counts at these sample sizes are frequently small.
- **Multiplicity.** One primary comparison is designated in advance
  ([TBD: Inkling vs. named baseline]); all remaining pairwise comparisons are
  Holm-corrected within condition.
- **Grounding gap** with a paired bootstrap CI over questions answered in all
  three grounding conditions.
- **Robustness** as the paired clean-versus-corrupt accuracy difference per
  model, with a bootstrap CI and an exact McNemar test. We additionally report
  the fraction of *above-chance* accuracy retained,
  (acc<sub>corrupt</sub> − chance) / (acc<sub>clean</sub> − chance), because
  equal absolute drops from unequal baselines are not equally severe; it is
  undefined and reported as such when clean accuracy is at or below chance. We
  also report the change in invalid-output rate, since degradation can move a
  model from answering wrongly to not answering at all.
- **Ranking stability** under degradation by Kendall's τ-b between the clean and
  corrupt model orderings, computed on the questions all models answered.
  **We attach no p-value to τ.** With five models a rank-correlation test has
  negligible power and a non-significant result would be uninformative. Instead
  we audit order inversions individually: a swap counts as evidence that the
  ranking changed only where the pair is significantly ordered one way in
  `clean` and significantly the opposite way in `corrupt`. Swaps between models
  that are not significantly separated are reported as consistent with noise.
- **Efficiency** reported primarily as a Pareto scatter of accuracy against
  mean reasoning tokens. We deliberately avoid a scalar accuracy-per-token
  ratio as a headline: it divides a bounded quantity by an unbounded one and is
  unstable and hard to interpret. Where any model in a comparison does not
  report `reasoning_tokens`, the shared axis for **every** model in that
  comparison falls back to total completion tokens, and the substitution is
  stated in the figure caption. Plotting one model's reasoning tokens against
  another's completion tokens would not be a comparison.
- **Cost** reported in an appendix table stamped with provider and quote date.

**On the dollar axis.** Accuracy per token is a property of a model. Accuracy
per dollar is a property of whoever is serving it this month; open-weights
pricing moves with promotions and with the hosting market. We therefore lead
with tokens and treat dollars as a dated snapshot. Readers should not expect
the cost figures to hold.

**Sample size** was set from measured per-call cost on a 20-question pilot
rather than from an estimate, and from the sample needed for a ±[TBD]
percentage-point confidence interval, whichever bound was tighter.

## 8. Limitations

1. One benchmark. Conclusions are about DriveLM MCQ, not driving reasoning
   generally.
2. The MCQ subset is not a random sample of DriveLM [distribution: TBD].
3. Train split only, because val ground truth is withheld. Models may have seen
   DriveLM train during pretraining. Our design cannot rule this out; the
   residual above-chance accuracy in `blind_notags` is consistent with either
   language priors or memorization, and we report it without attributing it.
4. Single thinking-effort setting per model. Each model appears as one
   operating point, not a curve.
5. Open-weights models are evaluated as a specific provider serves them,
   including quantization. Numbers may not transfer to other deployments.
6. Offline QA only. Nothing here speaks to closed-loop behaviour.
7. Single-seed [or: n-seed, TBD] collection; see §4 on determinism.

## 9. Reproducibility

Harness, prompt templates with version hashes, analysis code, and the full
response cache are released. Every record carries the prompt hash, decode
parameters, provider, served model string, and harness commit, so any reported
number can be traced to the exact configuration that produced it. No benchmark
data or model weights are redistributed.

---

## References to pull

- Sima et al. DriveLM: Driving with Graph Visual Question Answering. ECCV 2024.
- Caesar et al. nuScenes: A multimodal dataset for autonomous driving. CVPR 2020.
- Gururangan et al. Annotation Artifacts in Natural Language Inference Data. NAACL 2018.
- Hendrycks & Dietterich. Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations. ICLR 2019.
- Thinking Machines Lab. Inkling model card, 2026.
- AutoDrive-QA. arXiv:2503.15778. [cite for taxonomy only]
- [TBD] the 2026 text-bias-in-synthetic-driving-MCQ paper — find the exact citation.
