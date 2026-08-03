# Public result rows

This directory contains sanitized, question-level evaluation results intended
for public analysis. The raw provider caches and richer scored rows remain
gitignored under `results/`.

`inkling-through-2026-08-03.jsonl` contains 1,438 rows: 600 `blind_tags`, 600
`blind_notags`, and a budget-stopped 238-row `clean` prefix. The clean prefix is
not task-balanced and must be treated as interim evidence, not the final
confirmatory grounding analysis. See `docs/results-log.md` and
`study/runs/2026-08-03-inkling-credit-extension.json` for the analysis and
limitations.

Each JSON line uses schema `idq.public-result`, version 1. Rows include opaque
question IDs, scored outcomes, token/cost/latency measurements, model and decode
configuration, and reproducibility provenance. A fixed allowlist excludes
prompts, source annotations, response and reasoning text, images and local
paths, scene and frame IDs, raw provider payloads, cache keys, and credentials.

Rebuild the artifact offline with:

```bash
python -m idq.cli export-public \
  --scored results/inkling-blind-scored.jsonl \
  --scored results/inkling-clean-scored.jsonl \
  --out study/public-results/inkling-through-2026-08-03.jsonl
```

Expected SHA-256:
`541411e301386344e1a4777856cffcf754f3f1c86bfde1bd718c499bce0afdbb`.
