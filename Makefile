# One command per thing you actually want to do.
#
# Every target that spends money prints what it will cost and asks first.
# Every target that doesn't spend money just runs.
#
#   make test          offline test suite
#   make key           pull BASETEN_API_KEY from launchd into this shell
#   make preflight     everything that must be true before a paid run
#   make clean-pilot   20 image calls, ~$0.09
#   make clean-600     full clean condition, ~$2.68
#   make blind-topup   fill the 42 missing blind_tags questions, ~$0.02
#   make score         score + analyze everything collected (free)
#   make report        regenerate paper tables (free)

SHELL := /bin/bash
PY := python -m idq.cli
DATA := data/drivelm/v1_1_train_nus.json
COMMON := --adapter drivelm_converted --data $(DATA) \
          --provider openai_compat --base-url https://inference.baseten.co/v1 \
          --api-key-env BASETEN_API_KEY \
          --model-string thinkingmachines/inkling \
          --model-label inkling --quantization unknown \
          --thinking-effort medium
PRICES := --usd-in 1.00 --usd-out 4.05 --price-date 2026-08-02
IMAGES := --image-root data/drivelm

.PHONY: test key preflight clean-pilot clean-600 blind-topup score report confirm

test:
	python -m pytest -q

# `make key` can't export into your shell — a child process cannot modify its
# parent's environment. It prints the line to run instead.
key:
	@echo 'Run this in your shell (it will not echo the key):'
	@echo '  export BASETEN_API_KEY="$$(launchctl getenv BASETEN_API_KEY)"'

preflight:
	@fail=0; \
	if [ -z "$$BASETEN_API_KEY" ]; then \
	  echo "✗ BASETEN_API_KEY not set in this shell — run 'make key'"; fail=1; \
	else echo "✓ API key present ($${#BASETEN_API_KEY} chars)"; fi; \
	if [ -n "$$(git status --porcelain)" ]; then \
	  echo "✗ working tree is dirty — collection refuses it, and git_sha would not reproduce the run"; fail=1; \
	else echo "✓ tree clean at $$(git rev-parse --short HEAD)"; fi; \
	if [ ! -f "$(DATA)" ]; then echo "✗ missing $(DATA)"; fail=1; \
	else echo "✓ DriveLM annotations present"; fi; \
	if [ ! -d data/nuscenes/samples/CAM_FRONT ]; then \
	  echo "✗ images missing — see docs/runbook.md step 3b"; fail=1; \
	else echo "✓ images present"; fi; \
	python -m pytest -q >/dev/null 2>&1 && echo "✓ tests pass" || { echo "✗ tests failing"; fail=1; }; \
	exit $$fail

# Ask before spending. COST is set by each paid target.
confirm:
	@read -p "About to spend ~$(COST). Continue? [y/N] " a; [ "$$a" = "y" ]

clean-pilot: COST=\$$0.09
clean-pilot: preflight confirm
	$(PY) pilot $(COMMON) $(PRICES) $(IMAGES) \
	  --condition clean --budget 1 --n 20 --n-repeat 0 \
	  --cache results/clean-pilot.jsonl

clean-600: COST=\$$2.68
clean-600: preflight confirm
	$(PY) collect $(COMMON) $(PRICES) $(IMAGES) \
	  --condition clean --cohort study/cohorts/drivelm-balanced-600.json \
	  --cache results/inkling-clean-cache.jsonl

# The blind_tags run stopped 42 short of the cohort. The cache makes this
# fill only the gaps.
blind-topup: COST=\$$0.02
blind-topup: preflight confirm
	$(PY) collect $(COMMON) $(PRICES) \
	  --condition blind_tags --cohort study/cohorts/drivelm-balanced-600.json \
	  --cache results/inkling-blind-cache.jsonl

score:
	$(PY) score --adapter drivelm_converted --data $(DATA) \
	  --cache results/inkling-blind-cache.jsonl --out results/inkling-blind-scored.jsonl

report:
	$(PY) analyze --scored results/inkling-blind-scored.jsonl \
	  --primary "inkling,qwen3-vl-235b-thinking" \
	  --markdown results/inkling-blind-tables.md \
	  --json-out results/inkling-blind-report.json \
	  --title "Inkling DriveLM converted"
