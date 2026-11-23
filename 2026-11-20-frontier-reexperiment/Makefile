# Frontier re-experiment (2026-11-20) — test, dry-run, real-run and report targets.
#
#   make test      offline; runs the pytest suite (statistics + mock dry runs)
#   make dryrun    offline; mock artifacts + a mock report under results/mock/
#   make e1        real E1 run; needs OPENROUTER_API_KEY in the environment
#   make e2        local grading pass; needs no key, imports results/e1.jsonl
#   make e3        real E3 run; needs OPENROUTER_API_KEY in the environment
#   make report    recompute results/REPORT.md from the real JSONL files
#
# Overrides: MODELS=... LIMIT=N SEED=N RES=dir RUN_PY=... TEST_PY=...

RES      ?= results
MOCK_RES ?= $(RES)/mock
MODELS   ?= tencent/hy4-flash,z-ai/glm-5.3-flash,deepseek/deepseek-v4.1-flash-1017
SEED     ?= 20261120
LIMIT    ?=
LIMIT_FLAG := $(if $(strip $(LIMIT)),--limit $(LIMIT),)

E1_OUT := $(RES)/e1.jsonl
E2_OUT := $(RES)/e2.jsonl
E3_OUT := $(RES)/e3.jsonl

# Pick interpreters by capability, not by a single hardcoded path, so the same
# Makefile works on this laptop and on the VPS. The conda locations are last-resort
# candidates; a provisioned .venv/bin/python wins on the VPS. Nothing here needs
# the network.
PY_CANDIDATES := .venv/bin/python python3 python3.12 python3.11 python3.10 \
	$(HOME)/miniforge3/bin/python3 $(HOME)/miniconda3/bin/python3 $(HOME)/anaconda3/bin/python3

RUN_PY ?= $(shell for c in $(PY_CANDIDATES); do \
	if { [ -x "$$c" ] || command -v "$$c" >/dev/null 2>&1; } \
	   && "$$c" -c 'import harness.models' >/dev/null 2>&1; then echo "$$c"; break; fi; \
	done)
ifeq ($(strip $(RUN_PY)),)
RUN_PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else command -v python3 || echo python3; fi)
endif

TEST_PY ?= $(shell for c in $(PY_CANDIDATES); do \
	if { [ -x "$$c" ] || command -v "$$c" >/dev/null 2>&1; } \
	   && "$$c" -c 'import pytest' >/dev/null 2>&1; then echo "$$c"; break; fi; \
	done)

KEY_HINT := e1/e3 send real requests: export OPENROUTER_API_KEY (from the environment only, never a file in this repo)

.DEFAULT_GOAL := test
.PHONY: test dryrun e1 e2 e3 report clean-mock

$(RES):
	@mkdir -p $(RES)

$(MOCK_RES):
	@mkdir -p $(MOCK_RES)

test:
	@if [ -z "$(strip $(TEST_PY))" ]; then \
		echo "error: no python with pytest found (probed .venv/bin/python, python3, python3.12/11/10)" >&2; \
		echo "hint: python3 -m pip install -r requirements.txt   # needs pytest and httpx" >&2; \
		exit 2; \
	fi
	@echo "test interpreter: $(TEST_PY)"
	$(TEST_PY) -m pytest -q tests

dryrun: | $(MOCK_RES)
	@echo "runner interpreter: $(RUN_PY)"
	$(RUN_PY) run_e1.py --mock $(LIMIT_FLAG) --out $(MOCK_RES)/e1.jsonl
	$(RUN_PY) run_e2.py --mock $(LIMIT_FLAG) --e1 $(MOCK_RES)/e1.jsonl --out $(MOCK_RES)/e2.jsonl
	$(RUN_PY) run_e3.py --mock $(LIMIT_FLAG) --out $(MOCK_RES)/e3.jsonl
	$(RUN_PY) analysis/summarize.py --mock \
		--e1 $(MOCK_RES)/e1.jsonl --e2 $(MOCK_RES)/e2.jsonl --e3 $(MOCK_RES)/e3.jsonl \
		--out $(MOCK_RES)/REPORT.md
	@echo "mock artifacts: $(MOCK_RES)/ (not evidence — regenerate before reading)"

e1: | $(RES)
	@if [ -z "$${OPENROUTER_API_KEY+x}" ]; then echo "$(KEY_HINT)" >&2; exit 2; fi
	$(RUN_PY) run_e1.py --resume $(LIMIT_FLAG) --models $(MODELS) --seed $(SEED) --out $(E1_OUT)

e2: | $(RES)
	$(RUN_PY) run_e2.py --resume $(LIMIT_FLAG) --e1 $(E1_OUT) --seed $(SEED) --out $(E2_OUT)

e3: | $(RES)
	@if [ -z "$${OPENROUTER_API_KEY+x}" ]; then echo "$(KEY_HINT)" >&2; exit 2; fi
	$(RUN_PY) run_e3.py --resume $(LIMIT_FLAG) --models $(MODELS) --seed $(SEED) --out $(E3_OUT)

report:
	$(RUN_PY) analysis/summarize.py --e1 $(E1_OUT) --e2 $(E2_OUT) --e3 $(E3_OUT) --out $(RES)/REPORT.md

clean-mock:
	rm -rf $(MOCK_RES)
