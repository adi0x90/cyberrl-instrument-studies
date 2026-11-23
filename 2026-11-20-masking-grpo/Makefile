# Invalid-action masking in GRPO (2026-11-20) — test, dry-run, train, eval, report.
#
#   make test      offline; pytest (stdlib-only maths, no model, no GPU, no network)
#   make dryrun    offline; three arms + three evals + REPORT.md under results/dryrun/
#   make train-a   real control arm (LoRA on Qwen2.5-Coder-1.5B-Instruct) -> results/
#   make train-b   real masked arm                                        -> results/
#   make train-c   real penalty arm (docs/SPEC.md Amendment v2)           -> results/
#   make eval      held-out eval of a saved adapter; ARM=control|masked|penalty
#   make report    recompute results/REPORT.md from the real artefacts
#
# Interpreters are probed by capability, not by a hardcoded path, so the same
# Makefile works on the laptop (.venv) and on the GPU pod (/workspace/venv). It
# prints which interpreter it picked; override with TEST_PY=... / TRAIN_PY=... / PY=...
#
# Budget (docs/SPEC.md): 24 effective updates, hard cap 96 groups sampled per arm
# (the v1 protocol deviation / Amendment v2 uses 8 effective and a 40-group cap).
# Override the budget on the command line, never in the file, if a protocol
# deviation is intended:  make train-a EFFECTIVE_UPDATES=8 MAX_GROUPS=32

RES             ?= results
DRYRUN_RES      ?= $(RES)/dryrun

EFFECTIVE_UPDATES ?= 24
MAX_GROUPS        ?= 96
PROMPTS_PER_UPDATE ?= 4
G                 ?= 8
SEED              ?= 20261120
MAX_MINUTES       ?= 90
HOURLY_RATE       ?= 0
LOCAL_ONLY        ?=
ARM               ?= control

MODEL    ?= Qwen/Qwen2.5-Coder-1.5B-Instruct
REVISION ?= $(MASKING_GRPO_MODEL_REVISION)
REV_FLAG := $(if $(strip $(REVISION)),--model-revision $(REVISION),)
LOCAL_FLAG := $(if $(strip $(LOCAL_ONLY)),--local-files-only,)

# Ordered: the pod's venv first, then the local experiment/.venv, then the vault
# .venv this harness was developed against, then whatever python3 is on PATH.
PY_CANDIDATES := /workspace/venv/bin/python .venv/bin/python \
	../../../../../.venv/bin/python python3 python3.12 python3.11 python3.10 \
	$(HOME)/.venv/bin/python \
	$(HOME)/miniforge3/bin/python3 $(HOME)/miniconda3/bin/python3 $(HOME)/anaconda3/bin/python3

# The GPU interpreter must be able to import the training stack.
TRAIN_PY ?= $(shell for c in $(PY_CANDIDATES); do \
	if { [ -x "$$c" ] || command -v "$$c" >/dev/null 2>&1; } \
	   && "$$c" -c 'import torch, transformers, peft' >/dev/null 2>&1; then echo "$$c"; break; fi; \
	done)

# Tests, dry runs and the report need pytest only.
TEST_PY ?= $(shell for c in $(PY_CANDIDATES); do \
	if { [ -x "$$c" ] || command -v "$$c" >/dev/null 2>&1; } \
	   && "$$c" -c 'import pytest' >/dev/null 2>&1; then echo "$$c"; break; fi; \
	done)

PY ?= $(if $(strip $(TEST_PY)),$(TEST_PY),python3)

TRAIN_FLAGS := --arm $(ARM) --prompts-per-update $(PROMPTS_PER_UPDATE) --g $(G) \
	--seed $(SEED) --max-minutes $(MAX_MINUTES) --model $(MODEL) $(REV_FLAG) \
	--hourly-rate $(HOURLY_RATE) $(LOCAL_FLAG)

.DEFAULT_GOAL := test
.PHONY: test dryrun train-a train-b train-c eval report clean-dryrun

$(RES):
	@mkdir -p $(RES)

$(DRYRUN_RES):
	@mkdir -p $(DRYRUN_RES)

test:
	@if [ -z "$(strip $(TEST_PY))" ]; then \
		echo "error: no python with pytest found (probed .venv, /workspace/venv, python3, python3.12/11/10)" >&2; \
		echo "hint: python3 -m pip install -r requirements.txt" >&2; \
		exit 2; \
	fi
	@echo "test interpreter: $(TEST_PY)"
	$(TEST_PY) -m pytest -q tests

# Offline end-to-end: three arms, three held-out evals, the three-arm report.
# No model, no GPU. Amendment v2 fixes the evaluation at 16 decodings per task.
dryrun: | $(DRYRUN_RES)
	@echo "dry-run interpreter: $(PY)"
	$(PY) train.py --arm control --dry-run --effective-updates 8 --max-groups 40 \
		--out $(DRYRUN_RES)/rollouts_control.jsonl --metrics $(DRYRUN_RES)/metrics_control.json
	$(PY) train.py --arm masked --dry-run --effective-updates 8 --max-groups 40 \
		--out $(DRYRUN_RES)/rollouts_masked.jsonl --metrics $(DRYRUN_RES)/metrics_masked.json
	$(PY) train.py --arm penalty --dry-run --effective-updates 8 --max-groups 40 \
		--out $(DRYRUN_RES)/rollouts_penalty.jsonl --metrics $(DRYRUN_RES)/metrics_penalty.json
	$(PY) evaluate.py --arm control --dry-run --decodings 16 --out $(DRYRUN_RES)/eval_control.json
	$(PY) evaluate.py --arm masked --dry-run --decodings 16 --out $(DRYRUN_RES)/eval_masked.json
	$(PY) evaluate.py --arm penalty --dry-run --decodings 16 --out $(DRYRUN_RES)/eval_penalty.json
	$(PY) analysis/summarize.py --dry-run
	@echo "dry-run artefacts: $(DRYRUN_RES)/ (NOT evidence — regenerate from a GPU run before reading)"

train-a: | $(RES)
	@if [ -z "$(strip $(TRAIN_PY))" ]; then \
		echo "error: no interpreter with torch+transformers+peft found" >&2; \
		echo "hint: /workspace/venv/bin/python -m pip install -r requirements.txt" >&2; \
		exit 2; \
	fi
	@echo "train interpreter: $(TRAIN_PY)"
	$(TRAIN_PY) train.py $(TRAIN_FLAGS) --effective-updates $(EFFECTIVE_UPDATES) \
		--max-groups $(MAX_GROUPS) \
		--out $(RES)/rollouts_control.jsonl --metrics $(RES)/metrics_control.json \
		--adapter-out $(RES)/adapter_control

train-c: | $(RES)
	@if [ -z "$(strip $(TRAIN_PY))" ]; then \
		echo "error: no interpreter with torch+transformers+peft found" >&2; \
		echo "hint: /workspace/venv/bin/python -m pip install -r requirements.txt" >&2; \
		exit 2; \
	fi
	@echo "train interpreter: $(TRAIN_PY)"
	$(TRAIN_PY) train.py --arm penalty --prompts-per-update $(PROMPTS_PER_UPDATE) --g $(G) \
		--seed $(SEED) --max-minutes $(MAX_MINUTES) --model $(MODEL) $(REV_FLAG) \
		--hourly-rate $(HOURLY_RATE) $(LOCAL_FLAG) \
		--effective-updates $(EFFECTIVE_UPDATES) --max-groups $(MAX_GROUPS) \
		--out $(RES)/rollouts_penalty.jsonl --metrics $(RES)/metrics_penalty.json \
		--adapter-out $(RES)/adapter_penalty

train-b: | $(RES)
	@if [ -z "$(strip $(TRAIN_PY))" ]; then \
		echo "error: no interpreter with torch+transformers+peft found" >&2; \
		echo "hint: /workspace/venv/bin/python -m pip install -r requirements.txt" >&2; \
		exit 2; \
	fi
	@echo "train interpreter: $(TRAIN_PY)"
	$(TRAIN_PY) train.py --arm masked --prompts-per-update $(PROMPTS_PER_UPDATE) --g $(G) \
		--seed $(SEED) --max-minutes $(MAX_MINUTES) --model $(MODEL) $(REV_FLAG) \
		--hourly-rate $(HOURLY_RATE) $(LOCAL_FLAG) \
		--effective-updates $(EFFECTIVE_UPDATES) --max-groups $(MAX_GROUPS) \
		--out $(RES)/rollouts_masked.jsonl --metrics $(RES)/metrics_masked.json \
		--adapter-out $(RES)/adapter_masked

eval: | $(RES)
	@if [ -z "$(strip $(TRAIN_PY))" ]; then \
		echo "error: no interpreter with torch+transformers+peft found" >&2; exit 2; \
	fi
	@if [ ! -d "$(RES)/adapter_$(ARM)" ]; then \
		echo "error: $(RES)/adapter_$(ARM) not found; run make train-a (control), train-b (masked) or train-c (penalty) first" >&2; \
		exit 2; \
	fi
	$(TRAIN_PY) evaluate.py --arm $(ARM) --adapter $(RES)/adapter_$(ARM) \
		--model $(MODEL) $(REV_FLAG) $(LOCAL_FLAG) --out $(RES)/eval_$(ARM).json

report:
	$(PY) analysis/summarize.py --dir $(RES) --out $(RES)/REPORT.md

clean-dryrun:
	rm -rf $(DRYRUN_RES)
