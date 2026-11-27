# 2026-11-27 environment-factory PoC
PY ?= python3

a1:
	$(PY) -m propverifier.run_a1 --out evidence/a1 --seeds 20

a1-native:
	$(PY) -m propverifier.run_a1_native

a1-latency:
	$(PY) -m propverifier.latency --n 10000

a2:
	$(PY) scripts/run_a2_tests.py

a3:
	$(PY) -m microbench.run_a3

report:
	$(PY) analysis/build_report.py

verify:
	$(PY) verify.py

test:
	$(PY) -m pytest -q tests

packs:
	$(PY) scripts/export_packs.py

all: a1 a1-native a2 a3 report verify
