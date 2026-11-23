# 2026-11-20 frontier re-experiment

A measurement-instrument study of three failure modes that matter when a frontier model is
used as a reward source or a patch author for security-relevant code: an answer the harness
cannot **admit** (E1 — strict JSON vs fenced JSON vs unified diff), a patch that passes the
tests and **drops the property** (E2 — weak grader vs obligation grader), and a guard that is
well-formed, plausible and implements the **converse** of the policy (E3 — explicit vs neutral
prompt). Twelve hand-written tasks, three OpenRouter models, one serving path, one day. It is
deliberately *not* a capability benchmark: every arm measures whether an instrument can be
trusted, and the instruments are themselves under test.

Start with [`docs/DESIGN.md`](docs/DESIGN.md) before reading any number — it fixes the
endpoints, falsifiers and decision rules that were committed *before* the data. The interface
contract is [`docs/SPEC.md`](docs/SPEC.md) (normative), the run procedure is
[`docs/RUNBOOK.md`](docs/RUNBOOK.md), and the archived artifacts are indexed in
[`evidence/README.md`](evidence/README.md).

## Status

**Complete — run 20 November 2026.** All three arms finished: 396 model calls (E1 216, E3 180,
E2 none), every request HTTP 200, every manifest `status: complete`, `abort_reason: null`,
`mock: false`, and no config or task-hash drift within a shard. The ephemeral runner host
was torn down after the archive was copied back, so [`evidence/`](evidence/) is the primary
record and every number below is recomputed directly from those JSONL files.

## File map

```
2026-11-20-frontier-reexperiment/
  README.md                 # this page: orientation, results, reproduction
  docs/
    SPEC.md                 # normative interface contract (schemas, parsers, graders)
    DESIGN.md               # endpoint -> hypothesis -> falsifier, decision rules, limits
    RUNBOOK.md              # exact remote run procedure: rsync, key staging, shards, merge
  scripts/
    merge_shards.py         # merge per-model shards -> e1.jsonl / e3.jsonl + merged manifest
    reclassify_e3.py        # re-run the E3 polarity classifier over saved expressions
  harness/                  # models (OpenRouter + mock), envelopes, tasks_io, graders, polarity
  tasks/                    # 12 dependency-free task files (the corpus)
  mocks/                    # deterministic canned replies consumed by harness.models --mock
  tests/                    # pytest: statistics, schema, mock contract, resume idempotence
  analysis/summarize.py     # recomputes every table from the JSONL records
  run_e1.py  run_e2.py  run_e3.py
  Makefile  requirements.txt .gitignore
  evidence/                 # the archive (see evidence/README.md)
    README.md  REPORT.md
    e1.jsonl  e2.jsonl  e3.jsonl                      # AUTHORITATIVE merged evidence
    e1.manifest.json  e2.manifest.json  e3.manifest.json
    e1_{ds41,glm,hy4}.jsonl  e3_{ds41,glm,hy4}.jsonl  # per-model shards (provenance only)
    e1_*.manifest.json  e3_*.manifest.json
    smoke_e1.jsonl  smoke_e1.manifest.json            # one-task live smoke run, no denominator
    logs/                                             # runner stdout, one file per shard
  results/                  # gitignored runtime dir (make dryrun writes results/mock/)
```

`harness/`, `tasks/`, `mocks/`, `tests/`, `analysis/` and the three `run_*.py` files stay at
the experiment root on purpose: the runners and tests import `from harness...`, so moving them
breaks both the import path and the suite. Run every command in this README from this
directory.

`results/` is disposable: it is where `make dryrun` writes mock artifacts and where a re-run
writes shards. Nothing in `results/` is evidence.

## Headline results

Each is recomputed from the archived JSONL by `analysis/summarize.py` and printed in
[`evidence/REPORT.md`](evidence/REPORT.md); no number below was entered by hand.

| | result | exact numbers | pre-committed verdict |
|---|---|---|---|
| **E1** envelope cost | Admission falls as the output envelope gets harder — strict **88.9%** (64/72) vs toolcall **65.3%** (47/72), and the loss is admission, not understanding: of the 172 admitted patches, **172/172 (100%)** still hold the obligation | paired strict-vs-toolcall: 18 strict-only vs 1 toolcall-only, exact McNemar **p = 7.63e-05**; direction favourable in 3/3 models | **established** (rule: ≥2/3 models *and* pooled p < 0.05) |
| **E2** grader gap | The weak grader accepts and the obligation grader rejects **30/134 = 22.4%** of negative candidates (Wilson 95% **16.2%–30.2%**), while rejecting **0/48** genuinely correct fixes | weak-only HOLDS 30 vs obligation-only HOLDS 0 (McNemar p = 1.86e-09); Cohen's κ 0.782 | **not established at this n** (rule: ≥25% *and* Wilson lower bound above 0) — reported as a null |
| **E3** polarity | **0/180** converse guards, after the operator-boundary fix to the polarity classifier | converse rate 0.0% per (model, variant), Wilson 0.0%–11.4%; Fisher and paired McNemar both p = 1.0000; 22/180 `UNKNOWN` | **not established** (rule: ≥10%); the original 4/180 labels were a classifier artefact — see [`evidence/README.md`](evidence/README.md) |

The E1 "172/172" is the conditional reading, and it is the point: the envelope costs
*admissibility*, not correctness. An arm that only reported joint success would have made the
envelope look like a reasoning effect.

## Reproduce

Local, offline, no key — the plumbing and the statistics:

```bash
cd .../experiments/2026-11-20-frontier-reexperiment
make test        # pytest: statistics vs scipy-checked values, schema, mock dry-run, resume idempotence
make dryrun      # mock artifacts + mock report under results/mock/  (NOT evidence)
```

Rebuild the report from the archive:

```bash
python analysis/summarize.py \
  --e1 evidence/e1.jsonl --e2 evidence/e2.jsonl --e3 evidence/e3.jsonl \
  --out evidence/REPORT.md
```

(`make report` does the same thing but reads `results/e1,e2,e3.jsonl` — the runtime directory —
so use the explicit command to regenerate the archived report.)

Check that the archive is what it claims to be:

```bash
python scripts/merge_shards.py --exp e1 --results-dir evidence --out /tmp/e1.jsonl
cmp /tmp/e1.jsonl evidence/e1.jsonl                      # identical

python scripts/merge_shards.py --exp e3 --results-dir evidence --out /tmp/e3.jsonl
python scripts/reclassify_e3.py --evidence /tmp/e3.jsonl  # 4 CONVERSE -> UNKNOWN, as recorded
cmp /tmp/e3.jsonl evidence/e3.jsonl                      # identical

python scripts/reclassify_e3.py --evidence evidence/e3.jsonl --dry-run   # 0 changes
```

Re-run the experiment on a fresh VPS (real models, real calls): [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
It covers the rsync over SSH (`-e "ssh -i ~/.ssh/<vps-key>"`), staging the key at
`/root/.openrouter_key` (mode 600) and exporting it through `/root/cyberrl/env.sh`, the
per-model E1/E3 shards, the merge, E2, the report, and instance teardown.

> **The OpenRouter API key must never be written into this repository.** The runners read
> `OPENROUTER_API_KEY` from the environment only and never log it; `make e1`/`make e3` check
> only that the variable is set. If a key ever lands in a tracked file, rotate it.

## Conventions worth knowing

- **Interpreters.** The Makefile probes for one that can `import harness.models` (runners) and
  one that can `import pytest` (tests), preferring `.venv/bin/python`, and prints which it
  picked. Override with `make test TEST_PY=...` / `make dryrun RUN_PY=...`. Requirements:
  Python 3.10, `pytest`, `httpx`. `scipy` is not required by the report — the statistics are
  implemented in `analysis/summarize.py` and cross-checked against scipy at build time.
- **Mock artifacts are not evidence.** `--mock` replies are keyed on `sha256(model, messages)`,
  so they do not vary with temperature or seed and say nothing about any model. They exist to
  prove the plumbing.
- **One JSONL record per attempted unit**, failures included, with `raw_reply` kept so an
  admission decision can be re-litigated without another call. A sibling manifest is written
  at start (`status: running`) and at completion.
- **`--resume` is on by default** for the long targets: a re-run skips units already present
  and a killed process costs at most one torn trailing line.
- **Each runner is self-contained.** `run_e1.py`, `run_e2.py`, and `run_e3.py` inline their
  CLI, resume, and manifest scaffolding instead of importing a shared helper;
  `tests/test_runners_dryrun.py` asserts all three expose the same contract.
- **Do not edit `evidence/`.** It is measured data. The one post-run correction (four E3
  labels) is recorded in the records themselves and reproducible with
  `scripts/reclassify_e3.py`; see [`evidence/README.md`](evidence/README.md).

## Documentation

| file | read it when |
|---|---|
| [`docs/SPEC.md`](docs/SPEC.md) | changing a harness signature, a task schema field, or a runner CLI |
| [`docs/DESIGN.md`](docs/DESIGN.md) | reading any number: endpoints, falsifiers, stopping rules, threats to validity |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | running the experiment again, or auditing how it was run |
| [`evidence/README.md`](evidence/README.md) | finding which file a number comes from, or regenerating the report |
| [`evidence/REPORT.md`](evidence/REPORT.md) | the full tables, CIs, p-values, audit log and provenance |
