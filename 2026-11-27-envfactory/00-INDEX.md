# 00-INDEX — Cyber-RL Environment Factory PoC (2026-11-27)

Start here. This index maps every artifact in this experiment, what produced it, and how to
resume. Generated/curated by hand on 2026-11-27; the file inventory at the bottom is
machine-derived and refreshed by `find`/`b2sum`.

## What this is

Executable companion to `part3-cyber-rl-environment-factory/02-ARCHITECTURE-AND-POC-SPEC.md` v2
(frozen here as `docs/SPEC.md`, sha256 `b01ff5fa…eab440`). Four tracks:

| Track | Question | Status |
|---|---|---|
| A1 | Does a property-hardened Tier-1 kill finite-fixture shortcuts without rejecting unseen valid repairs? | **complete** |
| A2 | Can two-level certification (`pack_integrity` vs `policy_tested`) catch three injected defects with zero false FAILs? | **complete** |
| A3 | What does a state-equivalent reset cost, and is snapshot restore viable? | **partial** (2 of 5 targets; KVM absent) |
| B1/B2 | Does selection drift between Tier-1 and an independent Tier-2, and what does async audit cost the trainer? | **complete** — expression arm negative; native arm: authlib 90 % FAR, fastmcp 0 %; redis not run |

## Read in this order

1. `results/REPORT.md` — Track A, generated from evidence, every endpoint hit/miss/not-run.
2. `results/B1.md` — Track B expression arm, the negative result, quarantine accounting.
3. `results/B1_native.md` — Track B native-pack arm: authlib 36/40 refuted, fastmcp 0/66, with the
   verbatim diff + failing-variant output for every refutation.
3. `evidence/verification-ledger.md` — ATT-2966 primary-source verification (what was fetched,
   what changed, what could not be fetched).
5. `docs/FREEZE.json` — the spec hash every audit/stamp carries.

## Layout

```
2026-11-27-envfactory/
├── docs/            frozen spec + verification edits + freeze record
├── propverifier/    A1: harness (published, vendored), generators, oracle, runners
│   ├── harness/     12-task corpus support + the published graders
│   └── tasks/       12 hand-written security-guard tasks (byte-identical to the published corpus)
├── envcert/         A2: audit / stamp / verify CLI, checks C1–C10
├── microbench/      A3: state-equivalence gate, reset strategies, economics
├── bon/             B1/B2: pools, scoring, selection drift, quarantine, reports
├── analysis/        report builder + shared recomputation (Track A)
├── scripts/         pack exporter (git-derived dual SHAs), D1–D3 defect tests
├── evidence/        raw evidence, one subdirectory per track
├── results/         generated REPORT.md / B1.md + claims
├── tests/           vendored baseline grader tests
├── verify.py        independent offline recomputation of every Track A claim
└── Makefile         reproduce targets
```

## Evidence map

| Directory | Contents | Format |
|---|---|---|
| `evidence/a1/` | 5,700 verdict rows (20 seeds × 285 candidates); LOO; latency (serial, parallel control, single-task control); native-pack variants | JSONL + JSON |
| `evidence/a2/` | 5 clean-pack audits, D1/D2/D3 audits, stamps, deployment + observation fixtures | JSON |
| `evidence/a3/` | reset timings with raw samples, gate results, CRIU/QEMU probes, economics, `not_run` list | JSON |
| `evidence/bon/` | expression arm: 1,920 raw completions + 2,880 scored verdicts; native arm: 192 repair completions + 192 graded rows | JSONL |
| `evidence/packs/` | byte-exact snapshots of the five native packs, per-file git blob SHAs, `provenance.json` | bytes + JSON |
| `evidence/sources/` | 28 fetched documents (raw HTML retained) + prior-art search log | HTML + TXT |
| `evidence/logs/` | note only — RunPod run logs were on the pod volume, which was **terminated**; see the note | — |

`evidence/evidence.b2sum` covers every file under `evidence/` (167 entries) except itself.

## Provenance rules applied

- Packs are read from `/Users/adi/targets/*` with `git show` only; no pack branch was written.
- `source_injection_sha` (agent-visible) and `metadata_commit_sha` (auditor-only) are separated;
  `envcert` C2 fails if any metadata path appears in the untrusted snapshot.
- The harness and 12 tasks are byte-identical to the published replication package (sha256-checked).
- `results/REPORT.md` is generated; `verify.py` re-derives every claim with its own arithmetic
  (it does not import the report builder). Current status: **34/34**.

## Reproduce

```bash
make a1            # 20 seeds, ~1 min idle
make a1-latency    # serial, ~6 min; do not run co-resident
make a2            # needs Docker
make a3            # needs Docker + CRIU
make report verify # regenerate + independently check
```

Track B: local Qwen pools need a GPU (vLLM); the API pools run on any CPU box with `OPENROUTER_API_KEY`.

## Known gaps (deliberate, not silent)

- A3 covers redis + etcd only; fastmcp/authlib/coreMQTT are `not_run` with reasons.
- Track B native-pack arm covers authlib + fastmcp; **redis** is not run (its PoC needs an ASAN build and a
  live broker per candidate).
- Track B found **no measurable selection drift** on the 12-task corpus (1 refutation / 112 Tier-1
  accepts), but **90 % FAR on authlib** (36/40) and **0 % on fastmcp** — drift tracks how much of the
  flaw class the shipped PoC covers, not corpus difficulty.
- Pod run logs (`gen7b`, `genapi`, …) were lost when the pod was terminated; only progress, not
  outcomes. All substantive artifacts are local (`evidence/bon/`).

## File inventory

220 files, 15.2 MB. Types: 60 Python modules/tests, 5 generated Markdown reports, 12 task JSON,
9 evidence JSONL, ~40 evidence JSON, 5 pack source snapshots, 58 fetched source documents.

Per-directory counts: `docs` 4 · `propverifier` 7 (+6 harness, +12 tasks) · `envcert` 2 ·
`microbench` 3 · `bon` 9 · `analysis` 3 · `scripts` 3 · `tests` 2 · `evidence` 220 total across
subdirs · `results` 5.

Full paths are listed in the session hand-off; regenerate with:

```bash
find . -type f ! -path '*/__pycache__/*' | sort
```
