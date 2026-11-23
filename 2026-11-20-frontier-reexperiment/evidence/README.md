# evidence/ — the archived run artifacts

Everything the 2026-11-20 frontier re-experiment measured, plus the manifests that make each
number recomputable. The run is **complete** (20 November 2026); its VPS instance was
destroyed after the archive was copied back, so this directory is the primary record. The
report in `REPORT.md` is *derived* — it can be regenerated from the JSONL files, never the
other way round.

Preregistration: [`../docs/SPEC.md`](../docs/SPEC.md) (interfaces),
[`../docs/DESIGN.md`](../docs/DESIGN.md) (endpoints, falsifiers, decision rules).
Procedure: [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md).

## Read this first

1. **Only the merged files are authoritative**: `e1.jsonl`, `e2.jsonl`, `e3.jsonl`.
   `e1_*.jsonl` and `e3_*.jsonl` are **per-model shards** — the three separate runs that were
   concatenated into the merged file. They are retained as provenance and can rebuild it, but
   no report number is computed from a shard.
2. **These files are measured data.** Nothing here is edited to make a result nicer. The one
   post-run correction in the archive (four E3 `verdict` labels) is recorded in the record
   itself as `verdict_reclassifier_before`, and was recomputed from saved expressions with no
   new model calls — see "The one post-run correction" below.
3. **Mock artifacts are not evidence** and none are archived here: `results/mock/` is a local
   dry-run directory (`make dryrun`) and says nothing about any model.
4. Record counts: **E1 216, E2 279, E3 180** — 675 records in the three merged files.

## What each file is

| file | what it is |
|---|---|
| `e1.jsonl` | **E1 merged evidence — 216 records.** One record per attempted unit: 12 tasks × 3 envelopes (strict/fence/toolcall) × 3 models × 2 decodings. 172 admitted + obligation `HOLDS`; 44 never admitted (`UNKNOWN`). |
| `e1.manifest.json` | Merged E1 manifest: three SPEC model ids, seed 20261120, decoding table, per-task sha256, `shards`, `records_written: 216`, `mock: false`, `status: complete`. |
| `e1_ds41.jsonl` | E1 shard, `deepseek/deepseek-v4.1-flash-1017` — 72 records. Provenance only. |
| `e1_ds41.manifest.json` | Shard manifest for the above (model, `out`, task hashes, its own `utc`). |
| `e1_glm.jsonl` | E1 shard, `z-ai/glm-5.3-flash` — 72 records. Provenance only. |
| `e1_glm.manifest.json` | Shard manifest for the above. |
| `e1_hy4.jsonl` | E1 shard, `tencent/hy4-flash` — 72 records. Provenance only. |
| `e1_hy4.manifest.json` | Shard manifest for the above. |
| `e2.jsonl` | **E2 merged evidence — 279 records.** Local grading only, no model calls: 48 seeded-bad + 86 parseable mutations (27 `swap_op`, 11 `invert_membership`, 48 `weaken_guard`) + 12 reference + 36 alternate + 1 adversarial + 96 admitted E1 patches. |
| `e2.manifest.json` | E2 manifest: candidate kinds, `e1_patches_imported: 96`, `records_written: 279`, `mock: false`. |
| `e3.jsonl` | **E3 merged evidence — 180 records.** 10 polarity-sensitive tasks × 2 variants (explicit/neutral) × 3 models × 3 decodings. Verdicts: 158 `CORRECT`, 22 `UNKNOWN`, **0 `CONVERSE`**; 4 records carry `verdict_reclassifier_before` (see below). |
| `e3.manifest.json` | Merged E3 manifest: three model ids, seed, decoding table, `tasks_selected` / `tasks_skipped_not_polarity_sensitive`, `records_written: 180`. |
| `e3_ds41.jsonl` | E3 shard, `deepseek/deepseek-v4.1-flash-1017` — 60 records. Provenance only. |
| `e3_ds41.manifest.json` | Shard manifest for the above. |
| `e3_glm.jsonl` | E3 shard, `z-ai/glm-5.3-flash` — 60 records. Provenance only. |
| `e3_glm.manifest.json` | Shard manifest for the above. |
| `e3_hy4.jsonl` | E3 shard, `tencent/hy4-flash` — 60 records. Provenance only. |
| `e3_hy4.manifest.json` | Shard manifest for the above. |
| `smoke_e1.jsonl` | A one-task live smoke run (`--limit 1`, glm, 6 records) made before the sweep. **Not part of any denominator**; no report table reads it. |
| `smoke_e1.manifest.json` | Manifest for the smoke run (`limit: 1`, `mock: false`). |
| `REPORT.md` | The recomputed report: E1/E2/E3 endpoint tables, Wilson 95% CIs, exact McNemar, Fisher exact, Cohen's κ, limits, provenance. Derived — regenerate it, do not edit it. |
| `logs/e1_ds41.log` … `logs/e3_hy4.log` | Runner stdout, one file per shard (six files): progress lines and the completion line (`N written, 0 skipped -> results/<shard>.jsonl`). Human trace; contributes no number. |

## How to regenerate the report

From the experiment root, naming the archived files explicitly so the report's Provenance
table points at what a reader has:

```bash
python analysis/summarize.py \
  --e1 evidence/e1.jsonl --e2 evidence/e2.jsonl --e3 evidence/e3.jsonl \
  --out evidence/REPORT.md
```

The analyser implements Wilson, exact McNemar, Fisher exact and Cohen's κ in the standard
library, so this runs on a bare Python 3.10 with no third-party packages and no network. A
missing or empty input still produces the full report shape with "no records" rows.

## Which numbers come from which file

| report section | computed from | what it recomputes |
|---|---|---|
| `## 1. E1` (1.a–1.d) | `e1.jsonl` | admission rate and joint success per (model, envelope) with Wilson CIs; pooled rows; joint success given admission; paired McNemar across envelopes; the strict ⊆ fence instrument check |
| `## 2. E2` (2.a–2.f) | `e2.jsonl` | weak false accepts per seeded-bad family (pooled 30/134 = 22.4%, Wilson 16.2–30.2%); obligation false rejects on reference/alternate controls (0/48); weak-vs-obligation McNemar and κ; task-family breakdown; per-family adversarial probes; the 96 imported E1 patches |
| `## 3. E3` (3.a–3.d) | `e3.jsonl` | converse-guard rate per (model, variant) — 0/180; Fisher exact explicit vs neutral; paired McNemar; the audit log of every returned expression |
| `## Provenance` | `e1.manifest.json`, `e2.manifest.json`, `e3.manifest.json` | git sha, `utc`, `mock`, seed, model list, task-file count, combined task hash, `argv` — identifies the run; contributes no measurement |
| `## What this does not show` | — | fixed prose in the analyser; no data input |

Not used by any number: the six shard JSONL files and their manifests, `smoke_e1.*`, and
`logs/`. The shard manifests are what let you *rebuild* the merged files ("Verifying the archive"); the logs are a
trace.

## Verifying the archive

The merged files are reproducible from the retained shards by command:

```bash
# E1: merge the shards into a scratch path and compare
python scripts/merge_shards.py --exp e1 --results-dir evidence --out /tmp/e1.jsonl
cmp /tmp/e1.jsonl evidence/e1.jsonl            # identical

# E3: merge, then apply the recorded reclassification, and compare
python scripts/merge_shards.py --exp e3 --results-dir evidence --out /tmp/e3.jsonl
python scripts/reclassify_e3.py --evidence /tmp/e3.jsonl
cmp /tmp/e3.jsonl evidence/e3.jsonl            # identical
```

The E3 shards still carry the four original `CONVERSE` labels; the reclassification pass is
what turns them into the archived `UNKNOWN`. That is why `evidence/e3.jsonl` differs from a
plain concatenation while `evidence/e1.jsonl` does not.

## The one post-run correction, in detail

`harness.polarity.converse_guard` classifies a guard's polarity from markers in the task's
`polarity` block. The first version matched the converse marker `>` inside the composite
operator `>=`, which labelled four `session.epoch >= state.epoch` guards as `CONVERSE`. The
classifier was fixed to require an operator boundary (regression-pinned in
`tests/test_polarity.py`), and the four records were then re-labelled offline from their saved
`raw_expression` — no model call, no re-run, no data invented:

- each changed record keeps its original verdict in `verdict_reclassifier_before`;
- the tool is [`../scripts/reclassify_e3.py`](../scripts/reclassify_e3.py) (add `--dry-run` to
  inspect without writing);
- post-fix distribution: `CORRECT` 158, `UNKNOWN` 22, `CONVERSE` 0 — the E3 headline is
  **0/180**, not the original 4/180;
- `REPORT.md` was regenerated afterwards, and running the tool on the archived evidence today
  changes **0** records.

A second, unrepaired provenance defect is disclosed in `REPORT.md` and
[`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) §9: `tasks/bounds_content_length_01.json` was
edited between E3 shards, so 18 E3 and 18 E1 units pool two versions of that task. The
per-shard manifests show it (differing `task_sha256`); the earlier bytes were not retained.
