# Publication notes for `2026-11-27-envfactory/`

This directory is the executable companion to Part 3 of the *Cyber-RL Measurement* series
([*When a Passing Patch Still Fails*](https://adityagupta.org/blog/when-a-passing-patch-still-fails/)).
It is published so that every number in the essay can be recomputed offline:

```bash
python3 verify.py     # 60/60 claims, stdlib only, no network
```

## What is withheld, and why

**`evidence/packs/` is not published.** Those 70 files are byte-exact snapshots of five
forward-injected vulnerable forks (`etcd-revocation-skew`, `authlib-jwt-dupkey-canonicalization`,
`coremqtt-publish-varint-payloadlen`, `fastmcp-canonical-authz-alias`,
`redis-setrange-grow-undercount`), including each pack's injection spec, `manifest.json`,
`poc/`, and held-out `variants/`.

The pack estate is a contamination-controlled asset: the flaws are forward-injected and never
published, so that a policy training on them cannot have memorised the answer. The injection
specs and held-out variants *are* the answer key for those tasks, and the Part 3 protocol
deliberately withholds them from the model prompt (`bon/native_prompts.py` enforces a hard leak
check). Publishing them here would destroy the property the packs exist to provide.

The consequences for replication are bounded and stated plainly:

- `verify.py` recomputes **60/60** claims without `evidence/packs/`. The Track A1 native-pack
  verdicts and all Track B selection cells are stored in `evidence/a1/a1_native.json`,
  `evidence/bon/graded_*.jsonl`, and `results/B1_native.json`, which are published.
- `evidence/evidence.b2sum` is the manifest computed at run time over the full local tree
  (282 entries). Entries under `evidence/packs/` therefore name files that are intentionally
  absent from this repository; every other entry verifies.
- Re-running the native-pack arm from scratch requires the pack branches, which are not public.

## What is published

- **All code:** `propverifier/` (Tier-1 property verifier and runners), `envcert/` (C1–C10 audit,
  stamp, verify CLI), `microbench/` (state-equivalence gate, reset strategies, pool-priced
  economics), `bon/` (prompt builder with leak check, pool generation, grading, selection drift,
  quarantine simulation, reports), `analysis/`, `scripts/`, `tests/`.
- **The frozen spec:** `docs/SPEC.md` with `docs/SPEC.sha256` and `docs/FREEZE.json`.
- **Raw evidence:** `evidence/a1/` (5,700 verdict rows across 20 seeds + leave-one-out + latency),
  `evidence/a2/` (five clean-pack audits, D1–D3 defect audits, stamps, deployment and observation
  fixtures), `evidence/a3/` (reset timings with raw samples, gate results, CRIU/QEMU probes,
  economics), `evidence/bon/` (1,920 expression-control completions, 192 native-pack completions,
  scored verdicts for both), `evidence/sources/` (28 fetched primary sources), and
  `evidence/verification-ledger.md`.
- **Generated reports:** `results/REPORT.md`, `results/B1.md`, `results/B1_native.md`,
  `results/B1_native.json`, `results/claims.json`, and the 36 verbatim
  `results/adjudication/authlib/*.{diff,meta.json,variant.txt}` records quoted in the essay.

## Provenance rules applied during the run

- One pack branch was never written: pack content was read with `git show` only.
- `source_injection_sha` (visible to the agent) and `metadata_commit_sha` (visible only to the
  trusted audit worker) are pinned separately; `envcert` C2 fails if any metadata path appears in
  the untrusted snapshot.
- The 12-task expression corpus and its graders are byte-identical to the published Part 1
  replication package (sha256-checked).
- Apply failures, build failures, and runner errors are recorded as `untested` — never as a pass
  and never as a kill. The first `authlib` dry run produced six fake refutations from `cp -r`
  dropping an execute bit; the fix (mode-bit preservation plus run-error classification) is in the
  grader, not in the prose.
