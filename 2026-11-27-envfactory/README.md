# 2026-11-27 — Cyber-RL Environment Factory PoC (Part 3 companion)

Executable companion to `part3-cyber-rl-environment-factory/02-ARCHITECTURE-AND-POC-SPEC.md` (v2).
The spec is frozen here as `docs/SPEC.md`; its SHA-256 is recorded in `docs/SPEC.sha256` and hashed
into every audit/stamp produced by `envcert`.

## What ran

| Track | Status | Evidence |
|---|---|---|
| **A1** expression control (12 tasks × 20 seeds × 3 arms) | complete | `evidence/a1/a1_verdicts.jsonl`, `results/REPORT.md` |
| **A1** native pack (authlib, held-out valid repair) | complete | `evidence/a1/a1_native.json` |
| **A1** leave-one-out alternates (3-fold) | complete | `evidence/a1/a1_loo.json` |
| **A1** per-candidate latency (n=10,000/task) | complete | `evidence/a1/a1_latency.json` |
| **A2** `envcert` two-level certification + D1–D3 defects | complete (5 packs) | `evidence/a2/a2_tests.json` |
| **A3** reset benchmark (redis, etcd) | complete | `evidence/a3/a3_reset.json` |
| **A3** reset benchmark (authlib, fastmcp, coreMQTT) | **not run** — see REPORT | `evidence/a3/a3_reset.json` `not_run` |
| **A3** snapshot restore (QEMU/KVM) | **unsupported** — no `/dev/kvm` on the host | `evidence/a3/a3_reset.json` `qemu` |
| **B1** selection drift (expression arm, 5 models × 32 pool) | complete — **negative**: 1 refutation / 112 T1 accepts | `results/B1.md` |
| **B1** native-pack arm (authlib + fastmcp, 3 models × 32 each) | complete — authlib **36/40 T1 accepts refuted**, fastmcp **0/66** | `results/B1_native.md` |
| **B2** matched re-audit 7B vs 14B | complete on the expression control — no difference detectable | `results/B1.md` |
| **B2** frontier adversarial stress (neutral vs reward-seeker) | **not run** | — |

## Reproduce

```bash
# A1 (local, CPU only; ~1 min for 20 seeds on an idle host)
python3 -m propverifier.run_a1 --out evidence/a1 --seeds 20

# A1 native pack (needs the authlib pack staged at /root/work/authlib; see scripts/)
python3 -m propverifier.run_a1_native

# A1 latency (serial, ~6 min for 12 tasks; do NOT run co-resident — contention doubles the delta)
python3 -m propverifier.latency --n 10000 --workers 1

# A2 (needs Docker for C3)
python3 scripts/run_a2_tests.py

# A3 (needs Docker + CRIU)
python3 -m microbench.run_a3

# B1/B2 (local Qwen pools need a GPU; the API pools run on CPU with OPENROUTER_API_KEY)
python3 -m bon.generate_vllm --model Qwen/Qwen2.5-Coder-7B-Instruct --out evidence/bon/bon_qwen7b.jsonl
python3 -m bon.generate_api --models "deepseek/deepseek-v4.1-flash-1017,tencent/hy4-flash,z-ai/glm-5.3-flash" --out evidence/bon/bon_api.jsonl
python3 -m bon.score_pool --pool evidence/bon/bon_qwen7b.jsonl --out evidence/bon/scored_qwen7b.jsonl
python3 -m bon.select_drift --scored evidence/bon/scored_*.jsonl --out results/B1.json
python3 -m bon.quarantine_sim
python3 -m bon.report_b1

# report + offline verification (no network, no services)
python3 analysis/build_report.py
python3 verify.py
```

## Layout

| Path | Role |
|---|---|
| `docs/SPEC.md` | frozen v2 spec (hash in `docs/SPEC.sha256`) |
| `propverifier/` | A1: `harness/` (published graders), `gen.py`, `fastrun.py`, `grade_property.py`, runners |
| `envcert/` | A2: `audit` / `stamp` / `verify` CLI, checks C1–C10 |
| `microbench/` | A3: state-equivalence gate, reset strategies, economics |
| `scripts/` | pack exporter (git-derived dual SHAs) and the D1–D3 defect tests |
| `evidence/a1/` | Track A1: verdicts, LOO, latency (serial + parallel control), native pack |
| `evidence/a2/` | Track A2: audits, stamps, defect-injection results |
| `evidence/a3/` | Track A3: reset benchmark + gate + economics |
| `evidence/bon/` | Track B: raw pools and scored verdicts |
| `evidence/packs/` | byte-exact snapshots of the five native packs + provenance |
| `evidence/sources/` | 28 raw fetched documents behind ATT-2966 |
| `analysis/` | report builder and shared recomputation |
| `results/` | generated `REPORT.md` + `claims.json` |
| `verify.py` | independent recomputation of every claim |

## Provenance rules applied

- **Dual-commit pinning.** Each pack exports `source_injection_sha` (agent-visible) and
  `metadata_commit_sha` (trusted auditor only). The metadata tree is never in the untrusted snapshot;
  C2 fails if it appears.
- **No pack branch is written.** Packs are read out of `/Users/adi/targets/*` with `git show` only.
- **Nothing here is a result that was typed by hand.** `results/REPORT.md` is generated;
  `verify.py` re-derives every claim from `evidence/` with an independent implementation.

## Known limits

- The A1 expression corpus is 12 hand-written tasks; the native case is one repository.
- Property generators were written knowing the task *contracts*; the blind leave-one-out alternate is
  the guard against generator tuning, not an independent author.
- A3 measures substrate reset cost on digest-pinned service images; the injected pack source is not
  inside those images.
