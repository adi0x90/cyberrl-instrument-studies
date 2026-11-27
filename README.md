# Cyber-RL Instrument Studies

Code, preregistered specifications, offline test suites, and raw JSONL evidence for the measurement-instrument studies reported in the [*Cyber-RL Measurement*](https://adityagupta.org/blog/when-a-passing-patch-still-fails/) series:

- [Part 1 — *Keeping the Reward Honest in Cyber-RL Environments*](https://adityagupta.org/blog/keeping-the-reward-honest-in-cyber-rl-environments/)
- [Part 2 — *Five Cyber-RL Benchmarks, One OSS-Fuzz Corpus*](https://adityagupta.org/blog/five-cyber-rl-benchmarks-one-oss-fuzz-corpus/)
- [Part 3 — *When a Passing Patch Still Fails: Auditing Cyber-RL Verifiers on Real Repositories*](https://adityagupta.org/blog/when-a-passing-patch-still-fails/)

These are not capability benchmarks. Every study tests whether an evaluation or training instrument can be trusted before trusting any model score on top of it.

## Verify every number, offline (stdlib only)

```bash
python3 verify.py                                      # Parts 1-2: E1/E2/E3 + GRPO arms
(cd 2026-11-27-envfactory && python3 verify.py)        # Part 3: 49/49 claims
```

Each script reads the archived JSONL and JSON files directly and asserts every count, confidence interval, exact McNemar $p$-value, and false-accept rate cited in the corresponding essay. Neither needs a network connection or a third-party package.

To run the full test suites and regenerate the markdown reports from scratch:

```bash
(cd 2026-11-20-frontier-reexperiment && make test)   # 318 passed
(cd 2026-11-20-masking-grpo && make test)            # 86 passed
(cd 2026-11-27-envfactory && make a1 && make report) # property verifier + report rebuild
```

## Summary of archived results

### 1. Frontier instrument study (`2026-11-20-frontier-reexperiment/`)

12 hand-written single-expression Python security-guard tasks (authorization, bounds, canonicalization, containment, cookies, state drift) across three OpenRouter models (`deepseek/deepseek-v4.1-flash-1017`, `tencent/hy4-flash`, `z-ai/glm-5.3-flash`), 396 live HTTP-200 calls.

| Study | Question | Measured result | Pre-committed reading |
|---|---|---|---|
| **E1** | Output envelope cost (strict JSON vs. fenced JSON vs. unified diff) | Admission **88.9%** (64/72) vs. **84.7%** (61/72) vs. **65.3%** (47/72); 18 strict-only vs. 1 diff-only (McNemar $p = 7.63\times 10^{-5}$); **172/172 (100%)** admitted patches satisfy the obligation | **Established** (direction in 3/3 models, $p < 0.05$) |
| **E2** | Weak PoC grader vs. behavioral-and-AST obligation grader on 134 negatives + 48 positive controls | Weak false accepts **30/134 (22.4%, Wilson 16.2%–30.2%)** — feature deletion 12/24 (50.0%), symptom suppression 17/40 (42.5%); **0/48** false rejects; McNemar $p = 1.86\times 10^{-9}$, $\kappa = 0.782$ | **Not established at this $n$** (pre-committed threshold was $\ge 25\%$) |
| **E2.e** | Adversarial memorization probe (`bounds_content_length_01`) | `raw in {"4096", "1048576", ...}` passed **both** `grade_weak` and `grade_obligation` | Documented blind spot of finite fixture sets |
| **E3** | Converse-guard polarity (`!=` where `==` is required) | **0/180** converse guards after fixing an operator-boundary bug in `harness/polarity.py` that initially mislabeled 4/180 (`scripts/reclassify_e3.py`) | **Not established** (null; parser bug logged) |

### 2. Invalid-action handling in GRPO (`2026-11-20-masking-grpo/`)

Three-arm preregistered comparison on the same 12-task corpus (8 train, 4 held-out × 16 decodings = 64 held-out draws) using `Qwen2.5-Coder-1.5B-Instruct` (`evidence-1.5b/`) and `Qwen2.5-Coder-7B-Instruct` (`evidence-7b/`), two seeds (`20261120`, `20261121`).

- **Control:** invalid action receives `0.0`, kept in the all-$G$ baseline.
- **Masked:** invalid action removed from the baseline (groups with $<2$ admitted samples skipped).
- **Penalty:** invalid action receives `-1.0`, kept in the all-$G$ baseline.

Primary seed (`20261121`):

| Model | Arm | Effective updates | Rollouts / effective update | Held-out joint success [Wilson 95%] |
|---|---|---|---|---|
| 1.5B | control | 8/8 | 30.00 | 3/64 (4.7%) [1.6–12.9%] |
| 1.5B | masked | 8/8 | 30.00 | 3/64 (4.7%) [1.6–12.9%] |
| 1.5B | penalty | 8/8 | **8.00 (0.27×)** | 3/64 (4.7%) [1.6–12.9%] |
| 7B | control | 18/30 | 13.33 | 25/64 (39.1%) [28.1–51.3%] |
| 7B | masked | 12/30 | 20.00 (1.50×) | 26/64 (40.6%) [29.5–52.9%] |
| 7B | penalty | 28/30 | **8.57 (0.64×)** | 27/64 (42.2%) [30.9–54.4%] |

Held-out repair ended in **no measurable difference** under the pre-committed $\ge 12$ pp / non-overlapping Wilson rule. The rollout-efficiency split matches the pre-run proof in `tests/test_grpo.py`: for binary `{0, 1}` rewards, `masked-effective` implies `control-effective`, so masking can only lose effective gradient updates, whereas a `-1.0` penalty creates variance in mixed groups.

### 3. Environment factory: verification, certification, and selection drift (`2026-11-27-envfactory/`)

Four tracks over a frozen spec (`docs/SPEC.md`, sha256 `b01ff5fa…eab440`), recomputed by `verify.py` — **49/49 claims**.

| Track | Question | Measured result |
|---|---|---|
| **A1** | Does a property-hardened Tier-1 verifier kill finite-fixture shortcuts without rejecting valid repairs? | Kill rate **weak 67.2% → static 96.3% → property 100.0%** (3,780/3,780 across 20 seeds); **0/960** positive-control false rejects; +**2.48 ms** p50 serial per candidate call. Blind leave-one-out caught **1/36** — and contract adjudication showed *our own pre-authored valid alternate was buggy* (`canonicalize(...) is not None` dropped, so `None == None` accepted a percent-encoded path) |
| **A2** | Can two-level certification catch injected environment defects? | **5/5** clean packs reach `pack_integrity`; D1 (leaked metadata) → C2 FAIL, D2 (weak-verifier swap) → C4 FAIL, D3 (drifted deployment) → `verify` exit 2. **3/3 caught, 0 false FAILs** |
| **A3** | What does a state-equivalent reset cost, and is snapshot restore viable? | `redis` p50 **0.22–0.35 s**, `etcd` p50 **0.68–0.95 s**, 16-worker throughput 20.5k / 16.8k episodes per hour; dual-marker state-equivalence gate PASS on both; **CRIU restore verified on Redis** (`PONG`); KVM unavailable on the host; library targets (`authlib`, `coreMQTT`) `not_run` — no mutable service state |
| **B1** | Does Best-of-$N$ selection drift toward patches an independent Tier-2 refutes? | **12-task expression control:** 1 refutation / 112 Tier-1 accepts (0.9%); raw-trace adjudication of all 10 population refutations found **1 genuine bug (Qwen-7B dropped a delegated-scope check) vs 9 differential-oracle artifacts** (unconstrained generators penalising stricter or RFC-compliant patches) |
| **B1 native** | Same question on real repositories (Part 3's headline) | **`authlib-jwt-dupkey-canonicalization`: 36/40 Tier-1-accepted patches refuted (90.0%)**, 3/3 models refuted at $N \in \{4,8,16,32\}$ — 24 `scope`-only local fixes and 12 first-wins canonicalisations left duplicate-key confusion open on `aud` and `exp`; only 4/40 rejected duplicate JSON members. **`fastmcp-canonical-authz-alias`: 0/66 refuted (0.0%)**, 3/3 models clean at every $N$. Drift tracks *mechanism coverage* of the shipped PoC, not task difficulty. Cost: **$0.98** of API spend |
| **B2** | Does async Tier-2 latency matter more than sampling rate? | Under a 30 s step clock, the single refuted accept left **0 / 2 / 10** optimizer steps committed before ack at 0 / 60 / 300 s audit delay; the anomaly trigger fired **0** times and would have missed it; stale `policy_rev=1` acks were **ignored as stale in 3/3** arms |

All 192 native-pack completions with their graded verdicts, the 36 adjudication records (verbatim diff plus both failing variant outputs), and 28 fetched primary sources (58 files, raw HTML and text) are archived under `2026-11-27-envfactory/evidence/` and `2026-11-27-envfactory/results/`.

## Repository layout

```text
.
├── verify.py                           # recomputes all Parts 1-2 headline claims from JSONL (stdlib only)
├── LICENSE                             # MIT
├── 2026-11-20-frontier-reexperiment/   # E1, E2, E3 harness, 12 tasks, specs, 318 tests, evidence/
├── 2026-11-20-masking-grpo/            # 3-arm GRPO harness, specs, 86 tests, evidence-1.5b/, evidence-7b/
└── 2026-11-27-envfactory/              # Tier-1 property verifier, envcert CLI, reset microbench,
                                        # Best-of-N drift + quarantine simulation, 49/49 verify.py
```

Keep the experiment directories as siblings: `2026-11-20-masking-grpo/rollout.py` imports `harness.*` and `tasks/` from `../2026-11-20-frontier-reexperiment/`.

Within `2026-11-27-envfactory/`, `verify.py` recomputes Track A and Track B endpoints from `evidence/` independently of the report generator; `make report` regenerates `results/REPORT.md`, `results/B1.md`, and `results/B1_native.md`.

## What is excluded (and why)

- **LoRA adapter checkpoints** (~360 MB across 1.5B and 7B runs) are omitted so the archive stays small. Every rollout JSONL, metrics manifest, held-out eval JSON, and recomputed `REPORT.md` is included.
- **Scratch dry-run and smoke logs** are excluded so nothing non-evidential sits next to measured JSONL.
- **Native pack snapshots** (`2026-11-27-envfactory/evidence/packs/`) are withheld: the forward-injected pack estate is a contamination-controlled asset, and its injection specs, manifests, and held-out variants are the answer key for those tasks. `verify.py` recomputes **49/49** claims without them. See `2026-11-27-envfactory/PUBLICATION-NOTES.md`.
- **No API keys or host credentials** appear anywhere in the repo or git history; runners read `OPENROUTER_API_KEY` from the environment only.
