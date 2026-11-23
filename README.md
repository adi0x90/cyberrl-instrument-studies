# Cyber-RL Instrument Studies

Code, preregistered specifications, offline test suites (404 tests), and raw JSONL evidence for the two measurement-instrument studies reported in [*Keeping the Reward Honest in Cyber-RL Environments*](https://adityagupta.org/blog/keeping-the-reward-honest-in-cyber-rl-environments/).

These are not capability benchmarks. Both studies test whether an evaluation or training instrument can be trusted before trusting any model score on top of it.

## Verify every number in 5 seconds (offline, stdlib only)

```bash
python3 verify.py
```

`verify.py` reads the archived JSONL and JSON files directly and asserts every count, confidence interval, and exact McNemar $p$-value cited in the essay.

To run both full test suites and regenerate every markdown report from scratch:

```bash
(cd 2026-11-20-frontier-reexperiment && make test)   # 318 passed
(cd 2026-11-20-masking-grpo && make test)            # 86 passed
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

## Repository layout

```text
.
├── verify.py                           # recomputes all headline claims from JSONL (stdlib only)
├── LICENSE                             # MIT
├── 2026-11-20-frontier-reexperiment/   # E1, E2, E3 harness, 12 tasks, specs, 318 tests, evidence/
└── 2026-11-20-masking-grpo/            # 3-arm GRPO harness, specs, 86 tests, evidence-1.5b/, evidence-7b/
```

Keep both experiment directories as siblings: `2026-11-20-masking-grpo/rollout.py` imports `harness.*` and `tasks/` from `../2026-11-20-frontier-reexperiment/`.

## What is excluded (and why)

- **LoRA adapter checkpoints** (~360 MB across 1.5B and 7B runs) are omitted so the archive stays ~4.7 MB. Every rollout JSONL, metrics manifest, held-out eval JSON, and recomputed `REPORT.md` is included.
- **Scratch dry-run and smoke logs** are excluded so nothing non-evidential sits next to measured JSONL.
- **No API keys or host credentials** appear anywhere in the repo or git history; runners read `OPENROUTER_API_KEY` from the environment only.
