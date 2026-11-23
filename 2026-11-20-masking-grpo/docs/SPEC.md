# SPEC — invalid-action masking in group-relative policy gradient

Preregistration. Frozen before any training run. Changing anything after seeing data
is a protocol deviation and must be reported.

## Question

In a sparse-reward cyber-repair task where a strict action envelope rejects a
substantial fraction of rollouts, does **excluding invalid actions from the group
baseline** (masking) increase the number of effective (nonzero-advantage) updates and
improve held-out repair success, relative to scoring invalid actions as reward 0 in the
group statistics?

Motivation, stated precisely because the naive version is wrong. A rejected action is
not a wrong answer, but GRPO cannot tell them apart: invalid samples receive reward 0
and therefore a negative advantage whenever anything in the group succeeds. That negative
pressure is informative about *format*, not about repair. Masking removes it. But masking
also removes the contrast that makes an admitted sample's advantage positive, so if all
valid samples in a group are identical (all HOLDS or all VIOLATED) the masked group has
zero variance and yields no signal. The two treatments therefore trade format pressure
against reward contrast, and the direction of the effect is an empirical question.

## Arms

| Arm | Invalid action handling | Group statistics |
|---|---|---|
| **A — control** | reward 0.0, kept in the group | mean and std over all G samples |
| **B — masked** | recorded, removed from the group | mean and std over admitted samples only; if fewer than 2 admitted, the group is skipped and logged |

Both arms: identical model, prompts, tasks, decoding temperature, rollout budget and
optimizer. The only difference is the baseline computation.

## Model and training

- Policy: `Qwen/Qwen2.5-Coder-1.5B-Instruct` at a pinned revision, LoRA rank 16 on q/v,
  float16/bfloat16, batching by group.
- Objective: single-step group-relative policy gradient (GRPO-style without PPO
  clipping). Each batch of prompts produces one gradient step, so the importance ratio
  is 1 and clipping is inert; this is stated so no one reads it as PPO.
- G = 8 completions per prompt, temperature 1.0, max 96 new tokens, top_p 0.95.
- Optimizer: AdamW, lr 1e-5, 1 step per rollout batch, no KL term (stated as a limit).
- Tasks: the 12 corpus tasks in `../../../experiments/2026-11-20-frontier-reexperiment/tasks/`.
  **Train on the 8 lexicographically first; hold out the last 4.** Prompts use the
  strict JSON envelope only.

## Reward and admission

- Admission: `harness.envelopes.admit_strict` on the raw completion.
- Reward: 1.0 if admitted and `harness.graders.grade_obligation` returns HOLDS, else 0.0.
  Invalid actions get 0.0 (arm A) or are masked (arm B). No partial credit.
- Every rollout logs: task, group index, completion, admitted, verdict, reward, tokens.

## Endpoints

**Co-primary:**
1. **Held-out joint success** — 4 held-out tasks × 8 decodings = 32 draws, Wilson 95% CI.
2. **Held-out admission rate** — does the arm learn the action format.

**Secondary (mechanism and cost):**
1. Train joint success across the run.
2. **Rollouts consumed per effective update** — the cost of the treatment.
3. Mean group reward variance and mean |advantage| of admitted samples.
4. Groups sampled, groups skipped, groups with zero variance.

**Budget parity is on gradient steps, not rollouts.** Both arms collect **24 effective
updates** (a group whose treatment yields a nonzero-variance baseline), with a hard cap
of 96 groups sampled per arm. If an arm reaches the cap first, it stops and the shortfall
is reported; rollouts consumed are a reported cost, not a hidden confound.

## Decision rule (pre-committed)

- **Masked better** only if held-out joint success exceeds control by ≥ 12 percentage
  points with non-overlapping Wilson intervals.
- **Control better** under the same rule in the other direction.
- **No measurable difference** otherwise, reported with both CIs and the rollout-cost
  ratio. A cost difference with no outcome difference is itself reportable.


## Stopping rules

- 24 effective updates per arm, cap 96 groups sampled per arm. No early stopping on
  outcome.
- Abort an arm (and report) if > 10 consecutive update exceptions, OOM twice, or wall
  time exceeds 90 minutes.
- GPU budget cap: $3.00 total; log spend from the pod's hourly rate.

## Analysis

- Headline table: arm × (effective updates/total, mean group variance, admission rate,
  train success, held-out success with CI).
- Paired per-update comparison of group variance (same prompts in both arms).
- Report every rollout count, including skipped groups in arm B, with denominators.
- No stratified selection, no best-checkpoint selection; final update only for held-out.

## Threats (stated before the run)

1. **n = 8 training tasks / 4 held-out.** This is a mechanism test, not a capability
   claim. A null on held-out success does not refute the mechanism. At 32 held-out draws
   the design can only detect a large effect.
2. **Single seed per arm.** The paired per-update variance comparison is within-run; it
   is not a seed-level estimate.
3. **No PPO clipping, no KL.** The experiment isolates the baseline treatment; it does
   not reproduce a full GRPO recipe.
4. **1.5B on 8 tasks may floor.** If both arms are at 0, the mechanism endpoint still
   carries the result and the outcome is reported as unmeasurable.
5. **Format learning confound.** A model may learn the JSON envelope rather than the
   repair. Both arms face this equally; admission is reported separately from HOLDS.

## Artifacts

`results/rollouts_<arm>.jsonl`, `results/metrics_<arm>.json`, `results/REPORT.md`,
`docs/SPEC.md`, `docs/DESIGN.md`, and the git SHA of the harness. The API/model revision
and GPU type are recorded in the manifest.

## Protocol deviations recorded before the real run

Recorded before any real arm data (file mtime 22:44:59 IST; the identical spec_sha256 in both
manifests, 29c74fd4…, confirms the same text was in effect for both arms).
All changes are symmetric across arms.

1. **One-shot example added to the prompt.** The smoke run produced 0/8 admitted
   completions; the prompt now shows the exact reply shape (`{"expression": "..."}`)
   with an illustrative expression that is not any task's answer. Without it the model
   spends the group on prose and the experiment measures format compliance only.
2. **`max_new_tokens` 96 → 48.** Expressions are single lines; 96 tokens measurably
   doubled wall time without changing the action space.
3. **Target 24 → 12 → 8 effective updates, cap 96 → 60 → 40 groups.** Measured cost was ~1.5–2
   minutes per group on a local Apple M1 Max (MPS), so the original target implied ~3
   hours per arm. 12 effective updates keeps the mechanism comparison while fitting the
   run into one session. This reduces power; it is stated, not hidden.
4. **Hardware:** the run uses a local Apple M1 Max, not a cloud GPU. Correction recorded
   after the run: the loader resolves cuda-else-cpu and has no MPS branch, so the executed
   device was **CPU with float32** (the manifests record `device: cpu`, `dtype: float32`,
   `gpu: null`). This affects throughput only, not the treatment; both arms are identical.

---

# Amendment v2 — third arm, second seed, larger held-out (2026-11-21)

Recorded before any v2 data. Motivated by the v1 null (both arms 3/32 held-out) and by
the v1 mechanism signal (masked variance 8–0, p=0.0078) together with an admission
direction that favoured control.

## Amended design

| Arm | Invalid action | Group statistics |
|---|---|---|
| **A — control** (v1) | reward 0.0 | all G samples |
| **B — masked** (v1) | excluded | admitted samples only; skip if < 2 |
| **C — penalty** (new) | **reward −1.0** | all G samples |

Rationale for C: a format error is cheaper to fix than a wrong repair, so the reward
should rank it *below* a wrong-but-valid answer (0.0) rather than equal to it. C tests
whether explicit negative pressure on invalid actions outperforms both other arms.

## Seeds

- Seed 20261120: v1 data, arms A and B only.
- Seed 20261121: all three arms.
- Unequal seed counts are reported, not hidden. The pooled estimate for A and B uses both
  seeds; for C it uses the seeds actually run.

## Held-out evaluation

16 decodings × 4 held-out tasks = **64 draws per adapter**. All adapters, including the
v1 adapters, are re-evaluated at 16 decodings so the denominator is comparable.

## Decision rules for v2

- **Penalty better than control** only if held-out joint success exceeds control by
  ≥ 12 pp with non-overlapping Wilson intervals; the same rule applies to penalty vs
  masked and to masked vs control (unchanged).
- Pooled across seeds where two exist; per-seed results reported alongside.
- A cost ratio (rollouts per effective update) is reported for every pair.

## What v2 does not change

Model, tasks, envelope, reward for valid work, G, one step per effective group, target 8
effective updates, cap 40 groups, and the "no clipping, no KL" limitation.
