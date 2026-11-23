# DESIGN — invalid-action masking, why this shape

## The claim under test

In sparse-reward group-relative policy gradient, invalid actions are scored as reward 0.
That does two things at once: it removes them from the *signal* an admitted sample
provides (they depress the group mean, which raises the admitted sample's positive
advantage), and it gives them a *negative* advantage whenever anything in the group
succeeds, which pushes the policy away from the invalid behaviour.

Masking removes both effects. It is therefore not obvious that masking helps:

- Removing the negative pressure should *hurt* format learning (admission may fall).
- Removing the dilution should *raise* the advantage magnitude on admitted samples.
- If the admitted samples are all correct, masking leaves a zero-variance group and
  produces no signal at all.

The experiment measures the trade-off instead of assuming a direction.

## What is intentionally not claimed

- Not a GRPO reproduction: one gradient step per group, no PPO clipping, no KL term.
  The SPEC says so; the report repeats it.
- Not a capability claim: 8 training tasks, 4 held-out, 12 (originally 24) effective
  updates, one seed per arm.
- Not a test of the obligation reward's quality: the reward is held fixed across arms.

## Falsifiers

| Hypothesis | Falsified if |
|---|---|
| Masking changes the learning outcome | Held-out joint success and admission differ by < 12 points with overlapping Wilson intervals, and no cost difference |
| Control learns format better | Control's held-out admission is not higher, or the difference is within noise |
| Masking produces larger advantage on admitted samples | Mean \|advantage\| on admitted samples is not higher in the masked arm |
| Masking is cheaper per effective update | Rollouts per effective update are not lower in the masked arm |

## Why the arithmetic matters

For a fixed group, *masked-effective implies control-effective*: invalid samples carry
reward 0 in the control population, so a constant control population implies a constant
admitted subset. Masking can only lose effective groups, never create one. This was
derived and property-tested before the run (see `tests/test_grpo.py`). It means the
mechanism endpoint cannot favour masking; only the parameter trajectory — and therefore
the held-out endpoints — can.

## Threats

1. Single seed per arm; the arms share prompts and decoding seeds, so the comparison is
   paired at the prompt level but not at the initialisation level (same base model,
   same LoRA init seed).
2. 1.5B on 8 tasks: both arms may sit at the floor. The mechanism endpoints (effective
   updates, advantage magnitude, admission) still carry the result.
3. Format learning and repair learning are entangled: a model may learn the JSON shape
   first and the repair second. Admission is reported separately from HOLDS for this
   reason.
4. Local MPS hardware, not a datacenter GPU: throughput lower, numerics may differ from
   CUDA. Recorded in the manifest; no claim depends on it.
5. The corpus is the sibling frontier experiment's 12 tasks; the 8/4 split is
   lexicographic, not stratified by family. Family composition of the held-out set is
   reported.

## Pre-committed reading

The report applies the SPEC's rule mechanically and prints all three readings
(supported / falsified / not established), plus the cost ratio and the mechanism table.
A null is a result; it is not reported as equivalence.

## Amendment v2 (third arm, second seed, 16 decodings)

- **`penalty` keeps `control`'s all-G baseline** and changes exactly one thing: a rejected
  completion is worth −1.0 instead of 0.0. Its advantages are computed by identical
  arithmetic on a different reward vector, so a penalty-vs-control difference is a
  difference in how format failure is valued, not a different task or grader.
- **The masked-effective-implies-control-effective proof is specific to control's reward
  vector.** It does not extend to penalty: one HOLDS among seven rejected samples is a
  zero-variance control group (all zeros) but a variable penalty group (one 1.0, seven
  −1.0). Penalty can therefore create effective groups that control lacks.
- **Falsifiers for the new pairs** are the same pre-committed rule: penalty vs control and
  penalty vs masked are decided by ≥ 12 percentage points with non-overlapping Wilson
  intervals, and each pair reports its rollout-cost ratio.
- **Seeds:** the v1 seed covers arms A and B; the v2 seed covers all three. Unequal seed
  counts are reported, and the pooled row sums draws rather than averaging per-seed rates.
- The "one seed per arm" and "12 effective updates" statements above describe the v1 design.
  The run budget actually used is recorded in each manifest; the v1 deviation and v2 fix it
  at 8 effective updates under a 40-group cap.
