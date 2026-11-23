"""Group-relative advantages and the three arm treatments.

This module is the load-bearing maths of the experiment, and it is deliberately
dependency-free (standard library only).  ``train.py``, ``analysis/summarize.py``
and the unit tests all import it, and none of them may need a GPU, a model, a
tokenizer or the network to do so.

Protocol (``docs/SPEC.md``, "Arms" and "Amendment v2")
-----------------------------------------------------
A group is the ``G`` completions sampled for one prompt.  A completion the strict
envelope rejected is an **invalid action**; it is recorded, but it is not a wrong
answer.  What it is *worth* is the arm's decision, and that decision lives in the
reward vector, not in the baseline: see :func:`invalid_reward`.

* ``"control"`` -- invalid reward ``0.0``; the baseline (mean, population std) is
  computed over **all G** samples.  Invalid samples keep their reward-0 advantage,
  so whenever anything in the group succeeds they receive a *negative* advantage.
  That is gradient pressure against the invalid format, not against the repair.
* ``"masked"``  -- invalid reward ``0.0`` but *excluded*; the baseline is computed
  over the **admitted** samples only.  Invalid samples are removed from the group
  entirely: they get advantage ``0.0`` and contribute nothing to the loss.  Fewer
  than two admitted samples means there is nothing to normalise against, so the
  group yields no signal and is reported as ``effective=False`` (the protocol
  calls it "skipped").
* ``"penalty"`` -- invalid reward ``-1.0``; the baseline is computed over **all G**
  samples, exactly as in ``control``.  The only difference from ``control`` is
  the reward a rejected completion carries (``rollout.score_completion``), so the
  advantages are computed by identical arithmetic on a different reward vector.
  Amendment v2's rationale: a format error is cheaper to fix than a wrong repair,
  so it should rank *below* a wrong-but-valid answer (``0.0``), not equal to it.

A group is **effective** when its arm's population has non-zero standard
deviation.  That is exactly the condition under which the advantage vector is
non-degenerate and the update carries a gradient, which is why the protocol
budgets *effective updates* rather than rollouts.  For ``penalty`` that means any
group whose rewards are not all identical.

No treatment is assumed to be better.  Masking removes format pressure but
also removes the contrast that makes an admitted sample's advantage positive: if
every admitted sample in a group is identical (all HOLDS or all VIOLATED) the
masked population has zero variance and the group is not effective.  The
direction of the effect is what the experiment measures.

Numerics
--------
``STD_EPS`` is a numerical floor, not a hyper-parameter: a spread below it is a
zero-variance group by definition, and a denominator of ``std + STD_EPS`` cannot
divide by zero.  For any group that is effective the correction is < 1e-6
relative, so it does not change an advantage at any reported precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

#: The arms of the preregistered experiment (A, B and Amendment v2's C).  Order
#: matters only for display.
ARMS: tuple[str, ...] = ("control", "masked", "penalty")

#: Arms whose baseline population is every sample in the group.  ``penalty``
#: differs from ``control`` only in the reward a rejected completion carries.
ALL_SAMPLE_ARMS: frozenset[str] = frozenset({"control", "penalty"})

#: The reward a *rejected* completion carries, per arm.  Admitted samples are
#: unaffected: they are rewarded 1.0 iff HOLDS and 0.0 otherwise in every arm.
INVALID_REWARDS: dict[str, float] = {
    "control": 0.0,
    "masked": 0.0,
    "penalty": -1.0,
}

#: Numerical floor below which a population spread counts as zero variance.
STD_EPS: float = 1e-8


class AdvantageResult(NamedTuple):
    """Return type of :func:`group_advantage`.

    A named tuple, so callers may either unpack it (``advantages, effective =
    group_advantage(...)``) or use the attributes.
    """

    advantages: list[float]
    effective: bool


@dataclass(frozen=True)
class GroupStats:
    """Descriptive statistics of one group under one arm's treatment."""

    arm: str
    n: int
    n_valid: int
    n_used: int
    mean: float
    std: float
    variance: float
    effective: bool
    mean_abs_advantage_admitted: Optional[float]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    return arm


def invalid_reward(arm: str) -> float:
    """The reward an **admitted=False** completion carries under ``arm``.

    ``control`` and ``masked`` score a rejected completion ``0.0`` (masking
    removes it from the baseline; it still logs reward 0.0).  ``penalty`` scores
    it ``-1.0`` so it ranks below a wrong-but-valid attempt at ``0.0``.

    This is the *only* arm difference that lives outside the baseline: the two
    all-sample arms (``control`` and ``penalty``) compute advantages with
    identical arithmetic.  The helper is exhaustive over :data:`ARMS` by
    construction (:data:`INVALID_REWARDS` is keyed by every arm) and raises for
    anything else.  Admitted completions are never routed through it.
    """

    return INVALID_REWARDS[_check_arm(arm)]


def _coerce(rewards: Sequence[float], valid_mask: Sequence[bool]) -> tuple[list[float], list[bool]]:
    """Validate and coerce the two per-sample vectors to plain lists."""

    values = [float(value) for value in rewards]
    valid = [bool(flag) for flag in valid_mask]
    if len(values) != len(valid):
        raise ValueError(
            f"rewards and valid_mask must be the same length ({len(values)} != {len(valid)})"
        )
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"reward[{index}] is not finite: {value!r}")
    return values, valid


def _population_mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Population (not sample) mean and standard deviation."""

    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    # Guard against a tiny negative from floating-point cancellation.
    return mean, math.sqrt(max(0.0, variance))


# ---------------------------------------------------------------------------
# The arm treatments
# ---------------------------------------------------------------------------


def normalization_indices(valid_mask: Sequence[bool], arm: str) -> list[int]:
    """Indices of the samples that form the group baseline under ``arm``.

    Both all-sample arms (``control`` and ``penalty``) use every index;
    ``masked`` uses the admitted ones.
    """

    _check_arm(arm)
    if arm in ALL_SAMPLE_ARMS:
        return list(range(len(valid_mask)))
    return [index for index, flag in enumerate(valid_mask) if flag]


def treated_mask(valid_mask: Sequence[bool], arm: str) -> list[bool]:
    """Which samples contribute to the gradient under ``arm``.

    ``control`` and ``penalty`` backpropagate through every sample (the invalid
    ones carry their reward-0 or reward-(−1) advantage, which is the negative
    format pressure the protocol names).  ``masked`` backpropagates through
    admitted samples only: an invalid action is removed from the group, not
    merely from its mean.
    """

    _check_arm(arm)
    if arm in ALL_SAMPLE_ARMS:
        return [True] * len(valid_mask)
    return [bool(flag) for flag in valid_mask]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def group_advantage(
    rewards: Sequence[float],
    valid_mask: Sequence[bool],
    arm: str,
) -> AdvantageResult:
    """Group-relative advantages for one group under one arm.

    ``rewards`` are the per-sample rewards (``1.0`` iff admitted and the grader
    returned HOLDS, else ``0.0``).  ``valid_mask`` marks the admitted samples.
    ``arm`` is ``"control"`` or ``"masked"``.

    Returns the advantage vector (same length as ``rewards``; ``0.0`` at every
    position that does not enter the arm's population) and ``effective``.

    ``effective`` is ``True`` iff the group yielded a nonzero-variance baseline,
    i.e. iff at least two samples enter the population and their population
    standard deviation exceeds ``STD_EPS``.  When it is ``False`` every advantage
    is exactly ``0.0``: the group carries no gradient, and the caller must not
    take an optimizer step for it.
    """

    values, valid = _coerce(rewards, valid_mask)
    _check_arm(arm)

    indices = normalization_indices(valid, arm)
    if len(indices) < 2:
        # Fewer than two samples in the population: masked with < 2 admitted, or
        # a degenerate group size.  Nothing to normalise against.
        return AdvantageResult([0.0] * len(values), False)

    population = [values[index] for index in indices]
    mean, std = _population_mean_std(population)

    if std <= STD_EPS:
        return AdvantageResult([0.0] * len(values), False)

    denominator = std + STD_EPS
    advantages = [0.0] * len(values)
    for index in indices:
        advantages[index] = (values[index] - mean) / denominator
    return AdvantageResult(advantages, True)


def group_stats(rewards: Sequence[float], valid_mask: Sequence[bool], arm: str) -> GroupStats:
    """Descriptive statistics for one group under one arm (for metrics/report).

    ``mean``/``std``/``variance`` describe the arm's *normalisation population*
    (all G for control and penalty, admitted only for masked), which is what the
    protocol
    calls "group variance".  ``mean_abs_advantage_admitted`` is the mean absolute
    advantage over the admitted samples -- the secondary endpoint that shows
    whether the surviving samples carry usable signal -- and is ``None`` when
    nothing was admitted.
    """

    values, valid = _coerce(rewards, valid_mask)
    _check_arm(arm)

    indices = normalization_indices(valid, arm)
    population = [values[index] for index in indices]
    mean, std = _population_mean_std(population)

    result = group_advantage(values, valid, arm)

    admitted_indices = [index for index, flag in enumerate(valid) if flag]
    if admitted_indices:
        mean_abs = sum(abs(result.advantages[index]) for index in admitted_indices) / len(
            admitted_indices
        )
    else:
        mean_abs = None

    return GroupStats(
        arm=arm,
        n=len(values),
        n_valid=len(admitted_indices),
        n_used=len(indices),
        mean=mean,
        std=std,
        variance=std * std,
        effective=result.effective,
        mean_abs_advantage_admitted=mean_abs,
    )


def group_loss(
    advantages: Sequence[float],
    sum_logprobs: Sequence[float],
    treated: Sequence[bool],
) -> float:
    """The per-group loss: ``-mean(advantage * sum_logprob)`` over treated samples.

    ``sum_logprobs`` is the sum of the log-probabilities the policy assigned to a
    sample's completion tokens (the policy-gradient score of that sample).
    ``treated`` selects the samples that contribute (all G for control and
    penalty, admitted only for masked).  Returns ``0.0`` for a group with no
    treated samples, which is the no-signal case and must not be turned into an
    optimizer step.
    """

    if len(advantages) != len(sum_logprobs):
        raise ValueError(
            "advantages and sum_logprobs must be the same length "
            f"({len(advantages)} != {len(sum_logprobs)})"
        )
    if len(treated) != len(advantages):
        raise ValueError(
            "treated mask must have one entry per sample "
            f"({len(treated)} != {len(advantages)})"
        )

    picked = [index for index, flag in enumerate(treated) if flag]
    if not picked:
        return 0.0

    total = 0.0
    for index in picked:
        total += float(advantages[index]) * float(sum_logprobs[index])
    return -total / len(picked)


__all__ = [
    "ALL_SAMPLE_ARMS",
    "ARMS",
    "AdvantageResult",
    "INVALID_REWARDS",
    "GroupStats",
    "STD_EPS",
    "group_advantage",
    "group_loss",
    "group_stats",
    "invalid_reward",
    "normalization_indices",
    "treated_mask",
]
