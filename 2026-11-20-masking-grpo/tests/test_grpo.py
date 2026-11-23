#!/usr/bin/env python3
"""Offline, deterministic tests for the load-bearing maths in ``grpo.py``.

Nothing here needs a model, a tokenizer, a GPU, a network or the corpus: the arm
treatments are pure functions of ``(rewards, valid_mask, arm)``, which is the
point of keeping them in a stdlib-only module.

The by-hand numbers
-------------------
Every expected value below is derived on paper from the definitions in
``docs/SPEC.md``.  The canonical case is a group of G = 8 with exactly three
admitted samples whose rewards are ``1, 1, 0`` (two HOLDS, one VIOLATED) and five
rejected samples at reward 0:

control (population = all 8)
    mean  = 2/8                    = 0.25
    var   = 2/8 - (1/4)^2 = 3/16   -> std = sqrt(3)/4   = 0.4330127018922193
    adv(1) = (1 - 1/4) / std       = sqrt(3)            = 1.7320508075688772
    adv(0) = (0 - 1/4) / std       = -1/sqrt(3)         = -0.5773502691896257
    mean |adv| over the 3 admitted = (2*sqrt(3) + 1/sqrt(3)) / 3 = 1.3471506...

masked (population = the 3 admitted)
    mean  = 2/3
    var   = 2/3 - (2/3)^2 = 2/9    -> std = sqrt(2)/3   = 0.4714045207910317
    adv(1) = (1/3) / std           = 1/sqrt(2)          = 0.7071067811865476
    adv(0) = (-2/3) / std          = -sqrt(2)           = -1.4142135623730951
    mean |adv| over the 3 admitted = (2/sqrt(2) + sqrt(2)) / 3 = 0.9428090...

Read the trade-off off those numbers: masking gives the winning sample a *smaller*
advantage (0.707 vs 1.732) because five zeros no longer inflate the spread, and it
gives the failing-but-admitted sample a *larger* one (1.414 vs 0.577).  Control
additionally spends gradient on the five rejected samples at -1/sqrt(3) each --
that is the format pressure the protocol names, not neutral noise.

A relative tolerance of 1e-6 absorbs the ``std + STD_EPS`` denominator, which is a
< 2.3e-8 relative correction at these magnitudes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpo  # noqa: E402

REL = 1e-6

#: G = 8, the first three samples admitted with rewards 1, 1, 0.
THREE_OF_EIGHT_REWARDS = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
THREE_OF_EIGHT_VALID = [True, True, True, False, False, False, False, False]

CONTROL_MEAN = 0.25
CONTROL_STD = math.sqrt(3.0) / 4.0            # 0.4330127018922193
CONTROL_ADV_WIN = math.sqrt(3.0)              # 1.7320508075688772
CONTROL_ADV_FAIL = -1.0 / math.sqrt(3.0)      # -0.5773502691896257

MASKED_MEAN = 2.0 / 3.0
MASKED_STD = math.sqrt(2.0) / 3.0             # 0.4714045207910317
MASKED_ADV_WIN = 1.0 / math.sqrt(2.0)         # 0.7071067811865476
MASKED_ADV_FAIL = -math.sqrt(2.0)             # -1.4142135623730951


# ---------------------------------------------------------------------------
# The four cases the protocol sections name
# ---------------------------------------------------------------------------


def test_all_zero_group_is_not_effective_in_either_arm():
    rewards = [0.0] * 8
    valid = [False] * 8

    control, control_effective = grpo.group_advantage(rewards, valid, "control")
    masked, masked_effective = grpo.group_advantage(rewards, valid, "masked")

    assert control_effective is False
    assert masked_effective is False
    assert control == [0.0] * 8
    assert masked == [0.0] * 8


def test_all_zero_group_with_admitted_but_violated_samples_is_still_not_effective():
    """Admission alone creates no variance: three admitted samples that all fail are
    reward 0, so both arms see an all-zero population and neither is effective."""

    rewards = [0.0] * 8
    valid = [True, True, True, False, False, False, False, False]

    for arm in ("control", "masked"):
        advantages, effective = grpo.group_advantage(rewards, valid, arm)
        assert effective is False, arm
        assert advantages == [0.0] * 8, arm


def test_mixed_group_three_of_eight_valid_matches_hand_computation():
    control, control_effective = grpo.group_advantage(
        THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "control"
    )
    masked, masked_effective = grpo.group_advantage(
        THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "masked"
    )

    assert control_effective is True
    assert masked_effective is True

    # control: normalised over all 8, so the five invalid samples carry -1/sqrt(3).
    assert control[:3] == pytest.approx([CONTROL_ADV_WIN, CONTROL_ADV_WIN, CONTROL_ADV_FAIL], rel=REL)
    assert control[3:] == pytest.approx([CONTROL_ADV_FAIL] * 5, rel=REL)
    assert math.fsum(control) == pytest.approx(0.0, abs=1e-9)

    # masked: normalised over the 3 admitted; the invalid samples are removed.
    assert masked[:3] == pytest.approx([MASKED_ADV_WIN, MASKED_ADV_WIN, MASKED_ADV_FAIL], rel=REL)
    assert masked[3:] == [0.0] * 5
    assert math.fsum(masked) == pytest.approx(0.0, abs=1e-9)

    # The stats are over each arm's own population, which is the difference.
    control_stats = grpo.group_stats(THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "control")
    masked_stats = grpo.group_stats(THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "masked")
    assert (control_stats.mean, control_stats.std) == pytest.approx((CONTROL_MEAN, CONTROL_STD), rel=REL)
    assert (masked_stats.mean, masked_stats.std) == pytest.approx((MASKED_MEAN, MASKED_STD), rel=REL)
    assert masked_stats.n_used == 3 and masked_stats.n_valid == 3
    assert control_stats.n_used == 8 and control_stats.n_valid == 3

    # The secondary endpoint: mean |advantage| over the admitted samples.
    assert control_stats.mean_abs_advantage_admitted == pytest.approx(
        (2 * math.sqrt(3.0) + 1 / math.sqrt(3.0)) / 3, rel=REL
    )
    assert masked_stats.mean_abs_advantage_admitted == pytest.approx(
        (2 / math.sqrt(2.0) + math.sqrt(2.0)) / 3, rel=REL
    )

    # The trade-off, stated as arithmetic rather than as a preference: masking
    # shrinks the winning sample's advantage and enlarges the failing one's,
    # while control additionally spends gradient on the five invalid samples.
    assert abs(masked[0]) < abs(control[0])
    assert abs(masked[2]) > abs(control[2])
    assert control_stats.mean_abs_advantage_admitted > masked_stats.mean_abs_advantage_admitted


def test_all_valid_group_is_identical_across_arms():
    rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    valid = [True] * 8

    control, control_effective = grpo.group_advantage(rewards, valid, "control")
    masked, masked_effective = grpo.group_advantage(rewards, valid, "masked")

    assert control_effective is True and masked_effective is True
    assert control == pytest.approx(masked, rel=REL)
    # rewards: 4 holds of 8 -> mean 0.5, and each advantage is +/- 1 (std 0.5).
    assert control == pytest.approx([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0], rel=REL)


def test_single_valid_sample_is_skipped_by_masked():
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    valid = [True, False, False, False, False, False, False, False]

    masked, masked_effective = grpo.group_advantage(rewards, valid, "masked")
    assert masked_effective is False
    assert masked == [0.0] * 8

    control, control_effective = grpo.group_advantage(rewards, valid, "control")
    assert control_effective is True
    assert control[0] == pytest.approx(math.sqrt(7.0), rel=REL)


def test_two_valid_samples_with_identical_rewards_are_not_effective_for_masked():
    """The masked arm's failure mode: identical admitted samples have no contrast."""

    rewards = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    valid = [True, True, False, False, False, False, False, False]

    masked, masked_effective = grpo.group_advantage(rewards, valid, "masked")
    assert masked_effective is False
    assert masked == [0.0] * 8

    control, control_effective = grpo.group_advantage(rewards, valid, "control")
    assert control_effective is True
    assert control[:2] == pytest.approx([math.sqrt(3.0)] * 2, rel=REL)


def test_two_valid_samples_with_different_rewards_are_effective_for_masked():
    rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    valid = [True, True, False, False, False, False, False, False]

    masked, masked_effective = grpo.group_advantage(rewards, valid, "masked")
    assert masked_effective is True
    assert masked[:2] == pytest.approx([1.0, -1.0], rel=REL)
    assert masked[2:] == [0.0] * 6
    assert grpo.group_stats(rewards, valid, "masked").std == pytest.approx(0.5, rel=REL)


def test_all_valid_and_all_holding_is_zero_variance_in_both_arms():
    rewards = [1.0] * 8
    valid = [True] * 8

    for arm in ("control", "masked"):
        advantages, effective = grpo.group_advantage(rewards, valid, arm)
        assert effective is False, arm
        assert advantages == [0.0] * 8, arm


# ---------------------------------------------------------------------------
# The epsilon and the no-signal contract
# ---------------------------------------------------------------------------


def test_spread_below_epsilon_counts_as_zero_variance():
    rewards = [1e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    valid = [True] * 8

    for arm in ("control", "masked"):
        advantages, effective = grpo.group_advantage(rewards, valid, arm)
        assert effective is False, arm
        assert advantages == [0.0] * 8, arm


def test_masked_effective_implies_control_effective_for_binary_reward():
    """A structural property of the protocol's reward definition.

    Invalid samples are reward 0 in control, so control's population is the
    admitted rewards padded with zeros.  If control's population were constant,
    the admitted subset would be constant too, so masked would not be effective
    either.  Contrapositive: masked effective => control effective.  The
    mechanism endpoint can therefore only tie or favour control; that is why the
    protocol's co-primary endpoints are the held-out outcome and admission, and
    why it budgets effective steps rather than predicting a direction.
    """

    masks = [
        [True] * 8,
        [True, True, True, False, False, False, False, False],
        [True, True, False, False, False, False, False, False],
        [True, True, True, True, True, False, False, False],
        [False, False, True, True, True, True, True, True],
    ]
    reward_patterns = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0] * 8,
    ]
    for valid in masks:
        for rewards in reward_patterns:
            _, control_effective = grpo.group_advantage(rewards, valid, "control")
            _, masked_effective = grpo.group_advantage(rewards, valid, "masked")
            if masked_effective:
                assert control_effective, (valid, rewards)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        grpo.group_advantage([1.0, 0.0], [True], "control")


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        grpo.group_advantage([1.0, 0.0], [True, True], "bogus")


def test_non_finite_reward_is_rejected():
    with pytest.raises(ValueError, match="not finite"):
        grpo.group_advantage([1.0, float("nan")], [True, True], "control")


def test_empty_group_is_not_effective():
    for arm in ("control", "masked"):
        advantages, effective = grpo.group_advantage([], [], arm)
        assert advantages == [] and effective is False


def test_advantage_result_unpacks_like_a_tuple_and_has_attributes():
    result = grpo.group_advantage(THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "masked")
    advantages, effective = result
    assert advantages is result.advantages
    assert effective is result.effective


# ---------------------------------------------------------------------------
# treated_mask / group_loss
# ---------------------------------------------------------------------------


def test_treated_mask_follows_the_arm():
    valid = [True, False, True, False]
    assert grpo.treated_mask(valid, "control") == [True, True, True, True]
    assert grpo.treated_mask(valid, "masked") == valid
    with pytest.raises(ValueError):
        grpo.treated_mask(valid, "bogus")


def test_group_loss_is_minus_mean_advantage_times_sum_logprob():
    advantages = [1.0, -1.0, 0.5, 0.0]
    sum_logprobs = [-1.0, -3.0, -2.0, -4.0]

    # all four treated: -((1*-1) + (-1*-3) + (0.5*-2) + 0) / 4 = -(-1+3-1)/4 = -0.25
    assert grpo.group_loss(advantages, sum_logprobs, [True] * 4) == pytest.approx(-0.25)

    # masked: only the first two treated: -( -1 + 3 ) / 2 = -1.0
    assert grpo.group_loss(advantages, sum_logprobs, [True, True, False, False]) == pytest.approx(-1.0)

    # nothing treated: no gradient, exactly zero (and never a division by zero).
    assert grpo.group_loss(advantages, sum_logprobs, [False] * 4) == 0.0


def test_group_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        grpo.group_loss([1.0, 0.0], [-1.0], [True, True])
    with pytest.raises(ValueError, match="treated mask"):
        grpo.group_loss([1.0, 0.0], [-1.0, -1.0], [True])


def test_masked_loss_drops_the_invalid_samples_entirely():
    """The loss denominator differs because the treatment differs.

    Fixing one advantage vector (the masked one) and varying only the treated set
    shows what "removed from the group" means: control divides by all G, masked
    by the admitted samples only, so the invalid samples cannot dilute the
    admitted signal.  In the real loop each arm also uses its *own* advantages;
    this test isolates the denominator.
    """

    advantages, _ = grpo.group_advantage(
        THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "masked"
    )
    # Admitted samples have distinct policy scores, so the mean is not trivially 0.
    sum_logprobs = [-1.0, -2.0, -3.0, -5.0, -5.0, -5.0, -5.0, -5.0]

    masked_treated = grpo.treated_mask(THREE_OF_EIGHT_VALID, "masked")
    control_treated = grpo.treated_mask(THREE_OF_EIGHT_VALID, "control")

    masked_loss = grpo.group_loss(advantages, sum_logprobs, masked_treated)
    control_loss = grpo.group_loss(advantages, sum_logprobs, control_treated)

    # masked: -(adv0*-1 + adv1*-2 + adv2*-3) / 3
    expected_masked = -(
        MASKED_ADV_WIN * -1.0 + MASKED_ADV_WIN * -2.0 + MASKED_ADV_FAIL * -3.0
    ) / 3.0
    # control: the same numerator (invalid advantages are exactly 0) over 8 samples
    expected_control = -(
        MASKED_ADV_WIN * -1.0 + MASKED_ADV_WIN * -2.0 + MASKED_ADV_FAIL * -3.0
    ) / 8.0

    assert masked_treated == THREE_OF_EIGHT_VALID
    assert sum(1 for flag in masked_treated if flag) == 3
    assert sum(1 for flag in control_treated if flag) == 8
    assert advantages[3:] == [0.0] * 5
    assert masked_loss == pytest.approx(expected_masked, rel=REL)
    assert control_loss == pytest.approx(expected_control, rel=REL)
    # Diluting the same signal over eight samples makes the step 3/8 as large.
    assert abs(control_loss) < abs(masked_loss)


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def test_group_stats_reports_none_when_nothing_was_admitted():
    stats = grpo.group_stats([0.0] * 8, [False] * 8, "masked")
    assert stats.n_valid == 0
    assert stats.mean_abs_advantage_admitted is None
    assert stats.effective is False
    assert stats.variance == 0.0


def test_group_stats_variance_is_std_squared():
    stats = grpo.group_stats(THREE_OF_EIGHT_REWARDS, THREE_OF_EIGHT_VALID, "control")
    assert stats.variance == pytest.approx(CONTROL_STD**2, rel=REL)
    assert stats.std == pytest.approx(CONTROL_STD, rel=REL)
    assert stats.n == 8
