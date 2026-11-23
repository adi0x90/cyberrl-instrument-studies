#!/usr/bin/env python3
"""Offline tests for the third arm (``penalty``) and the three-arm report.

What is asserted, in order:

1. ``grpo.invalid_reward`` is exhaustive over the arm set and agrees with the
   protocol (0.0 for control/masked, -1.0 for penalty);
2. ``penalty`` and ``control`` are the *same* baseline computation: given the same
   reward vector they produce identical advantages, statistics and treated sets
   (the arm difference lives in the reward vector, not in the maths);
3. a rejected completion is scored ``invalid_reward(arm)`` end to end through
   ``rollout.run_group``, while admitted samples keep 1.0/HOLDS and 0.0 otherwise;
4. ``analysis/summarize.py`` renders a three-arm report, marks an absent arm as
   "not run" (never a zero), and applies the decision rule to all three pairs;
5. seed detection and the pooled-vs-per-seed arithmetic: the pooled row sums
   draws, it never averages per-seed rates.

Requirement: no network, no GPU, no model, no API key. The grading calls are
monkeypatched, so no child process is spawned either.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpo  # noqa: E402
import rollout  # noqa: E402  (also puts the frontier experiment on sys.path)
import harness.tasks_io as tasks_io  # noqa: E402

TRAIN = ROOT / "train.py"
SUMMARIZE = ROOT / "analysis" / "summarize.py"
TASK_ID = "authz_delegated_write_01"


def load_summarize():
    spec = importlib.util.spec_from_file_location("summarize_penalty_under_test", SUMMARIZE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarize_penalty_under_test"] = module
    spec.loader.exec_module(module)
    return module


SUMM = load_summarize()


def run_python(script: Path, args: list[str], timeout: int = 120):
    import subprocess
    return subprocess.run(
        [sys.executable, str(script)] + args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# 1. The helper
# ---------------------------------------------------------------------------


def test_the_arm_set_includes_penalty():
    assert "penalty" in grpo.ARMS
    assert len(grpo.ARMS) == 3
    assert len(set(grpo.ARMS)) == len(grpo.ARMS)


def test_invalid_reward_is_exhaustive_over_the_arm_set():
    # Every arm has a value, and there is no value for a non-arm.
    assert set(grpo.INVALID_REWARDS) == set(grpo.ARMS)
    for arm in grpo.ARMS:
        value = grpo.invalid_reward(arm)
        assert isinstance(value, float)
        assert math.isfinite(value)
    assert grpo.invalid_reward("control") == 0.0
    assert grpo.invalid_reward("masked") == 0.0
    assert grpo.invalid_reward("penalty") == -1.0


def test_invalid_reward_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        grpo.invalid_reward("bogus")


def test_penalty_is_an_all_sample_arm():
    valid = [True, False, True, False]
    assert grpo.normalization_indices(valid, "penalty") == grpo.normalization_indices(
        valid, "control"
    )
    assert grpo.treated_mask(valid, "penalty") == grpo.treated_mask(valid, "control")


# ---------------------------------------------------------------------------
# 2. Identical quantities given identical rewards
# ---------------------------------------------------------------------------

#: Each vector is fed to both arms unchanged: the point is that the *maths* is
#: identical, not that the reward vectors are identical in a real run.
IDENTICAL_INPUT_CASES = [
    (
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [True, True, True, False, False, False, False, False],
    ),
    (
        [1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        [True, True, False, False, False, False, False, False],
    ),
    ([0.0, 0.0, 0.0, 0.0], [False, False, False, False]),
    ([-1.0, -1.0, -1.0, -1.0], [False, False, False, False]),
    ([1.0, 1.0, 1.0, 1.0], [True, True, True, True]),
    ([1.0, 0.0, 1.0, 0.0], [True, True, True, True]),
]


@pytest.mark.parametrize("rewards,valid", IDENTICAL_INPUT_CASES)
def test_penalty_and_control_produce_identical_advantages_for_identical_rewards(rewards, valid):
    control = grpo.group_advantage(rewards, valid, "control")
    penalty = grpo.group_advantage(rewards, valid, "penalty")

    assert penalty.effective is control.effective
    assert penalty.advantages == pytest.approx(control.advantages, rel=1e-12, abs=1e-12)

    control_stats = grpo.group_stats(rewards, valid, "control")
    penalty_stats = grpo.group_stats(rewards, valid, "penalty")
    for field in ("n", "n_valid", "n_used", "mean", "std", "variance", "effective"):
        assert getattr(penalty_stats, field) == getattr(control_stats, field), field
    assert penalty_stats.mean_abs_advantage_admitted == control_stats.mean_abs_advantage_admitted


def test_penalty_and_control_differ_only_when_the_reward_vector_differs():
    valid = [True, True, False, False, False, False, False, False]
    # What rollout.py actually produces: rejected samples are 0.0 in control and
    # -1.0 in penalty; the admitted rewards are identical.
    control_rewards = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    penalty_rewards = [1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    control_adv, control_effective = grpo.group_advantage(control_rewards, valid, "control")
    penalty_adv, penalty_effective = grpo.group_advantage(penalty_rewards, valid, "penalty")

    assert control_effective is True and penalty_effective is True
    assert control_adv != pytest.approx(penalty_adv)
    # The admitted samples' own rewards are untouched by the arm...
    assert control_rewards[:2] == penalty_rewards[:2]
    # ...but their advantages move, because the baseline moved.
    assert control_adv[0] > 0.0 and penalty_adv[0] > 0.0
    assert control_adv[0] != pytest.approx(penalty_adv[0])


def test_penalty_group_is_effective_iff_its_rewards_are_not_all_identical():
    all_invalid = grpo.group_advantage([-1.0] * 8, [False] * 8, "penalty")
    assert all_invalid.effective is False
    assert all_invalid.advantages == [0.0] * 8

    mixed = grpo.group_advantage([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0], [False] * 8, "penalty")
    assert mixed.effective is True
    assert any(advantage > 0.0 for advantage in mixed.advantages)
    assert any(advantage < 0.0 for advantage in mixed.advantages)


# ---------------------------------------------------------------------------
# 3. Propagation through a fake rollout
# ---------------------------------------------------------------------------


def fake_execute_patch(task_arg, expression):
    """The reference fix passes; anything else fails security (=> VIOLATED)."""

    security_total = len(task_arg["security_cases"])
    passed = security_total if expression == task_arg["reference_fix"] else 0
    return tasks_io.ExecutionResult(
        compiled=True,
        security_passed=passed,
        security_total=security_total,
        benign_passed=len(task_arg["benign_cases"]),
        benign_total=len(task_arg["benign_cases"]),
        detail=[],
        error=None,
    )


def fixed_generate(completions):
    def generate(prompt: str, n: int, seed: int):
        assert n == len(completions)
        return [rollout.Generation(text=text, prompt_tokens=5, completion_tokens=3) for text in completions]

    return generate


def mixed_group(monkeypatch, arm: str):
    """One group: a HOLDS, an admitted VIOLATED, six rejected completions."""

    monkeypatch.setattr(rollout, "execute_patch", fake_execute_patch)
    task = rollout.load_task_by_id(TASK_ID)
    holds = json.dumps({"expression": task["reference_fix"]}, sort_keys=True)
    violated = json.dumps({"expression": task["seeded_bad"][0]["expression"]}, sort_keys=True)
    completions = [holds, violated] + ["not json at all"] * 6
    return rollout.run_group(task, fixed_generate(completions), group_index=0, g=8, seed=0, arm=arm)


def test_penalty_rewards_rejected_completions_minus_one(monkeypatch):
    group = mixed_group(monkeypatch, "penalty")

    assert group.rewards == [1.0, 0.0] + [-1.0] * 6
    assert group.valid_mask == [True, True] + [False] * 6
    assert [score.verdict for score in group.scores] == ["HOLDS", "VIOLATED"] + ["UNKNOWN"] * 6

    for record in group.records:
        assert record["admitted"] is (record["verdict"] != "UNKNOWN")
        if record["admitted"]:
            # Admitted samples are unaffected by the arm: 1.0 iff HOLDS, else 0.0.
            assert record["reward"] == (1.0 if record["verdict"] == "HOLDS" else 0.0)
        else:
            # Rejected: displayed as UNKNOWN/unadmitted, valued at -1.0.
            assert record["reward"] == grpo.invalid_reward("penalty") == -1.0
            assert record["verdict"] == "UNKNOWN"


def test_control_and_masked_keep_reward_zero_for_rejected_completions(monkeypatch):
    for arm in ("control", "masked"):
        group = mixed_group(monkeypatch, arm)
        assert group.rewards == [1.0, 0.0] + [0.0] * 6, arm


def test_penalty_reward_vector_flows_into_the_advantage(monkeypatch):
    penalty_group = mixed_group(monkeypatch, "penalty")
    control_group = mixed_group(monkeypatch, "control")

    penalty_adv, penalty_effective = grpo.group_advantage(
        penalty_group.rewards, penalty_group.valid_mask, "penalty"
    )
    control_adv, control_effective = grpo.group_advantage(
        control_group.rewards, control_group.valid_mask, "control"
    )

    assert penalty_effective is True and control_effective is True
    # Every rejected sample sits below the penalty baseline, so it is pushed down.
    assert all(advantage < 0.0 for advantage in penalty_adv[2:])
    assert penalty_adv != pytest.approx(control_adv)


# ---------------------------------------------------------------------------
# Artifact helpers for the report tests
# ---------------------------------------------------------------------------


def write_arm_artifacts(
    directory: Path,
    arm: str,
    *,
    seed: int,
    samples: list[tuple[bool, str, float]],
    eval_holds: int,
    eval_draws: int,
    eval_admitted: int,
) -> None:
    """Write a minimal but self-consistent rollout/metrics/eval triple."""

    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (admitted, verdict, reward) in enumerate(samples):
        records.append(
            {
                "task_id": "train_task",
                "group_id": "g0000",
                "group_index": 0,
                "sample_index": index,
                "arm": arm,
                "admitted": admitted,
                "verdict": verdict,
                "reward": reward,
            }
        )
    (directory / f"rollouts_{arm}.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    (directory / f"metrics_{arm}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "dry_run": True,
                "budget": {"target_effective_updates": 1, "max_groups": 4},
                "counts": {},
                "groups": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    eval_records = []
    for index in range(eval_draws):
        if index < eval_holds:
            verdict, admitted = "HOLDS", True
        elif index < eval_admitted:
            verdict, admitted = "VIOLATED", True
        else:
            verdict, admitted = "UNKNOWN", False
        eval_records.append(
            {
                "task_id": "held_out_task",
                "decoding": index,
                "admitted": admitted,
                "verdict": verdict,
                "reward": 1.0 if verdict == "HOLDS" else 0.0,
            }
        )
    (directory / f"eval_{arm}.json").write_text(
        json.dumps(
            {"arm": arm, "seed": seed, "dry_run": True, "records": eval_records},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


TWO_GROUPS = [
    (True, "HOLDS", 1.0),
    (True, "VIOLATED", 0.0),
    (False, "UNKNOWN", 0.0),
    (False, "UNKNOWN", 0.0),
]


# ---------------------------------------------------------------------------
# 4. Three-arm summarisation, one arm missing
# ---------------------------------------------------------------------------


def test_three_arm_report_marks_the_missing_arm_not_run(tmp_path):
    write_arm_artifacts(
        tmp_path, "control", seed=20261121, samples=TWO_GROUPS,
        eval_holds=4, eval_draws=16, eval_admitted=10,
    )
    write_arm_artifacts(
        tmp_path, "masked", seed=20261121, samples=TWO_GROUPS,
        eval_holds=8, eval_draws=16, eval_admitted=8,
    )
    # penalty is deliberately absent.

    out = tmp_path / "REPORT.md"
    completed = run_python(SUMMARIZE, ["--dir", str(tmp_path), "--out", str(out)])
    assert completed.returncode == 0, completed.stderr
    text = out.read_text(encoding="utf-8")

    # Three-arm headline table, with an explicit not-run row for penalty.
    assert "## Headline" in text
    headline = text.split("## Headline", 1)[1].split("## 1. Inputs", 1)[0]
    assert "| arm | status |" in headline
    assert "| control | ran |" in headline
    assert "| masked | ran |" in headline
    penalty_row = next(line for line in headline.splitlines() if line.startswith("| penalty |"))
    assert "not run" in penalty_row
    assert "0/0" not in penalty_row  # never a fabricated zero

    # All three preregistered pair comparisons are present, and the two that need
    # penalty are explicitly not run rather than silently omitted.
    for label in ("masked vs control", "penalty vs control", "penalty vs masked"):
        assert label in text, label
    decision_section = text.split("## 3. Decision rule", 1)[1].split("### 3.1 Cost", 1)[0]
    assert "penalty vs control" in decision_section
    assert "penalty vs masked" in decision_section
    assert "NOT RUN" in decision_section

    # The existing sections and the limits block survive.
    for heading in (
        "## 1. Inputs",
        "## 2. Co-primary endpoints",
        "### 2.1 Per-task breakdown",
        "## 4. Secondary endpoints",
        "## 5. Paired per-group variance",
        "## 6. Counts, denominators and the audit trail",
        "## 7. Seeds",
        "## What this does not show",
        "## Provenance",
    ):
        assert heading in text, heading

    # Single seed run: no pooled table is invented.
    assert "Only one seed run was found" in text
    assert "pooled (2 seeds)" not in text


def test_three_arm_report_includes_penalty_when_present(tmp_path):
    for arm, holds in (("control", 2), ("masked", 6), ("penalty", 12)):
        write_arm_artifacts(
            tmp_path, arm, seed=20261121, samples=TWO_GROUPS,
            eval_holds=holds, eval_draws=16, eval_admitted=13,
        )
    out = tmp_path / "REPORT.md"
    completed = run_python(SUMMARIZE, ["--dir", str(tmp_path), "--out", str(out)])
    assert completed.returncode == 0, completed.stderr
    text = out.read_text(encoding="utf-8")
    headline = text.split("## Headline", 1)[1].split("## 1. Inputs", 1)[0]
    penalty_row = next(line for line in headline.splitlines() if line.startswith("| penalty |"))
    assert "not run" not in penalty_row
    assert "12/16" in penalty_row
    # 12/16 vs 2/16 is a 62.5 pp gap with non-overlapping Wilson intervals.
    assert "PENALTY BETTER" in text


# ---------------------------------------------------------------------------
# 5. Seeds: per-seed rows and the pooled row
# ---------------------------------------------------------------------------


def seed_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Two seed directories: v1 (A/B) and v2 (A/B/C)."""

    v1 = tmp_path / "seed20261120"
    write_arm_artifacts(
        v1, "control", seed=20261120, samples=TWO_GROUPS,
        eval_holds=1, eval_draws=4, eval_admitted=2,
    )
    write_arm_artifacts(
        v1, "masked", seed=20261120, samples=TWO_GROUPS,
        eval_holds=3, eval_draws=4, eval_admitted=3,
    )

    v2 = tmp_path / "seed20261121"
    write_arm_artifacts(
        v2, "control", seed=20261121, samples=TWO_GROUPS,
        eval_holds=5, eval_draws=16, eval_admitted=10,
    )
    write_arm_artifacts(
        v2, "masked", seed=20261121, samples=TWO_GROUPS,
        eval_holds=1, eval_draws=16, eval_admitted=9,
    )
    write_arm_artifacts(
        v2, "penalty", seed=20261121, samples=TWO_GROUPS,
        eval_holds=9, eval_draws=16, eval_admitted=12,
    )
    return v1, v2


EMPTY_OVERRIDES = {
    arm: {"rollouts": None, "metrics": None, "eval": None} for arm in SUMM.ARMS
}


def test_seed_runs_are_detected_from_the_metrics_files(tmp_path):
    seed_dirs(tmp_path)
    runs = SUMM.discover_seed_runs(tmp_path, EMPTY_OVERRIDES)

    assert len(runs) == 2
    assert [run.seed for run in runs] == [20261120, 20261121]
    assert [run.label for run in runs] == ["seed20261120", "seed20261121"]


def test_pooled_estimate_sums_draws_not_rates(tmp_path):
    seed_dirs(tmp_path)
    runs = SUMM.discover_seed_runs(tmp_path, EMPTY_OVERRIDES)
    analyses_by_run = [
        {arm: SUMM.analyse_arm(arm, run.base, run.paths[arm], seed=run.seed) for arm in SUMM.ARMS}
        for run in runs
    ]

    control_analyses = [run_analyses["control"] for run_analyses in analyses_by_run]
    # Per-seed: 1/4 = 25% and 5/16 = 31.25%.
    assert [analysis.eval_draws for analysis in control_analyses] == [4, 16]
    assert [analysis.eval_holds for analysis in control_analyses] == [1, 5]

    pooled = SUMM.pool_arm(control_analyses)
    assert pooled["runs"] == 2
    assert pooled["draws"] == 20
    assert pooled["holds"] == 6
    assert pooled["joint_success"] == pytest.approx(6 / 20)
    # The naive average of rates would be (0.25 + 0.3125) / 2 = 0.28125, which is
    # not the pooled proportion; the pooled row must be the draw-weighted one.
    assert pooled["joint_success"] != pytest.approx((0.25 + 0.3125) / 2)
    assert pooled["joint_ci"] == pytest.approx(SUMM.wilson_ci(6, 20))
    assert pooled["admission_rate"] == pytest.approx((2 + 10) / 20)

    # An arm present in only one seed pools only that seed (v1 covers A and B).
    penalty_pooled = SUMM.pool_arm([run_analyses["penalty"] for run_analyses in analyses_by_run])
    assert penalty_pooled["runs"] == 1
    assert penalty_pooled["draws"] == 16
    assert penalty_pooled["holds"] == 9


def test_report_prints_per_seed_rows_alongside_the_pooled_row(tmp_path):
    seed_dirs(tmp_path)
    out = tmp_path / "REPORT.md"
    completed = run_python(SUMMARIZE, ["--dir", str(tmp_path), "--out", str(out)])
    assert completed.returncode == 0, completed.stderr
    text = out.read_text(encoding="utf-8")

    seeds_section = text.split("## 7. Seeds", 1)[1].split("## What this does not show", 1)[0]
    assert "seed20261120 (seed 20261120)" in seeds_section
    assert "seed20261121 (seed 20261121)" in seeds_section
    assert "pooled (2 seeds)" in seeds_section
    assert "| pooled (2 seeds) | control |" in seeds_section
    # control: 1/4 and 5/16 rows, then 6/20 in the pooled row.
    assert "1/4 (25.0%)" in seeds_section
    assert "5/16 (31.2%)" in seeds_section
    assert "6/20 (30.0%)" in seeds_section
    # The v2-only arm pools one seed of its own, which is stated in the row label.
    assert "| pooled (1 seed) | penalty |" in seeds_section

    # The primary run is the one with the most arms (the v2 directory).
    assert "Primary run: `seed20261121` (seed 20261121)" in text
