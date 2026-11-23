#!/usr/bin/env python3
"""Offline tests for the rollout path, the dry-run CLIs and the report.

What is asserted, in order:

1. the strict-JSON prompt (and that it cannot leak the answer key);
2. the lexicographic 8/4 task split and its failure on a wrong-sized corpus;
3. admission -> execute -> grade -> reward, on real corpus tasks;
4. one group's records and determinism, with an injected ``generate_fn`` so no
   model is needed;
5. the grader cache (a repeat must not re-execute the patch);
6. ``train.py --dry-run`` for both arms: record counts, metrics fields, the
   no-step-for-zero-variance rule, and byte-for-byte determinism;
7. ``evaluate.py --dry-run``: 4 held-out tasks x N decodings, Wilson intervals;
8. ``analysis/summarize.py``: the statistics, the decision rule on synthetic
   held-out draws, the recomputation check, and the report from real dry-run
   artefacts.

Requirement: no network, no GPU, no model, no API key. Everything a model would
provide is injected. ``harness.tasks_io`` is forced onto the ``fork`` start
method inside this module: under pytest the parent's ``__main__`` is the pytest
console script, so the harness's default ``spawn`` would re-execute it in the
child. ``fork`` is the harness's own fallback path for callers whose main module
is not re-importable, so this exercises supported behaviour, not a stub.

Run with:  python -m pytest -q tests
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rollout  # noqa: E402  (also puts the frontier experiment on sys.path)
import grpo  # noqa: E402
import model_utils  # noqa: E402
import train as train_under_test  # noqa: E402  (import is side-effect free)
import harness.tasks_io as tasks_io  # noqa: E402

TRAIN = ROOT / "train.py"
EVALUATE = ROOT / "evaluate.py"
SUMMARIZE = ROOT / "analysis" / "summarize.py"

TASK_ID = "authz_delegated_write_01"


# --- execution model for the harness's child processes ----------------------


@pytest.fixture(autouse=True)
def _fork_case_workers(monkeypatch):
    """Run the harness's per-case workers with ``fork`` (see module docstring)."""

    monkeypatch.setattr(
        tasks_io, "_execution_context", lambda: multiprocessing.get_context("fork")
    )


# --- helpers ----------------------------------------------------------------


def load_summarize():
    spec = importlib.util.spec_from_file_location("summarize_under_test", SUMMARIZE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarize_under_test"] = module
    spec.loader.exec_module(module)
    return module


SUMM = load_summarize()


def task_by_id(task_id: str = TASK_ID) -> dict:
    return rollout.load_task_by_id(task_id)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_python(script: Path, args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONWARNINGS", None)
    return subprocess.run(
        [sys.executable, str(script)] + args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def fixed_generate(completions: list[str]):
    """A generate_fn that always returns ``completions`` (one per sample)."""

    def generate(prompt: str, n: int, seed: int):
        assert n == len(completions)
        return [rollout.Generation(text=text, prompt_tokens=11, completion_tokens=7) for text in completions]

    return generate


def reference_completion(task: dict) -> str:
    return json.dumps({"expression": task["reference_fix"]}, sort_keys=True)


# --- 1. prompt --------------------------------------------------------------


def test_prompt_states_the_strict_json_contract():
    prompt = rollout.build_prompt(task_by_id())
    lower = prompt.lower()

    assert "one json object" in lower
    assert "single string field" in lower
    assert "`expression`" in prompt
    assert "nothing else" in lower
    assert "markdown fence" in lower
    # The contract is the only thing that defines admission, so it must be explicit
    # about both what is required and what is forbidden.
    assert "do not add prose" in lower


def test_prompt_carries_the_task_material():
    task = task_by_id()
    prompt = rollout.build_prompt(task)

    assert task["id"] in prompt
    assert task["property"] in prompt
    assert task["source"] in prompt
    assert task["placeholder"] in prompt
    assert task["editable"]["path"] in prompt


def test_prompt_never_leaks_the_answer_key():
    for task in rollout.load_corpus():
        prompt = rollout.build_prompt(task)
        assert "reference_fix" not in prompt
        assert "seeded_bad" not in prompt
        assert "alternate_fixes" not in prompt
        secrets = [str(task["reference_fix"])]
        secrets += [str(item) for item in (task.get("alternate_fixes") or [])]
        secrets += [str(entry.get("expression")) for entry in (task.get("seeded_bad") or [])]
        for secret in secrets:
            if len(secret) >= 12:
                assert secret not in prompt, f"{task['id']}: prompt leaks {secret!r}"


def test_prompt_is_deterministic_and_task_specific():
    first = task_by_id("authz_delegated_write_01")
    second = task_by_id("bounds_content_length_01")

    assert rollout.build_prompt(first) == rollout.build_prompt(first)
    assert rollout.build_prompt(first) != rollout.build_prompt(second)


def test_decoding_constants_are_the_protocol_values():
    assert rollout.TEMPERATURE == 1.0
    assert rollout.TOP_P == 0.95
    assert rollout.MAX_NEW_TOKENS == 48
    assert rollout.G == 8


# --- 2. corpus split --------------------------------------------------------


def test_split_is_the_lexicographic_8_and_4():
    train, held_out = rollout.split_train_heldout()

    corpus_ids = [task["id"] for task in rollout.load_corpus()]
    assert corpus_ids == sorted(corpus_ids), "corpus must be read in lexicographic order"
    assert len(train) == 8
    assert len(held_out) == 4
    assert [task["id"] for task in train] == corpus_ids[:8]
    assert [task["id"] for task in held_out] == corpus_ids[8:]
    assert {task["id"] for task in train}.isdisjoint({task["id"] for task in held_out})
    assert len({task["id"] for task in train + held_out}) == 12


def test_a_wrong_sized_corpus_is_rejected(tmp_path):
    source = rollout.DEFAULT_TASKS_DIR
    corpus = tmp_path / "tasks"
    corpus.mkdir()
    for path in sorted(source.glob("*.json"))[:3]:
        (corpus / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(rollout.CorpusError, match="12"):
        rollout.split_train_heldout(corpus)


# --- 3. admission, grading and reward --------------------------------------


def test_reward_is_one_iff_admitted_and_holds():
    task = task_by_id()

    admitted = rollout.score_completion(task, reference_completion(task))
    assert admitted.admitted is True
    assert admitted.verdict == "HOLDS"
    assert admitted.reward == 1.0
    assert admitted.expression

    fenced = rollout.score_completion(task, "```json\n" + reference_completion(task) + "\n```")
    assert fenced.admitted is False
    assert fenced.expression is None
    assert fenced.verdict == "UNKNOWN"
    assert fenced.reward == 0.0
    assert "fence" in fenced.admission_reason.lower()

    prose = rollout.score_completion(task, "Sure! The fix is " + reference_completion(task))
    assert prose.admitted is False and prose.reward == 0.0

    extra_field = rollout.score_completion(
        task, json.dumps({"expression": task["reference_fix"], "confidence": 0.9})
    )
    assert extra_field.admitted is False
    assert extra_field.reward == 0.0

    unparseable = rollout.score_completion(task, json.dumps({"expression": "((("}))
    assert unparseable.admitted is True
    assert unparseable.verdict == "UNKNOWN"
    assert unparseable.reward == 0.0

    seeded = task["seeded_bad"][0]["expression"]
    violated = rollout.score_completion(task, json.dumps({"expression": seeded}))
    assert violated.admitted is True
    assert violated.verdict == "VIOLATED"
    assert violated.reward == 0.0
    assert violated.reasons


def test_score_completion_rejects_a_non_string():
    with pytest.raises(TypeError):
        rollout.score_completion(task_by_id(), None)


# --- 4. one group -----------------------------------------------------------


def test_run_group_records_every_required_field():
    task = task_by_id()
    completions = [reference_completion(task)] * 8

    group = rollout.run_group(
        task, fixed_generate(completions), group_index=3, g=8, seed=1, arm="control"
    )

    assert group.group_id == "g0003"
    assert group.group_index == 3
    assert group.task_id == task["id"]
    assert group.arm == "control"
    assert len(group.records) == 8
    assert group.rewards == [1.0] * 8
    assert group.valid_mask == [True] * 8

    required = {
        "task_id",
        "group_id",
        "group_index",
        "sample_index",
        "completion",
        "admitted",
        "verdict",
        "reward",
        "prompt_tokens",
        "completion_tokens",
    }
    for index, record in enumerate(group.records):
        assert required <= set(record)
        assert record["sample_index"] == index
        assert record["completion"] == completions[index]
        assert record["prompt_tokens"] == 11
        assert record["completion_tokens"] == 7
        assert isinstance(record["admitted"], bool)


def test_run_group_rejects_the_wrong_number_of_completions():
    task = task_by_id()

    def short_generate(prompt: str, n: int, seed: int):
        return ["{}"] * 3

    with pytest.raises(ValueError, match="expected 8"):
        rollout.run_group(task, short_generate, group_index=0, g=8)


def test_run_group_is_deterministic_for_the_same_seed():
    task = task_by_id()
    pool = rollout.dry_run_pool(task)

    def generate(prompt: str, n: int, seed: int):
        # A deterministic function of (n, seed) only, like a real sampler.
        return [pool[(seed + index) % len(pool)] for index in range(n)]

    first = rollout.run_group(task, generate, group_index=7, seed=42, arm="masked")
    second = rollout.run_group(task, generate, group_index=7, seed=42, arm="masked")
    assert first.records == second.records
    assert first.seed == second.seed


def test_group_seed_is_a_stable_function_of_its_inputs():
    base = rollout.group_seed(1, "task-a", 0)
    assert base == rollout.group_seed(1, "task-a", 0)
    assert base != rollout.group_seed(2, "task-a", 0)
    assert base != rollout.group_seed(1, "task-b", 0)
    assert base != rollout.group_seed(1, "task-a", 1)
    assert 0 <= base < 2**32


def test_masked_group_with_one_admitted_sample_has_no_effective_treatment():
    task = task_by_id()
    malformed = ["not json at all"] * 7 + [reference_completion(task)]
    group = rollout.run_group(
        task, fixed_generate(malformed), group_index=0, g=8, seed=0, arm="masked"
    )

    stats = grpo.group_stats(group.rewards, group.valid_mask, "masked")
    assert stats.n_valid == 1
    assert stats.effective is False
    # One admitted sample is not a population: the advantages are all zero, so the
    # admitted signal the secondary endpoint measures is exactly 0.0 (not None --
    # None is reserved for "nothing was admitted at all").
    assert stats.mean_abs_advantage_admitted == 0.0
    assert stats.n_used == 1
    # This is the condition train.py reports as a skipped group.
    assert stats.n_valid < 2


def test_grader_cache_serves_repeats_without_re_executing(monkeypatch):
    task = task_by_id()
    calls = []

    def fake_execute_patch(task_arg, expression):
        calls.append(expression)
        return tasks_io.ExecutionResult(
            compiled=True,
            security_passed=len(task_arg["security_cases"]),
            security_total=len(task_arg["security_cases"]),
            benign_passed=len(task_arg["benign_cases"]),
            benign_total=len(task_arg["benign_cases"]),
            detail=[],
            error=None,
        )

    monkeypatch.setattr(rollout, "execute_patch", fake_execute_patch)
    cache = rollout.GraderCache()
    completion = reference_completion(task)

    first = rollout.score_completion(task, completion, cache)
    second = rollout.score_completion(task, completion, cache)
    third = rollout.score_completion(task, completion, cache)

    assert first == second == third
    assert first.verdict == "HOLDS"
    assert len(calls) == 1, "the patch was re-executed despite the cache"
    assert cache.hits == 2
    assert len(cache) == 1


# --- 5. dry-run CLIs --------------------------------------------------------


def test_train_dry_run_control_writes_records_and_metrics(tmp_path):
    out = tmp_path / "rollouts_control.jsonl"
    metrics = tmp_path / "metrics_control.json"

    completed = run_python(
        TRAIN,
        [
            "--arm", "control",
            "--dry-run",
            "--effective-updates", "2",
            "--max-groups", "4",
            "--out", str(out),
            "--metrics", str(metrics),
        ],
    )
    assert completed.returncode == 0, completed.stderr

    records = read_jsonl(out)
    payload = json.loads(metrics.read_text(encoding="utf-8"))

    assert payload["arm"] == "control"
    assert payload["dry_run"] is True
    assert payload["status"] == "complete"
    assert payload["stop_reason"] in {"effective-target", "group-cap"}
    assert payload["model"]["requested_revision"] == "dry-run"
    assert payload["model"]["resolved_revision"] == "dry-run"
    assert len(payload["spec_sha256"]) == 64
    assert payload["budget"]["target_effective_updates"] == 2
    assert payload["budget"]["max_groups"] == 4
    assert len(payload["tasks"]["train"]) == 8
    assert len(payload["tasks"]["held_out"]) == 4
    assert len(payload["tasks"]["sha256"]) == 12

    counts = payload["counts"]
    assert counts["groups_sampled"] == len(payload["groups"])
    assert counts["groups_sampled"] <= 4
    assert counts["records_written"] == len(records)
    assert len(records) == counts["groups_sampled"] * 8
    assert counts["rollouts_consumed"] == len(records)
    assert counts["effective_updates"] <= 2
    assert counts["optimizer_steps"] == counts["effective_updates"]
    assert counts["groups_zero_variance"] == counts["groups_sampled"] - counts["effective_updates"]
    assert counts["groups_skipped"] == 0, "control never skips a group"

    # The arm treatment, checked against the per-group log.
    for group in payload["groups"]:
        assert group["task_id"] in payload["tasks"]["train"]
        if not group["effective"]:
            assert group["loss"] == 0.0
            assert group["optimizer_step"] is False
            assert group["zero_variance"] is True
        else:
            assert group["optimizer_step"] is True
            assert group["zero_variance"] is False
            assert group["loss"] != 0.0
            # A group that produced a gradient must have admitted samples whose
            # mean |advantage| is positive.
            assert group["mean_abs_advantage_admitted"] > 0.0

    assert len(records) > 0
    for record in records:
        assert set(record) >= {
            "task_id", "group_id", "group_index", "sample_index", "completion",
            "admitted", "verdict", "reward", "prompt_tokens", "completion_tokens",
            "sum_logprob",
        }
        assert record["reward"] == (1.0 if (record["admitted"] and record["verdict"] == "HOLDS") else 0.0)
        assert isinstance(record["sum_logprob"], (float, type(None)))


def test_train_dry_run_masked_arm_skips_without_a_step(tmp_path):
    out = tmp_path / "rollouts_masked.jsonl"
    metrics = tmp_path / "metrics_masked.json"

    completed = run_python(
        TRAIN,
        [
            "--arm", "masked",
            "--dry-run",
            "--effective-updates", "3",
            "--max-groups", "4",
            "--out", str(out),
            "--metrics", str(metrics),
        ],
    )
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(metrics.read_text(encoding="utf-8"))
    groups = payload["groups"]

    assert payload["arm"] == "masked"
    assert groups
    assert payload["counts"]["groups_skipped"] == sum(1 for group in groups if group["skipped"])
    for group in groups:
        assert group["skipped"] is (group["valid_count"] < 2)
        if group["skipped"]:
            assert group["effective"] is False
            assert group["loss"] == 0.0
        if not group["effective"]:
            assert group["optimizer_step"] is False
    # Budget parity is on effective steps: the cap can bind before the target.
    assert payload["counts"]["effective_updates"] <= 3
    assert payload["counts"]["shortfall_effective_updates"] == max(
        0, 3 - payload["counts"]["effective_updates"]
    )


def test_train_dry_run_is_byte_deterministic(tmp_path):
    def run(tag: str):
        out = tmp_path / f"rollouts_{tag}.jsonl"
        metrics = tmp_path / f"metrics_{tag}.json"
        completed = run_python(
            TRAIN,
            [
                "--arm", "control",
                "--dry-run",
                "--effective-updates", "1",
                "--max-groups", "3",
                "--out", str(out),
                "--metrics", str(metrics),
            ],
        )
        assert completed.returncode == 0, completed.stderr
        return out.read_bytes(), json.loads(metrics.read_text(encoding="utf-8"))

    first_bytes, first_metrics = run("a")
    second_bytes, second_metrics = run("b")

    assert first_bytes == second_bytes, "the dry run is not reproducible"
    assert first_metrics["counts"] == second_metrics["counts"]

    def without_timing(groups):
        return [{key: value for key, value in group.items() if key != "wall_time_s"} for group in groups]

    assert without_timing(first_metrics["groups"]) == without_timing(second_metrics["groups"])


def test_train_rejects_invalid_budgets(tmp_path):
    for extra, needle in (
        (["--g", "1"], "--g"),
        (["--effective-updates", "0"], "--effective-updates"),
        (["--max-groups", "0"], "--max-groups"),
        (["--prompts-per-update", "0"], "--prompts-per-update"),
    ):
        completed = run_python(
            TRAIN,
            ["--arm", "control", "--dry-run", "--out", str(tmp_path / "x.jsonl"),
             "--metrics", str(tmp_path / "x.json")] + extra,
        )
        assert completed.returncode != 0, extra
        assert needle in (completed.stderr + completed.stdout), extra


def test_train_aborts_after_two_oom_failures(tmp_path, monkeypatch):
    """The SPEC's stopping rule: OOM twice aborts the arm, and says so in the metrics."""

    def oom(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(model_utils.DryRunModel, "update", oom)

    metrics_path = tmp_path / "metrics.json"
    rc = train_under_test.main(
        [
            "--arm", "control",
            "--dry-run",
            "--effective-updates", "5",
            "--max-groups", "16",
            "--out", str(tmp_path / "rollouts.jsonl"),
            "--metrics", str(metrics_path),
        ]
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert rc == 3
    assert payload["status"] == "aborted"
    assert payload["abort_reason"] == "oom"
    assert payload["stop_reason"] == "oom-abort"
    assert payload["counts"]["groups_errored"] == train_under_test.MAX_OOM
    assert payload["counts"]["effective_updates"] == 0
    # Every failed group is logged with its exception text.
    errors = [group for group in payload["groups"] if group.get("error")]
    assert len(errors) == train_under_test.MAX_OOM
    assert all("out of memory" in group["error"].lower() for group in errors)


def test_train_aborts_after_ten_consecutive_exceptions(tmp_path, monkeypatch):
    """> 10 consecutive group exceptions aborts the arm (SPEC stopping rule)."""

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic group failure")

    monkeypatch.setattr(train_under_test.rollout, "run_group", explode)

    metrics_path = tmp_path / "metrics.json"
    rc = train_under_test.main(
        [
            "--arm", "masked",
            "--dry-run",
            "--effective-updates", "5",
            "--max-groups", "40",
            "--out", str(tmp_path / "rollouts.jsonl"),
            "--metrics", str(metrics_path),
        ]
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert rc == 3
    assert payload["status"] == "aborted"
    assert payload["abort_reason"] == "consecutive-exceptions"
    assert payload["counts"]["groups_errored"] == train_under_test.MAX_CONSECUTIVE_EXCEPTIONS + 1
    assert payload["counts"]["records_written"] == 0
    assert payload["counts"]["rollouts_consumed"] == 0


def test_train_aborts_on_the_wall_clock(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    rc = train_under_test.main(
        [
            "--arm", "control",
            "--dry-run",
            "--effective-updates", "24",
            "--max-groups", "96",
            "--max-minutes", "0.0001",  # ~6 ms: the first group exceeds it
            "--out", str(tmp_path / "rollouts.jsonl"),
            "--metrics", str(metrics_path),
        ]
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert rc == 3
    assert payload["status"] == "aborted"
    assert payload["abort_reason"] == "wall-time-exceeded"
    assert payload["counts"]["groups_sampled"] < 96


def test_train_help_documents_the_budget_flags():
    completed = run_python(TRAIN, ["--help"])
    assert completed.returncode == 0
    for flag in ("--arm", "--effective-updates", "--max-groups", "--prompts-per-update",
                 "--dry-run", "--max-minutes", "--model-revision", "--hourly-rate",
                 "--local-files-only"):
        assert flag in completed.stdout, flag
    # The old rollout-budget flag is gone by design: parity is on gradient steps.
    assert "--updates" not in completed.stdout


def test_evaluate_dry_run_writes_the_held_out_endpoint(tmp_path):
    out = tmp_path / "eval_control.json"
    completed = run_python(
        EVALUATE,
        ["--arm", "control", "--dry-run", "--decodings", "2", "--out", str(out)],
    )
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["arm"] == "control"
    assert payload["dry_run"] is True
    assert len(payload["held_out_tasks"]) == 4
    assert [task["id"] for task in rollout.split_train_heldout()[1]] == payload["held_out_tasks"]

    records = payload["records"]
    assert len(records) == 4 * 2 == payload["endpoints"]["draws"]
    assert payload["counts"]["draws"] == len(records)
    assert payload["counts"]["admitted"] == sum(1 for record in records if record["admitted"])
    assert payload["counts"]["holds"] == sum(1 for record in records if record["verdict"] == "HOLDS")
    assert len(payload["per_task"]) == 4

    low, high = payload["endpoints"]["joint_success_wilson95"]
    assert 0.0 <= low <= payload["endpoints"]["joint_success"] <= high <= 1.0
    low, high = payload["endpoints"]["admission_rate_wilson95"]
    assert 0.0 <= low <= payload["endpoints"]["admission_rate"] <= high <= 1.0


def test_evaluate_real_mode_requires_an_adapter(tmp_path):
    completed = run_python(
        EVALUATE, ["--arm", "control", "--out", str(tmp_path / "eval.json")]
    )
    assert completed.returncode != 0
    assert "adapter" in (completed.stderr + completed.stdout).lower()


# --- 6. the report ---------------------------------------------------------


@pytest.mark.parametrize(
    "successes,n,low,high",
    [
        (5, 20, 0.111868, 0.468705),
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (16, 32, 0.336306, 0.663694),
    ],
)
def test_wilson_ci_matches_reference(successes, n, low, high):
    got_low, got_high = SUMM.wilson_ci(successes, n)
    assert got_low == pytest.approx(low, abs=5e-5)
    assert got_high == pytest.approx(high, abs=5e-5)


def test_wilson_ci_handles_an_empty_sample():
    assert SUMM.wilson_ci(0, 0) == (0.0, 0.0)
    assert SUMM.wilson_ci(3, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    "b,c,expected",
    [(0, 0, 1.0), (10, 0, 0.001953125), (1, 4, 0.375), (7, 1, 0.0703125), (3, 3, 1.0)],
)
def test_paired_sign_test_matches_reference(b, c, expected):
    assert SUMM.paired_sign_test_p(b, c) == pytest.approx(expected, abs=1e-12)
    assert SUMM.paired_sign_test_p(b, c) == SUMM.paired_sign_test_p(c, b)


def synthetic_arm(arm: str, holds: int, draws: int, admitted: int) -> "SUMM.ArmAnalysis":
    records = []
    for index in range(draws):
        if index < holds:
            verdict, is_admitted = "HOLDS", True
        elif index < admitted:
            verdict, is_admitted = "VIOLATED", True
        else:
            verdict, is_admitted = "UNKNOWN", False
        records.append(
            {
                "task_id": "held_out_task",
                "decoding": index,
                "admitted": is_admitted,
                "verdict": verdict,
                "reward": 1.0 if verdict == "HOLDS" else 0.0,
            }
        )
    return SUMM.ArmAnalysis(
        arm=arm,
        rollouts_path=Path("/nonexistent/rollouts.jsonl"),
        metrics_path=Path("/nonexistent/metrics.json"),
        eval_path=Path("/nonexistent/eval.json"),
        eval_payload={"records": records, "dry_run": True},
    )


def test_decision_rule_reports_masked_better_on_a_large_non_overlapping_gap():
    decision = SUMM.decide(
        synthetic_arm("masked", holds=16, draws=32, admitted=24),
        synthetic_arm("control", holds=4, draws=32, admitted=20),
    )
    assert decision.diff == pytest.approx(0.375)
    assert decision.threshold_met is True
    assert decision.non_overlapping is True
    assert decision.verdict == "masked-better"
    assert "MASKED BETTER" in decision.readable


def test_decision_rule_reports_control_better_in_the_other_direction():
    decision = SUMM.decide(
        synthetic_arm("masked", holds=4, draws=32, admitted=20),
        synthetic_arm("control", holds=16, draws=32, admitted=24),
    )
    assert decision.diff == pytest.approx(-0.375)
    assert decision.verdict == "control-better"


def test_decision_rule_reports_no_measurable_difference_below_the_threshold():
    decision = SUMM.decide(
        synthetic_arm("masked", holds=12, draws=32, admitted=24),
        synthetic_arm("control", holds=10, draws=32, admitted=25),
    )
    assert decision.threshold_met is False
    assert decision.verdict == "no-measurable-difference"


def test_decision_rule_requires_both_conditions():
    """Neither condition is sufficient on its own.

    A 12.5-point gap (1/8 vs 0/8) clears the margin but its Wilson intervals
    overlap.  So does 14/32 vs 10/32 at the protocol's own n.  In both cases the
    rule must refuse to call a difference.  (The converse -- disjoint intervals on
    a gap below 12 points -- needs a sample size far beyond the protocol's 32 per
    arm, which is exactly why non-overlap is the binding constraint.)
    """

    decision = SUMM.decide(
        synthetic_arm("masked", holds=1, draws=8, admitted=8),
        synthetic_arm("control", holds=0, draws=8, admitted=6),
    )
    assert decision.threshold_met is True
    assert decision.non_overlapping is False
    assert decision.verdict == "no-measurable-difference"

    decision = SUMM.decide(
        synthetic_arm("masked", holds=14, draws=32, admitted=24),
        synthetic_arm("control", holds=10, draws=32, admitted=26),
    )
    assert decision.diff == pytest.approx(0.125)
    assert decision.threshold_met is True
    assert decision.non_overlapping is False
    assert decision.verdict == "no-measurable-difference"


def test_decision_rule_without_held_out_data_is_not_established():
    decision = SUMM.decide(
        SUMM.ArmAnalysis(
            arm="masked", rollouts_path=Path("/x"), metrics_path=Path("/y"), eval_path=Path("/z")
        ),
        SUMM.ArmAnalysis(
            arm="control", rollouts_path=Path("/x"), metrics_path=Path("/y"), eval_path=Path("/z")
        ),
    )
    assert decision.verdict == "not-established"
    assert decision.diff is None


def test_summarize_survives_completely_missing_inputs(tmp_path):
    out = tmp_path / "REPORT.md"
    completed = run_python(SUMMARIZE, ["--dir", str(tmp_path / "nothing"), "--out", str(out)])
    assert completed.returncode == 0, completed.stderr

    text = out.read_text(encoding="utf-8")
    for heading in (
        "## 1. Inputs",
        "## 2. Co-primary endpoints",
        "## 3. Decision rule",
        "## 4. Secondary endpoints",
        "## What this does not show",
        "## Provenance",
    ):
        assert heading in text, heading
    assert "not found" in text
    assert "NOT ESTABLISHED" in text


# --- 7. end-to-end dry run: two arms, two evals, one report -----------------


@pytest.fixture(scope="session")
def dry_artifacts(tmp_path_factory):
    work = tmp_path_factory.mktemp("dryrun_report")

    for arm in ("control", "masked"):
        trained = run_python(
            TRAIN,
            [
                "--arm", arm,
                "--dry-run",
                "--effective-updates", "2",
                "--max-groups", "4",
                "--out", str(work / f"rollouts_{arm}.jsonl"),
                "--metrics", str(work / f"metrics_{arm}.json"),
            ],
        )
        assert trained.returncode == 0, trained.stderr
        evaluated = run_python(
            EVALUATE,
            ["--arm", arm, "--dry-run", "--decodings", "2", "--out", str(work / f"eval_{arm}.json")],
        )
        assert evaluated.returncode == 0, evaluated.stderr

    report = work / "REPORT.md"
    completed = run_python(SUMMARIZE, ["--dir", str(work), "--out", str(report)])
    assert completed.returncode == 0, completed.stderr
    return {"dir": work, "report": report}


def test_report_has_the_required_sections(dry_artifacts):
    text = dry_artifacts["report"].read_text(encoding="utf-8")

    assert text.startswith("# Invalid-action masking in GRPO")
    assert "DRY RUN — NOT EVIDENCE" in text
    for heading in (
        "## 1. Inputs",
        "## 2. Co-primary endpoints",
        "## 3. Decision rule",
        "### 3.1 Cost",
        "## 4. Secondary endpoints",
        "## 5. Paired per-group variance",
        "## 6. Counts, denominators and the audit trail",
        "## What this does not show",
        "## Provenance",
    ):
        assert heading in text, heading
    assert "MASKED BETTER" in text or "CONTROL BETTER" in text or "NO MEASURABLE DIFFERENCE" in text
    assert "rollout-cost ratio" in text.lower()
    assert "%" in text


def test_report_numbers_are_recomputed_from_the_records(dry_artifacts):
    work = dry_artifacts["dir"]
    for arm in ("control", "masked"):
        analysis = SUMM.analyse_arm(
            arm, work, {"rollouts": None, "metrics": None, "eval": None}
        )
        records = read_jsonl(work / f"rollouts_{arm}.jsonl")
        assert analysis.records == len(records)
        assert analysis.groups_sampled == len({record["group_index"] for record in records})
        assert analysis.rollouts_consumed == len(records)
        assert analysis.effective_updates == sum(
            1 for group in analysis.groups if group.effective
        )
        # The metrics file agrees with the recomputation: no integrity problem.
        assert analysis.discrepancies == [], analysis.discrepancies


def test_report_pairing_only_uses_comparable_groups(dry_artifacts):
    work = dry_artifacts["dir"]
    masked = SUMM.analyse_arm("masked", work, {"rollouts": None, "metrics": None, "eval": None})
    control = SUMM.analyse_arm("control", work, {"rollouts": None, "metrics": None, "eval": None})

    pairs = SUMM.paired_groups(masked, control)
    for masked_group, control_group in pairs:
        assert masked_group.group_index == control_group.group_index
        assert masked_group.task_id == control_group.task_id


def test_report_flags_a_protocol_hash_mismatch(tmp_path):
    """A post-hoc SPEC edit must be visible at the top of the report."""

    rollouts = tmp_path / "rollouts_control.jsonl"
    metrics = tmp_path / "metrics_control.json"
    rollouts.write_text("", encoding="utf-8")
    metrics.write_text(
        json.dumps({"arm": "control", "spec_sha256": "0" * 64, "dry_run": True, "counts": {}}),
        encoding="utf-8",
    )
    out = tmp_path / "REPORT.md"
    completed = run_python(SUMMARIZE, ["--dir", str(tmp_path), "--out", str(out)])
    assert completed.returncode == 0, completed.stderr
    assert "PROTOCOL HASH MISMATCH" in out.read_text(encoding="utf-8")
