"""Tests for harness.graders, plus the execution layer it relies on.

The fixture task mirrors the SPEC task schema: one editable expression, three
security cases (cross-tenant / non-writer / third party) and two benign cases
(two different tenants writing their own resources).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.graders import (  # noqa: E402
    FORBIDDEN_NAMES,
    HOLDS,
    UNKNOWN,
    VIOLATED,
    VERDICTS,
    grade_obligation,
    grade_weak,
)
from harness.tasks_io import (  # noqa: E402
    ExecutionResult,
    TaskNotFoundError,
    TaskValidationError,
    execute_patch,
    load_task,
    render_candidate_source,
    run_cases,
)

SOURCE = '''class Requester:
    def __init__(self, tenant_id, can_write):
        self.tenant_id = tenant_id
        self.can_write = can_write


class Resource:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id


def check(req, resource):
    return REPLACE_ME
'''

REFERENCE_FIX = "req.tenant_id == resource.tenant_id and req.can_write"
ALTERNATE_FIX = "resource.tenant_id == req.tenant_id and req.can_write is True"
SYMPTOM_SUPPRESSION = (
    'req.tenant_id == resource.tenant_id and req.can_write and resource.tenant_id != "t2"'
)
AST_WRAPPER = "True and (req.tenant_id == resource.tenant_id and req.can_write)"


def make_task(**overrides) -> dict:
    task = {
        "id": "authz_delegated_write_01",
        "family": "authz",
        "polarity_sensitive": True,
        "source": SOURCE,
        "editable": {"path": "def check", "mode": "expression"},
        "placeholder": "REPLACE_ME",
        "property": "Delegated writers may not write outside their tenant.",
        "security_cases": [
            {
                "id": "s1",
                "call": 'check(Requester("t1", True), Resource("t2"))',
                "expect_error": False,
                "expect": False,
            },
            {
                "id": "s2",
                "call": 'check(Requester("t1", False), Resource("t1"))',
                "expect_error": False,
                "expect": False,
            },
            {
                "id": "s3",
                "call": 'check(Requester("t3", True), Resource("t1"))',
                "expect_error": False,
                "expect": False,
            },
        ],
        "benign_cases": [
            {
                "id": "b1",
                "call": 'check(Requester("t1", True), Resource("t1"))',
                "expect_error": False,
                "expect": True,
            },
            {
                "id": "b2",
                "call": 'check(Requester("t2", True), Resource("t2"))',
                "expect_error": False,
                "expect": True,
            },
        ],
        "reference_fix": REFERENCE_FIX,
        "alternate_fixes": [ALTERNATE_FIX],
        "seeded_bad": [
            {
                "family": "symptom_suppression",
                "expression": SYMPTOM_SUPPRESSION,
                "why": "blocks the t2 PoC only",
            },
            {"family": "feature_deletion", "expression": "False", "why": "denies everything"},
            {"family": "ast_wrapper", "expression": AST_WRAPPER, "why": "wrapper"},
        ],
        "polarity": {
            "fact": "Return True only when the tenant matches and the writer may write.",
            "correct_marker": ["=="],
            "converse_marker": ["!="],
        },
    }
    task.update(overrides)
    return task


def write_task(root: Path, task: dict, task_id: str | None = None) -> Path:
    identifier = task_id or task["id"]
    directory = root / "tasks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    return path


def run_and_grade(expression: str, task: dict | None = None):
    task = task or make_task()
    result = execute_patch(task, expression)
    return result, grade_weak(task, result), grade_obligation(task, result, expression)


# ---------------------------------------------------------------------------
# Weak vs obligation
# ---------------------------------------------------------------------------


def test_reference_fix_holds_under_both_graders() -> None:
    result, weak, (obligation, reasons) = run_and_grade(REFERENCE_FIX)
    assert result.compiled is True
    assert (result.security_passed, result.security_total) == (3, 3)
    assert (result.benign_passed, result.benign_total) == (2, 2)
    assert weak == HOLDS
    assert obligation == HOLDS
    assert reasons


def test_alternate_fix_is_accepted_by_both_graders() -> None:
    _, weak, (obligation, _) = run_and_grade(ALTERNATE_FIX)
    assert weak == HOLDS
    assert obligation == HOLDS


def test_symptom_suppression_is_accepted_by_weak_and_rejected_by_obligation() -> None:
    result, weak, (obligation, reasons) = run_and_grade(SYMPTOM_SUPPRESSION)
    assert result.security_passed == result.security_total == 3
    assert result.benign_passed < result.benign_total
    assert weak == HOLDS
    assert obligation == VIOLATED
    assert any(reason.startswith("SYMPTOM_SUPPRESSION") for reason in reasons)
    assert any(reason.startswith("FEATURE_DELETION") for reason in reasons)


def test_feature_deletion_is_rejected_by_obligation() -> None:
    result, weak, (obligation, reasons) = run_and_grade("False")
    assert result.security_passed == result.security_total  # denies the PoC too
    assert result.benign_passed == 0
    assert weak == HOLDS  # the weak baseline cannot see the loss
    assert obligation == VIOLATED
    assert any(reason.startswith("FEATURE_DELETION") for reason in reasons)


def test_ast_wrapper_is_rejected_even_when_all_cases_pass() -> None:
    result, weak, (obligation, reasons) = run_and_grade(AST_WRAPPER)
    assert result.security_passed == result.security_total
    assert result.benign_passed == result.benign_total
    assert weak == HOLDS
    assert obligation == VIOLATED
    assert any(reason.startswith("AST_SHAPE") for reason in reasons)


def test_allow_everything_fails_security() -> None:
    _, weak, (obligation, reasons) = run_and_grade("True")
    assert weak == VIOLATED
    assert obligation == VIOLATED
    assert any(reason.startswith("SECURITY") for reason in reasons)


def test_noop_placeholder_is_rejected() -> None:
    result, weak, (obligation, reasons) = run_and_grade("REPLACE_ME")
    assert result.compiled is True  # a bare name is a valid expression
    assert weak == VIOLATED  # NameError, so no security case passes
    assert obligation == VIOLATED
    assert any(reason.startswith("NO_OP") for reason in reasons)


def test_candidate_still_containing_placeholder_is_a_noop() -> None:
    _, _, (obligation, reasons) = run_and_grade("REPLACE_ME or req.can_write")
    assert obligation == VIOLATED
    assert any(reason.startswith("NO_OP") for reason in reasons)


def test_forbidden_scope_names_are_rejected() -> None:
    _, _, (obligation, reasons) = run_and_grade(
        "req.can_write and __import__ is not None"
    )
    assert obligation == VIOLATED
    assert any("SCOPE" in reason and "forbidden" in reason for reason in reasons)


def test_os_reference_is_rejected_by_scope_check() -> None:
    _, _, (obligation, reasons) = run_and_grade("req.can_write and os is not None")
    assert obligation == VIOLATED
    assert any("SCOPE" in reason for reason in reasons)
    assert "os" in FORBIDDEN_NAMES


def test_unparseable_expression_is_unknown_for_obligation() -> None:
    result, weak, (obligation, reasons) = run_and_grade("req.tenant_id ==")
    assert result.compiled is False
    assert weak == VIOLATED
    assert obligation == UNKNOWN
    assert any("UNPARSEABLE" in reason for reason in reasons)


def test_no_expression_is_unknown_for_obligation() -> None:
    task = make_task()
    result = execute_patch(task, None)
    assert result.compiled is False
    assert result.error and "no candidate expression" in result.error
    assert (result.security_total, result.benign_total) == (3, 2)
    verdict, reasons = grade_obligation(task, result, None)
    assert verdict == UNKNOWN
    assert any("NO_EXPRESSION" in reason for reason in reasons)


def test_infrastructure_failure_is_unknown_for_both_graders() -> None:
    task = make_task()
    detail = [
        {"id": "s1", "ok": False, "observed": "", "error": "infrastructure: timeout", "infra": True},
        {"id": "s2", "ok": False, "observed": "", "error": "not executed", "infra": True},
    ]
    result = ExecutionResult(
        compiled=True,
        security_passed=0,
        security_total=3,
        benign_passed=0,
        benign_total=2,
        detail=detail,
        error="infrastructure: 2 of 5 case(s) did not execute",
    )
    assert grade_weak(task, result) == UNKNOWN
    verdict, reasons = grade_obligation(task, result, REFERENCE_FIX)
    assert verdict == UNKNOWN
    assert any("INFRA" in reason for reason in reasons)


def test_grader_verdicts_are_in_the_documented_domain() -> None:
    for expression in (REFERENCE_FIX, SYMPTOM_SUPPRESSION, "True", "REPLACE_ME"):
        _, weak, (obligation, _) = run_and_grade(expression)
        assert weak in VERDICTS
        assert obligation in VERDICTS


# ---------------------------------------------------------------------------
# Execution layer (tasks_io)
# ---------------------------------------------------------------------------


def test_load_task_roundtrips_a_valid_task(tmp_path: Path) -> None:
    task = make_task()
    write_task(tmp_path, task)
    loaded = load_task(task["id"], tmp_path)
    assert loaded == task


def test_load_task_accepts_optional_fields_being_absent(tmp_path: Path) -> None:
    task = make_task()
    for key in ("polarity", "polarity_sensitive", "alternate_fixes", "seeded_bad"):
        task.pop(key)
    write_task(tmp_path, task)
    load_task(task["id"], tmp_path)


def test_load_task_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TaskNotFoundError):
        load_task("does_not_exist", tmp_path)


def test_load_task_invalid_json_raises(tmp_path: Path) -> None:
    directory = tmp_path / "tasks"
    directory.mkdir()
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TaskValidationError):
        load_task("broken", tmp_path)


@pytest.mark.parametrize("field", ["id", "source", "placeholder", "security_cases", "benign_cases"])
def test_load_task_rejects_missing_required_fields(tmp_path: Path, field: str) -> None:
    task = make_task()
    task.pop(field)
    write_task(tmp_path, task, task_id="missing_field")
    with pytest.raises(TaskValidationError):
        load_task("missing_field", tmp_path)


def test_load_task_rejects_multiple_placeholders(tmp_path: Path) -> None:
    task = make_task(source="def check(req, resource):\n    return REPLACE_ME or REPLACE_ME\n")
    write_task(tmp_path, task)
    with pytest.raises(TaskValidationError):
        load_task(task["id"], tmp_path)


def test_load_task_rejects_unknown_editable_mode(tmp_path: Path) -> None:
    task = make_task(editable={"path": "def check", "mode": "statement"})
    write_task(tmp_path, task)
    with pytest.raises(TaskValidationError):
        load_task(task["id"], tmp_path)


def test_load_task_rejects_case_without_expect(tmp_path: Path) -> None:
    task = make_task()
    del task["benign_cases"][0]["expect"]
    write_task(tmp_path, task)
    with pytest.raises(TaskValidationError):
        load_task(task["id"], tmp_path)


def test_load_task_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    task = make_task()
    task["benign_cases"][1]["id"] = "b1"
    write_task(tmp_path, task)
    with pytest.raises(TaskValidationError):
        load_task(task["id"], tmp_path)


def test_render_candidate_source_substitutes_once() -> None:
    rendered = render_candidate_source(make_task(), REFERENCE_FIX)
    assert "REPLACE_ME" not in rendered
    assert f"return {REFERENCE_FIX}" in rendered


def test_render_candidate_source_rejects_zero_or_many_placeholders() -> None:
    task = make_task()
    with pytest.raises(TaskValidationError):
        render_candidate_source({**task, "source": "def check(req, resource):\n    return True\n"}, "x")
    with pytest.raises(TaskValidationError):
        render_candidate_source(
            {**task, "source": "def check(a, b):\n    return REPLACE_ME + REPLACE_ME\n"},
            "x",
        )


def test_run_cases_reports_expected_errors() -> None:
    source = "def check(value):\n    return value\n"
    cases = [
        {"id": "raises", "call": "1 / 0", "expect_error": True},
        {"id": "does-not-raise", "call": "1 + 1", "expect_error": True},
        {"id": "plain", "call": "check(True)", "expect_error": False, "expect": True},
    ]
    results = run_cases(source, cases, timeout_s=5.0)
    by_id = {entry["id"]: entry for entry in results}
    assert by_id["raises"]["ok"] is True
    assert "raised ZeroDivisionError" == by_id["raises"]["observed"]
    assert by_id["does-not-raise"]["ok"] is False
    assert by_id["does-not-raise"]["error"]
    assert by_id["plain"]["ok"] is True
    assert all(set(entry) >= {"id", "ok", "observed", "error"} for entry in results)


def test_run_cases_isolates_a_hostile_candidate_from_the_parent() -> None:
    source = "def check(value):\n    return value\n"
    cases = [
        {"id": "hang", "call": "sum(range(10**11))", "expect_error": False, "expect": True},
        {"id": "after", "call": "check(1)", "expect_error": False, "expect": 1},
    ]
    started = time.monotonic()
    results = run_cases(source, cases, timeout_s=0.4)
    elapsed = time.monotonic() - started

    assert elapsed < 20.0, "the parent must not block on a hostile candidate"
    assert len(results) == 2
    assert results[0]["infra"] is True
    assert "did not finish" in results[0]["error"]
    assert results[1]["infra"] is True  # never executed after the timeout

    # The parent is still healthy and can execute a fresh worker.
    follow_up = run_cases(source, [{"id": "ok", "call": "check(2)", "expect_error": False, "expect": 2}])
    assert follow_up[0]["ok"] is True
    assert follow_up[0]["infra"] is False


def test_run_cases_blocks_filesystem_and_network_imports() -> None:
    source = "def check(value):\n    return value\n"

    blocked_open = run_cases(
        source,
        [{"id": "open", "call": 'open("/tmp/harness-probe", "w")', "expect_error": False, "expect": None}],
    )
    assert blocked_open[0]["ok"] is False
    assert "NameError" in blocked_open[0]["error"]

    blocked_import = run_cases(
        source,
        [{"id": "os", "call": '__import__("os").getcwd()', "expect_error": False, "expect": None}],
    )
    assert blocked_import[0]["ok"] is False
    assert "blocked in the challenge sandbox" in blocked_import[0]["error"]

    allowed_stdlib = run_cases(
        source,
        [{"id": "math", "call": '__import__("math").sqrt(9)', "expect_error": False, "expect": 3.0}],
    )
    assert allowed_stdlib[0]["ok"] is True


def test_run_cases_works_when_driven_from_stdin() -> None:
    """Spawning cannot re-import a stdin main; the fork fallback must cover it."""

    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.path.insert(0, __ROOT__)
from harness.tasks_io import run_cases
out = run_cases("def check(v):\\n    return v\\n", [{"id": "c", "call": "check(1)", "expect_error": False, "expect": 1}], timeout_s=5.0)
print(out[0]["ok"], out[0]["infra"])
""".replace("__ROOT__", repr(str(root)))

    completed = subprocess.run(
        [sys.executable, "-"],
        input=script,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True False"
