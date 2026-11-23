"""Offline validator for the task corpus of the 2026-11-20 frontier re-experiment.

Every assertion here is static or in-process: no network, no filesystem writes, no
model APIs, no clock. The corpus is ``tasks/<id>.json`` and the schema is defined
by ``docs/SPEC.md``.

What is checked, per task
-------------------------
* the schema (keys, types, ranges), the single ``REPLACE_ME`` occurrence sitting in
  one ``return`` expression, the family vocabulary, and the case counts;
* the module loads through ``exec`` in a fresh namespace;
* ``reference_fix`` passes 100% of the security and benign cases, and returns real
  booleans rather than truthy stand-ins;
* every ``alternate_fixes`` entry also passes 100% of both lists;
* ``seeded_bad`` carries exactly the four seeded-bad families, each entry fails at
  least one case, and each entry behaves the way its family claims;
* polarity-sensitive tasks declare a fact plus non-empty ``correct_marker`` and
  ``converse_marker`` lists, the reference matches the correct side, and the
  reference's syntactic converse (built by deterministic token substitution)
  matches the converse side. When that cannot be decided mechanically the task is
  marked ``UNKNOWN`` and that one assertion is skipped, per docs/SPEC.md conservatism;
  ``test_polarity_summary`` requires that the corpus never actually needs the
  escape hatch;
* the reference is deterministic: two runs observe identical results.

The seeded-bad failures are recorded in ``SEEDED_BAD_FINDINGS`` so the findings can
be reported without re-running the corpus.

Run from the experiment root::

    python -m pytest tests/test_tasks.py -q
"""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Corpus constants
# ---------------------------------------------------------------------------

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = EXPERIMENT_ROOT / "tasks"

PLACEHOLDER = "REPLACE_ME"

EXPECTED_TASKS = 12
FAMILIES = (
    "authz",
    "containment",
    "cookie",
    "bounds",
    "canonicalization",
    "state_drift",
)
MIN_FAMILIES = 4
MIN_POLARITY_SENSITIVE = 3

SEEDED_BAD_FAMILIES = (
    "symptom_suppression",
    "guard_too_weak",
    "feature_deletion",
    "ast_wrapper",
)

CASE_MIN, CASE_MAX = 6, 10
SOURCE_MIN_LINES, SOURCE_MAX_LINES = 8, 25

REQUIRED_FIELDS: dict[str, type] = {
    "id": str,
    "family": str,
    "polarity_sensitive": bool,
    "source": str,
    "editable": dict,
    "placeholder": str,
    "property": str,
    "security_cases": list,
    "benign_cases": list,
    "reference_fix": str,
    "alternate_fixes": list,
    "seeded_bad": list,
}
OPTIONAL_FIELDS = {"polarity", "polarity_sensitive", "adversarial"}
CASE_FIELDS = {"id", "call", "expect_error", "expect"}

#: A challenge module may only import deterministic stdlib helpers. Everything a
#: candidate could use to reach the network, the filesystem, the clock or the
#: interpreter itself is refused.
ALLOWED_MODULE_IMPORTS = frozenset(
    {
        "collections",
        "functools",
        "ipaddress",
        "itertools",
        "math",
        "operator",
        "posixpath",
        "re",
        "string",
        "unicodedata",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
        "asyncio",
        "multiprocessing",
        "os",
        "pathlib",
        "random",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "threading",
        "time",
        "urllib",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "__class__",
        "__code__",
        "__dict__",
        "__globals__",
        "__subclasses__",
        "now",
        "time",
    }
)

#: Deterministic inverses used to build the syntactic converse of a reference.
INVERSE_MARKERS: dict[str, str] = {
    "==": "!=",
    "!=": "==",
    "<": ">=",
    ">=": "<",
    "<=": ">",
    ">": "<=",
    "is None": "is not None",
    "is not None": "is None",
    "is True": "is False",
    "is False": "is True",
    "startswith": "endswith",
    "endswith": "startswith",
    "in": "not in",
    "not in": "in",
}

#: Populated by test_seeded_bad_entries: task id -> family -> failing case ids.
SEEDED_BAD_FINDINGS: dict[str, dict[str, list[str]]] = {}
#: Populated by test_polarity_markers: task id -> PASS | UNKNOWN.
POLARITY_STATUS: dict[str, str] = {}

_WORDY_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*$")
_AST_WRAPPER = re.compile(r"^True\s+and\s*\(")

TASK_IDS: list[str] = sorted(path.stem for path in TASKS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_RAW_TASKS: dict[str, dict] = {}


def load_task(task_id: str) -> dict:
    """Load ``tasks/<task_id>.json`` once and cache it."""

    if task_id not in _RAW_TASKS:
        path = TASKS_DIR / f"{task_id}.json"
        assert path.is_file(), f"missing task file: {path}"
        try:
            _RAW_TASKS[task_id] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            pytest.fail(f"{path} is not valid JSON: {exc}")
    return _RAW_TASKS[task_id]


def all_tasks() -> list[dict]:
    return [load_task(task_id) for task_id in TASK_IDS]


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def render_source(source: str, expression: str) -> str:
    """Substitute the single placeholder, exactly the way a patch is applied."""

    assert source.count(PLACEHOLDER) == 1, "source must hold exactly one placeholder"
    assert isinstance(expression, str) and expression.strip(), "candidate is empty"
    return source.replace(PLACEHOLDER, expression)


def exec_module(source: str) -> dict:
    """Execute a rendered challenge module in a fresh namespace."""

    namespace: dict = {"__name__": "challenge_module"}
    exec(compile(source, "<challenge_module>", "exec"), namespace)  # noqa: S102
    return namespace


def _case_result(case: dict, namespace: dict) -> dict:
    call = case["call"]
    expect_error = bool(case.get("expect_error", False))
    expect = case.get("expect")
    try:
        value = eval(compile(call, "<case>", "eval"), namespace)  # noqa: S307
    except Exception as exc:  # noqa: BLE001 - a case may raise anything
        return {
            "id": case["id"],
            "ok": expect_error,
            "value": None,
            "observed": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    observed = repr(value)
    if expect_error:
        return {
            "id": case["id"],
            "ok": False,
            "value": value,
            "observed": observed,
            "error": "expected the call to raise, but it returned a value",
        }
    ok = (bool(value) is expect) if isinstance(expect, bool) else (value == expect)
    return {
        "id": case["id"],
        "ok": bool(ok),
        "value": value,
        "observed": observed,
        "error": None if ok else f"expected {expect!r}, observed {observed}",
    }


def evaluate(rendered_source: str, cases: list[dict]) -> list[dict]:
    """Evaluate every case against one freshly loaded module instance."""

    namespace = exec_module(rendered_source)
    return [_case_result(case, namespace) for case in cases]


def both_case_lists(task: dict) -> list[dict]:
    return list(task["security_cases"]) + list(task["benign_cases"])


def _failures(results: list[dict]) -> list[dict]:
    return [result for result in results if not result["ok"]]


def _describe(failures: list[dict]) -> str:
    return "; ".join(f"{entry['id']}: {entry['error']}" for entry in failures[:8])


def seeded_bad_findings(task: dict) -> dict[str, list[str]]:
    """Run every seeded-bad candidate and return family -> failing case ids."""

    findings: dict[str, list[str]] = {}
    for entry in task["seeded_bad"]:
        rendered = render_source(task["source"], entry["expression"])
        results = evaluate(rendered, both_case_lists(task))
        findings[entry["family"]] = [result["id"] for result in results if not result["ok"]]
    return findings


# ---------------------------------------------------------------------------
# Marker machinery (polarity)
# ---------------------------------------------------------------------------


def blank_string_literals(expression: str) -> str:
    """Return ``expression`` with string literals and comments blanked out.

    Offsets are preserved, so matches inside the blanked text can be applied to the
    original string. Ported behaviour from the harness's polarity analysis: a marker
    that only appears inside a string literal must not count as evidence.
    """

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expression).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return expression

    line_starts = [0]
    for line in expression.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def offset(position: tuple[int, int]) -> int:
        row, column = position
        row = min(max(row - 1, 0), len(line_starts) - 1)
        return min(line_starts[row] + column, len(expression))

    characters = list(expression)
    for token in tokens:
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        start, end = offset(token.start), offset(token.end)
        for index in range(start, min(end, len(characters))):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def token_pattern(tokens: list[str]) -> re.Pattern[str] | None:
    """Longest-first alternation with identifier boundaries on word-like tokens."""

    if not tokens:
        return None
    pieces = []
    for token in sorted(set(tokens), key=len, reverse=True):
        escaped = re.escape(token)
        if _WORDY_TOKEN.match(token):
            pieces.append(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        else:
            pieces.append(escaped)
    return re.compile("|".join(pieces))


def marker_matches(expression: str, markers: list[str]) -> list[str]:
    """Which of ``markers`` occur in ``expression``'s code (literals blanked)."""

    pattern = token_pattern(markers)
    if pattern is None:
        return []
    return sorted({match.group(0) for match in pattern.finditer(blank_string_literals(expression))})


def marker_sides(
    expression: str, pairs: list[tuple[str, str]]
) -> tuple[list[str], list[str]]:
    """Which declared markers occur, split into (correct, converse) sides.

    The union is scanned longest-marker-first, so ``not in`` claims the span of
    text before the shorter ``in`` can see it and ``is not None`` before
    ``is None``. Without that, a converse that legitimately contains the shorter
    correct marker would be misread as mixed evidence.
    """

    correct = [marker for marker, _ in pairs]
    converse = [inverse for _, inverse in pairs]
    hits = marker_matches(expression, correct + converse)
    hit_correct = [hit for hit in hits if hit in correct]
    hit_converse = [hit for hit in hits if hit in converse]
    return hit_correct, hit_converse


def substitute_markers(expression: str, mapping: dict[str, str]) -> str:
    """Replace markers with their inverses in code, leaving literals untouched."""

    pattern = token_pattern(list(mapping))
    if pattern is None:
        return expression
    masked = blank_string_literals(expression)
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(masked):
        start, end = match.span()
        pieces.append(expression[cursor:start])
        pieces.append(mapping[match.group(0)])
        cursor = end
    pieces.append(expression[cursor:])
    return "".join(pieces)


def inverse_pairs(polarity: dict) -> tuple[list[tuple[str, str]] | None, str]:
    """Pair each correct marker with its declared converse.

    Returns ``(pairs, reason)``; ``pairs is None`` means the polarity block cannot
    be analysed mechanically and the task must be treated as UNKNOWN.
    """

    correct = [item for item in polarity.get("correct_marker") or [] if isinstance(item, str) and item]
    converse = [item for item in polarity.get("converse_marker") or [] if isinstance(item, str) and item]
    if not correct or not converse:
        return None, "one or both marker lists are empty"

    pairs: list[tuple[str, str]] = []
    for marker in correct:
        inverse = INVERSE_MARKERS.get(marker)
        if inverse is None:
            return None, f"no known inverse for correct_marker {marker!r}"
        if inverse not in converse:
            return None, f"inverse {inverse!r} of {marker!r} missing from converse_marker"
        pairs.append((marker, inverse))

    produced = sorted({inverse for _, inverse in pairs})
    if sorted(converse) != produced:
        return None, "converse_marker is not exactly the inverse image of correct_marker"
    return pairs, ""


def polarity_status(task: dict) -> tuple[str, list[str]]:
    """Classify a task's polarity metadata as PASS, UNKNOWN or FAIL."""

    polarity = task.get("polarity")
    if not isinstance(polarity, dict):
        return "UNKNOWN", ["no polarity object"]
    fact = polarity.get("fact")
    if not isinstance(fact, str) or not fact.strip():
        return "UNKNOWN", ["polarity.fact is missing"]

    pairs, reason = inverse_pairs(polarity)
    if pairs is None:
        return "UNKNOWN", [reason]

    reference = task["reference_fix"]
    try:
        ast.parse(reference, mode="eval")
    except SyntaxError:
        return "UNKNOWN", ["reference_fix does not parse as one expression"]

    correct = [marker for marker, _ in pairs]
    converse = [inverse for _, inverse in pairs]
    problems: list[str] = []

    ref_correct, ref_converse = marker_sides(reference, pairs)
    missing_correct = [marker for marker in correct if marker not in ref_correct]
    if missing_correct:
        problems.append(f"reference_fix lacks correct_marker {missing_correct}")
    if ref_converse:
        problems.append(f"reference_fix already carries converse_marker {ref_converse}")

    synthesised = substitute_markers(reference, dict(pairs))
    if synthesised.strip() == reference.strip():
        return "UNKNOWN", ["converse substitution changed nothing"]

    synth_correct, synth_converse = marker_sides(synthesised, pairs)
    missing_converse = [marker for marker in converse if marker not in synth_converse]
    if missing_converse:
        problems.append(f"synthesised converse lacks converse_marker {missing_converse}")
    if synth_correct:
        problems.append(f"synthesised converse still carries correct_marker {synth_correct}")

    if problems:
        return "FAIL", problems
    return "PASS", [f"converse: {synthesised}"]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def assert_schema(task: dict, task_id: str) -> None:
    assert isinstance(task, dict), f"{task_id}: task must be a JSON object"
    unknown = set(task) - set(REQUIRED_FIELDS) - OPTIONAL_FIELDS
    assert not unknown, f"{task_id}: unexpected schema fields {sorted(unknown)}"
    for field, expected_type in REQUIRED_FIELDS.items():
        assert field in task, f"{task_id}: missing required field {field!r}"
        assert isinstance(task[field], expected_type), (
            f"{task_id}: field {field!r} must be {expected_type.__name__}, "
            f"got {type(task[field]).__name__}"
        )

    assert task["id"] == task_id, f"{task_id}: 'id' must equal the file name"
    assert task["family"] in FAMILIES, (
        f"{task_id}: family {task['family']!r} is not one of {FAMILIES}"
    )
    assert task["placeholder"] == PLACEHOLDER, (
        f"{task_id}: placeholder must be {PLACEHOLDER!r}"
    )
    assert task["source"].strip(), f"{task_id}: source must not be empty"
    assert task["property"].strip(), f"{task_id}: property must not be empty"

    editable = task["editable"]
    assert set(editable) == {"path", "mode"}, f"{task_id}: editable keys must be path and mode"
    assert isinstance(editable["path"], str) and editable["path"].strip()
    assert editable["mode"] in {"expression", "line"}, f"{task_id}: bad editable.mode"

    assert isinstance(task["alternate_fixes"], list) and task["alternate_fixes"], (
        f"{task_id}: at least one alternate fix is required"
    )
    for index, alternate in enumerate(task["alternate_fixes"]):
        assert isinstance(alternate, str) and alternate.strip(), (
            f"{task_id}: alternate_fixes[{index}] must be a non-empty string"
        )
        ast.parse(alternate, mode="eval")
    ast.parse(task["reference_fix"], mode="eval")

    assert isinstance(task["polarity_sensitive"], bool)
    if task["polarity_sensitive"]:
        assert "polarity" in task, f"{task_id}: polarity_sensitive needs a polarity block"
        assert isinstance(task["polarity"], dict)
        assert isinstance(task["polarity"].get("fact"), str)
    else:
        assert "polarity" not in task, (
            f"{task_id}: non-polarity-sensitive task must not declare polarity markers"
        )

    assert_cases(task, task_id, "security_cases", expect=False)
    assert_cases(task, task_id, "benign_cases", expect=True)
    security_ids = {case["id"] for case in task["security_cases"]}
    benign_ids = {case["id"] for case in task["benign_cases"]}
    assert not security_ids & benign_ids, f"{task_id}: case ids collide across lists"


def assert_cases(task: dict, task_id: str, field: str, *, expect: bool) -> None:
    cases = task[field]
    assert isinstance(cases, list)
    assert CASE_MIN <= len(cases) <= CASE_MAX, (
        f"{task_id}: {field} needs {CASE_MIN}-{CASE_MAX} cases, found {len(cases)}"
    )
    seen: set[str] = set()
    for case in cases:
        assert isinstance(case, dict), f"{task_id}: {field} entries must be objects"
        assert set(case) <= CASE_FIELDS, f"{task_id}: {field} case has extra fields {sorted(set(case) - CASE_FIELDS)}"
        assert isinstance(case.get("id"), str) and case["id"], f"{task_id}: {field} case id"
        assert case["id"] not in seen, f"{task_id}: duplicate {field} id {case['id']!r}"
        seen.add(case["id"])
        assert isinstance(case.get("call"), str) and case["call"].strip(), (
            f"{task_id}: {field} case {case['id']!r} needs a call expression"
        )
        assert isinstance(case.get("expect_error"), bool), (
            f"{task_id}: {field} case {case['id']!r} needs a boolean expect_error"
        )
        if not case["expect_error"]:
            assert "expect" in case, f"{task_id}: {field} case {case['id']!r} needs expect"
            assert case["expect"] is expect, (
                f"{task_id}: {field} case {case['id']!r} must expect {expect!r}"
            )
        ast.parse(case["call"], mode="eval")


# ---------------------------------------------------------------------------
# Corpus-level tests
# ---------------------------------------------------------------------------


def test_corpus_is_present() -> None:
    assert TASKS_DIR.is_dir(), f"missing tasks directory: {TASKS_DIR}"
    assert len(TASK_IDS) == EXPECTED_TASKS, (
        f"expected {EXPECTED_TASKS} task files, found {len(TASK_IDS)}: {TASK_IDS}"
    )


def test_corpus_shape() -> None:
    tasks = all_tasks()
    assert len(tasks) == EXPECTED_TASKS
    assert len({task["id"] for task in tasks}) == EXPECTED_TASKS

    families = {task["family"] for task in tasks}
    assert families <= set(FAMILIES)
    assert len(families) >= MIN_FAMILIES, (
        f"need at least {MIN_FAMILIES} families, found {sorted(families)}"
    )

    sensitive = [task for task in tasks if task["polarity_sensitive"]]
    assert len(sensitive) >= MIN_POLARITY_SENSITIVE, (
        f"need at least {MIN_POLARITY_SENSITIVE} polarity-sensitive tasks, "
        f"found {len(sensitive)}"
    )
    for task in sensitive:
        polarity = task["polarity"]
        assert polarity.get("correct_marker"), f"{task['id']}: empty correct_marker"
        assert polarity.get("converse_marker"), f"{task['id']}: empty converse_marker"


def test_polarity_summary() -> None:
    """The conservative UNKNOWN escape hatch must not actually be needed.

    Recomputes instead of reading ``POLARITY_STATUS`` so the outcome does not depend
    on test execution order.
    """

    statuses: dict[str, str] = {}
    for task in all_tasks():
        if not task["polarity_sensitive"]:
            continue
        status, notes = polarity_status(task)
        POLARITY_STATUS[task["id"]] = status
        statuses[task["id"]] = status if status == "PASS" else f"{status}: {notes}"

    assert statuses, "no polarity-sensitive tasks found"
    undecided = {task_id: status for task_id, status in statuses.items() if status != "PASS"}
    assert not undecided, "polarity could not be confirmed for: " + ", ".join(
        f"{task_id}={status}" for task_id, status in undecided.items()
    )


def test_seeded_bad_evidence_recorded() -> None:
    """Every seeded-bad candidate must fail at least one case, and be recorded."""

    findings = {task["id"]: seeded_bad_findings(task) for task in all_tasks()}
    SEEDED_BAD_FINDINGS.update(findings)
    for task_id, families in findings.items():
        assert set(families) == set(SEEDED_BAD_FAMILIES), task_id
        for family, failing in families.items():
            assert failing, f"{task_id}: seeded {family} fails no case"


# ---------------------------------------------------------------------------
# Per-task tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_schema(task_id: str) -> None:
    assert_schema(load_task(task_id), task_id)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_source_shape(task_id: str) -> None:
    task = load_task(task_id)
    source = task["source"]

    lines = source.splitlines()
    assert SOURCE_MIN_LINES <= len(lines) <= SOURCE_MAX_LINES, (
        f"{task_id}: source must be {SOURCE_MIN_LINES}-{SOURCE_MAX_LINES} lines, "
        f"found {len(lines)}"
    )

    placeholder_lines = [index for index, line in enumerate(lines) if PLACEHOLDER in line]
    assert source.count(PLACEHOLDER) == 1, f"{task_id}: placeholder must occur exactly once"
    assert len(placeholder_lines) == 1
    marker_line = lines[placeholder_lines[0]].strip()
    assert marker_line == f"return {PLACEHOLDER}", (
        f"{task_id}: the placeholder must be the whole of a single return expression, "
        f"found {marker_line!r}"
    )

    editable = task["editable"]
    assert editable["mode"] == "expression", f"{task_id}: editable mode must be expression"
    assert any(line.strip().startswith(editable["path"]) for line in lines), (
        f"{task_id}: editable.path {editable['path']!r} does not name a line in the source"
    )
    function_name = editable["path"].split()[-1]
    namespace = exec_module(render_source(source, task["reference_fix"]))
    assert callable(namespace.get(function_name)), (
        f"{task_id}: rendered module does not expose callable {function_name!r}"
    )

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root in ALLOWED_MODULE_IMPORTS, (
                    f"{task_id}: import of {alias.name!r} is not allowed in a task module"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{task_id}: relative imports are not allowed"
            root = (node.module or "").split(".")[0]
            assert root in ALLOWED_MODULE_IMPORTS, (
                f"{task_id}: import from {node.module!r} is not allowed in a task module"
            )
        elif isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_NAMES, (
                f"{task_id}: task module references forbidden name {node.id!r}"
            )
        elif isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_ATTRIBUTES, (
                f"{task_id}: task module references forbidden attribute {node.attr!r}"
            )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_module_loads(task_id: str) -> None:
    task = load_task(task_id)
    rendered = render_source(task["source"], task["reference_fix"])
    namespace = exec_module(rendered)
    assert namespace, f"{task_id}: exec produced an empty namespace"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_fix_passes(task_id: str) -> None:
    task = load_task(task_id)
    rendered = render_source(task["source"], task["reference_fix"])
    results = evaluate(rendered, both_case_lists(task))
    failures = _failures(results)
    assert not failures, (
        f"{task_id}: reference_fix fails {len(failures)}/{len(results)} cases -> "
        f"{_describe(failures)}"
    )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_returns_real_booleans(task_id: str) -> None:
    task = load_task(task_id)
    rendered = render_source(task["source"], task["reference_fix"])
    namespace = exec_module(rendered)
    not_boolean = [
        case["id"]
        for case in both_case_lists(task)
        if not isinstance(_case_result(case, namespace)["value"], bool)
    ]
    assert not not_boolean, (
        f"{task_id}: reference observes non-boolean values for {not_boolean}; case "
        "observations must be real booleans so any grader compares them identically"
    )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_alternate_fixes_pass(task_id: str) -> None:
    task = load_task(task_id)
    cases = both_case_lists(task)
    for index, alternate in enumerate(task["alternate_fixes"]):
        rendered = render_source(task["source"], alternate)
        results = evaluate(rendered, cases)
        failures = _failures(results)
        assert not failures, (
            f"{task_id}: alternate_fixes[{index}] fails {len(failures)}/{len(results)} "
            f"cases -> {_describe(failures)}"
        )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_seeded_bad_entries(task_id: str) -> None:
    task = load_task(task_id)
    source = task["source"]
    entries = task["seeded_bad"]
    assert len(entries) == len(SEEDED_BAD_FAMILIES), (
        f"{task_id}: seeded_bad must hold exactly {len(SEEDED_BAD_FAMILIES)} entries, "
        f"found {len(entries)}"
    )
    families = [entry["family"] for entry in entries]
    assert sorted(families) == sorted(SEEDED_BAD_FAMILIES), (
        f"{task_id}: seeded_bad families must be exactly {SEEDED_BAD_FAMILIES}, "
        f"found {sorted(families)}"
    )
    for entry in entries:
        assert isinstance(entry["expression"], str) and entry["expression"].strip()
        ast.parse(entry["expression"], mode="eval")
        why = entry["why"]
        assert isinstance(why, str) and why.strip()
        assert why.rstrip().endswith("."), f"{task_id}: seeded {entry['family']} why needs one sentence"
        assert ". " not in why.rstrip()[:-1], (
            f"{task_id}: seeded {entry['family']} why must be a single sentence"
        )

    findings = seeded_bad_findings(task)
    for entry in entries:
        family = entry["family"]
        rendered = render_source(source, entry["expression"])
        failed = findings[family]
        assert failed, f"{task_id}: seeded {family} fails no security or benign case"

        security = evaluate(rendered, task["security_cases"])
        benign = evaluate(rendered, task["benign_cases"])

        if family == "symptom_suppression":
            assert all(result["ok"] for result in security), (
                f"{task_id}: symptom_suppression must block the whole PoC set, "
                f"failed {_describe(_failures(security))}"
            )
            assert any(not result["ok"] for result in benign), (
                f"{task_id}: symptom_suppression must break some benign behaviour"
            )
        elif family == "guard_too_weak":
            assert any(result["ok"] for result in security), (
                f"{task_id}: guard_too_weak must still pass some security case"
            )
            assert any(not result["ok"] for result in security), (
                f"{task_id}: guard_too_weak must fail at least one security case"
            )
        elif family == "feature_deletion":
            body = ast.parse(entry["expression"], mode="eval").body
            assert isinstance(body, ast.Constant) and not body.value, (
                f"{task_id}: feature_deletion must be a literal falsey expression"
            )
            assert all(not result["ok"] for result in benign), (
                f"{task_id}: feature_deletion must fail every benign case"
            )
        elif family == "ast_wrapper":
            assert _AST_WRAPPER.match(entry["expression"]), (
                f"{task_id}: ast_wrapper must be shaped 'True and (<weak expression>)'"
            )
            assert any(not result["ok"] for result in security), (
                f"{task_id}: ast_wrapper must still admit a security case"
            )

    SEEDED_BAD_FINDINGS[task_id] = findings


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_polarity_markers(task_id: str) -> None:
    task = load_task(task_id)
    if not task["polarity_sensitive"]:
        assert "polarity" not in task, (
            f"{task_id}: non-polarity-sensitive task must not declare a polarity block"
        )
        return

    status, notes = polarity_status(task)
    POLARITY_STATUS[task_id] = status
    if status != "PASS":
        # Conservative per docs/SPEC.md: an undecidable or unmatched polarity is not
        # evidence against the task, so this one assertion is skipped. The corpus
        # still has to be decidable; test_polarity_summary enforces that.
        pytest.skip(f"{task_id}: polarity {status} -> {notes}")


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_is_deterministic(task_id: str) -> None:
    task = load_task(task_id)
    rendered = render_source(task["source"], task["reference_fix"])
    cases = both_case_lists(task)

    def snapshot() -> list[tuple[str, bool, str, str | None]]:
        return [
            (result["id"], result["ok"], result["observed"] or "", result["error"])
            for result in evaluate(rendered, cases)
        ]

    first, second = snapshot(), snapshot()
    assert first == second, f"{task_id}: reference observations are not deterministic"
