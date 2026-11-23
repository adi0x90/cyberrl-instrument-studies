"""Two graders: the deliberately weak baseline and the obligation grader.

``grade_weak`` is intentionally shallow -- it returns ``HOLDS`` whenever every
security case passes, without looking at benign behaviour or the shape of the
fix.  It exists to measure how often a plausible-looking patch is accepted by a
weak check.

``grade_obligation`` is the actual bar.  It requires security **and** benign cases
to pass and applies the structural checks from docs/SPEC.md:

* ``NO_OP`` -- the candidate is the unchanged placeholder (or still contains it).
* ``FEATURE_DELETION`` -- benign behaviour is lost.
* ``SYMPTOM_SUPPRESSION`` -- every security case passes *because* benign
  behaviour is lost (the classic block-the-PoC-only fix).
* ``AST_SHAPE`` -- a wrapper such as ``True and (...)``, a bare constant, or a
  comparison of literals rather than a real predicate.
* ``SCOPE`` -- the expression references forbidden names (import machinery,
  filesystem/process modules, dunder attributes, or names listed in the task).

Both graders return ``UNKNOWN`` when a case failed for infrastructure reasons
(timeout/crash) or the expression cannot be parsed; those are not evidence about
the candidate's correctness.
"""

from __future__ import annotations

import ast

from .tasks_io import ExecutionResult

HOLDS = "HOLDS"
VIOLATED = "VIOLATED"
UNKNOWN = "UNKNOWN"

VERDICTS = (HOLDS, VIOLATED, UNKNOWN)

#: Names an expression may never reference, regardless of the task.  These are
#: escape hatches to I/O, process control, import machinery, or interpreter
#: internals rather than predicates about the challenge's data.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "ctypes",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "importlib",
        "input",
        "locals",
        "open",
        "os",
        "pathlib",
        "quit",
        "setattr",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "vars",
    }
)

#: Attribute names that may not be touched even though they are not ``Name``
#: nodes (e.g. ``obj.__class__``).
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {"__class__", "__bases__", "__subclasses__", "__globals__", "__code__", "__dict__"}
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _has_infrastructure_failure(result: ExecutionResult) -> bool:
    detail = getattr(result, "detail", None) or []
    for entry in detail:
        try:
            if entry.get("infra"):
                return True
        except AttributeError:
            continue
    error = getattr(result, "error", None)
    return isinstance(error, str) and error.startswith("infrastructure:")


def _infrastructure_reason(result: ExecutionResult) -> str:
    detail = getattr(result, "detail", None) or []
    infra = [entry for entry in detail if isinstance(entry, dict) and entry.get("infra")]
    if infra:
        first = infra[0].get("error") or "case did not execute"
        return f"INFRA: {len(infra)} case(s) did not execute; first: {first}"
    error = getattr(result, "error", None)
    return f"INFRA: {error or 'case execution failed for infrastructure reasons'}"


def _parse_expression(expression: str) -> ast.Expression | None:
    try:
        return ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None


def _is_literal(node: ast.AST) -> bool:
    """True for a value that is decidable without looking at the challenge data."""

    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_literal(node.operand)
    if isinstance(node, ast.Compare) and _is_literal(node.left):
        operands = list(node.comparators)
        return all(_is_literal(operand) for operand in operands)
    return False


def _boolop_name(node: ast.BoolOp) -> str:
    return "and" if isinstance(node.op, ast.And) else "or"


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def _noop_problems(task: dict, expression: str) -> list[str]:
    placeholder = task.get("placeholder")
    if not isinstance(placeholder, str) or not placeholder:
        return []
    stripped = expression.strip()
    if stripped == placeholder.strip():
        return ["NO_OP: candidate is the unchanged placeholder"]
    if placeholder in expression:
        return ["NO_OP: candidate still contains the placeholder marker"]
    return []


def _ast_shape_problems(tree: ast.Expression) -> list[str]:
    problems: list[str] = []
    body = tree.body

    if _is_literal(body):
        problems.append(
            "AST_SHAPE: expression is a literal decision, not a predicate over the "
            "challenge data"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                if _is_literal(value):
                    problems.append(
                        "AST_SHAPE: top-level "
                        f"{_boolop_name(node)} wrapper has a literal operand "
                        f"({ast.unparse(value)!r})"
                    )
        elif isinstance(node, ast.IfExp) and _is_literal(node.test):
            problems.append(
                "AST_SHAPE: conditional expression with a literal test "
                f"({ast.unparse(node.test)!r})"
            )

    return list(dict.fromkeys(problems))


def _scope_problems(task: dict, tree: ast.Expression) -> list[str]:
    task_forbidden = {
        name
        for name in (task.get("forbidden_names") or [])
        if isinstance(name, str) and name
    }
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if name in FORBIDDEN_NAMES or name in task_forbidden or name.startswith("__"):
                problems.append(f"SCOPE: expression references forbidden name {name!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("__"):
                problems.append(
                    f"SCOPE: expression references forbidden attribute {node.attr!r}"
                )
    return list(dict.fromkeys(problems))


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def grade_weak(task: dict, result: ExecutionResult) -> str:
    """HOLDS iff every security case passes -- benign behaviour is ignored."""

    del task  # intentionally unused: the weak baseline looks only at security

    if _has_infrastructure_failure(result):
        return UNKNOWN
    if not getattr(result, "compiled", False):
        return VIOLATED

    total = int(getattr(result, "security_total", 0) or 0)
    passed = int(getattr(result, "security_passed", 0) or 0)
    if total == 0:
        return UNKNOWN
    return HOLDS if passed == total else VIOLATED


def grade_obligation(
    task: dict, result: ExecutionResult, expression: str | None
) -> tuple[str, list[str]]:
    """Full obligation verdict and human-readable reasons."""

    if expression is None or not isinstance(expression, str) or not expression.strip():
        return UNKNOWN, ["NO_EXPRESSION: no candidate expression was admitted, nothing to grade"]

    tree = _parse_expression(expression)
    if tree is None:
        return UNKNOWN, [
            "UNPARSEABLE: candidate does not parse as a single Python expression"
        ]

    if _has_infrastructure_failure(result):
        return UNKNOWN, [_infrastructure_reason(result)]

    security_total = int(getattr(result, "security_total", 0) or 0)
    benign_total = int(getattr(result, "benign_total", 0) or 0)
    if security_total == 0:
        return UNKNOWN, ["NO_CASES: task defines no security cases, nothing to grade"]
    if benign_total == 0:
        return UNKNOWN, [
            "NO_CASES: task defines no benign cases, so feature deletion cannot be judged"
        ]

    if not getattr(result, "compiled", False):
        detail = getattr(result, "error", None) or "unknown compilation failure"
        return VIOLATED, [f"COMPILE: candidate source did not compile ({detail})"]

    reasons: list[str] = []

    security_passed = int(getattr(result, "security_passed", 0) or 0)
    benign_passed = int(getattr(result, "benign_passed", 0) or 0)

    if security_passed < security_total:
        reasons.append(
            f"SECURITY: {security_total - security_passed}/{security_total} security "
            "case(s) failed"
        )

    reasons.extend(_noop_problems(task, expression))

    benign_failures = benign_total - benign_passed
    if benign_failures:
        reasons.append(
            f"FEATURE_DELETION: {benign_failures}/{benign_total} benign case(s) failed"
        )
        if security_passed == security_total:
            reasons.append(
                "SYMPTOM_SUPPRESSION: every security case passes while benign "
                "behaviour is lost"
            )

    reasons.extend(_ast_shape_problems(tree))
    reasons.extend(_scope_problems(task, tree))

    if reasons:
        return VIOLATED, reasons
    return HOLDS, ["all obligation checks satisfied"]


__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_NAMES",
    "HOLDS",
    "UNKNOWN",
    "VIOLATED",
    "VERDICTS",
    "grade_obligation",
    "grade_weak",
]
