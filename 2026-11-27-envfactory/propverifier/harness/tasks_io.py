"""Task loading, placeholder rendering, and isolated case execution.

Design notes
------------
* ``load_task`` validates the docs/SPEC.md task schema and fails loudly on malformed
  tasks.  Fields that only some task families need (``polarity``,
  ``polarity_sensitive``, ``alternate_fixes``, ``seeded_bad``) are optional;
  everything a grader or runner cannot work without is required.
* ``render_candidate_source`` substitutes exactly one placeholder occurrence.
* ``run_cases`` executes the rendered challenge module in a **child process**
  (``spawn`` where possible, ``fork`` for stdin/REPL callers) with a per-case
  wall-clock timeout, so a hostile candidate can hang, spin or crash only
  itself.  The child runs with a restricted ``__builtins__``:
  ``open``/``eval``/``exec``/``compile`` and friends are unavailable, and
  ``__import__`` only resolves a small stdlib whitelist (no ``os``, ``sys``,
  ``socket``, ``subprocess``, ``pathlib``, ...).  That is defence in depth on top
  of the OS process boundary, not a certified sandbox.
* ``execute_patch`` parses the candidate with ``ast.parse(mode="eval")``,
  compiles the rendered module, and returns an :class:`ExecutionResult`.

Infrastructure failures (timeout, worker crash, unrenderable/uncompilable
candidate) are surfaced via ``error`` on the result and an ``infra`` flag on the
affected detail entries.  ``harness.graders`` maps those to ``UNKNOWN``.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CASE_TIMEOUT_S = 5.0
_MAX_REPR_CHARS = 200

#: Stdlib modules a challenge module may import.  Everything else (notably any
#: I/O, process, socket, or filesystem module) is blocked in the child.
SAFE_MODULES: frozenset[str] = frozenset(
    {
        "abc",
        "array",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "collections",
        "collections.abc",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fnmatch",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "hmac",
        "ipaddress",
        "itertools",
        "json",
        "math",
        "operator",
        "pprint",
        "re",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "typing",
        "unicodedata",
        "urllib.parse",
    }
)

_ALLOWED_BUILTIN_NAMES = (
    "abs all any ascii bin bool bytearray bytes callable chr dict divmod enumerate "
    "filter float format frozenset getattr hasattr hash hex int isinstance issubclass "
    "iter len list map max min next oct ord pow range repr reversed round set slice "
    "sorted str sum super tuple type zip"
).split()

_ALLOWED_EXCEPTIONS = (
    "ArithmeticError AssertionError AttributeError BaseException BufferError "
    "EOFError Exception FileNotFoundError FloatingPointError ImportError "
    "IndentationError IndexError KeyError KeyboardInterrupt LookupError MemoryError "
    "NameError NotImplementedError OSError OverflowError PermissionError "
    "RecursionError RuntimeError StopIteration SyntaxError SystemExit TypeError "
    "UnicodeDecodeError UnicodeEncodeError UnicodeError ValueError ZeroDivisionError"
).split()


class TaskError(ValueError):
    """Base class for task loading/validation problems."""


class TaskNotFoundError(TaskError):
    pass


class TaskValidationError(TaskError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    compiled: bool
    security_passed: int
    security_total: int
    benign_passed: int
    benign_total: int
    detail: list[dict]
    error: str | None


# ---------------------------------------------------------------------------
# Task loading / validation
# ---------------------------------------------------------------------------


def load_task(task_id: str, root: Path) -> dict:
    """Load and validate ``<root>/tasks/<task_id>.json``."""

    if not isinstance(task_id, str) or not task_id.strip():
        raise TaskValidationError("task_id must be a non-empty string")

    path = Path(root) / "tasks" / f"{task_id}.json"
    if not path.is_file():
        raise TaskNotFoundError(f"task file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TaskError(f"could not read task file {path}: {exc}") from exc

    try:
        task = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskValidationError(f"{path}: invalid JSON: {exc}") from exc

    _validate_task(task, path)
    return task


def _validate_task(task: Any, path: Path) -> None:
    where = str(path)
    if not isinstance(task, dict):
        raise TaskValidationError(f"{where}: top-level JSON value must be an object")

    _require(task, "id", str, where)
    _require(task, "family", str, where)
    _require(task, "source", str, where)
    _require(task, "placeholder", str, where)
    _require(task, "property", str, where)
    _require(task, "reference_fix", str, where)
    _require(task, "editable", dict, where)
    _require(task, "security_cases", list, where)
    _require(task, "benign_cases", list, where)

    if not task["id"].strip():
        raise TaskValidationError(f"{where}: 'id' must be non-empty")
    if not task["family"].strip():
        raise TaskValidationError(f"{where}: 'family' must be non-empty")
    if not task["source"].strip():
        raise TaskValidationError(f"{where}: 'source' must be non-empty")
    if not task["placeholder"]:
        raise TaskValidationError(f"{where}: 'placeholder' must be non-empty")

    placeholder = task["placeholder"]
    occurrences = task["source"].count(placeholder)
    if occurrences != 1:
        raise TaskValidationError(
            f"{where}: the editable region must contain exactly one placeholder "
            f"{placeholder!r}; found {occurrences}"
        )

    editable = task["editable"]
    _require(editable, "path", str, f"{where}: editable", allow_empty=False)
    _require(editable, "mode", str, f"{where}: editable")
    if editable["mode"] not in {"expression", "line"}:
        raise TaskValidationError(
            f"{where}: editable.mode must be 'expression' or 'line', got "
            f"{editable['mode']!r}"
        )

    if "polarity_sensitive" in task and not isinstance(task["polarity_sensitive"], bool):
        raise TaskValidationError(f"{where}: 'polarity_sensitive' must be a boolean")

    if "polarity" in task:
        _validate_polarity(task["polarity"], where)

    if "alternate_fixes" in task:
        _validate_str_list(task["alternate_fixes"], "alternate_fixes", where)

    if "seeded_bad" in task:
        seeded_bad = task["seeded_bad"]
        if not isinstance(seeded_bad, list) or not seeded_bad:
            raise TaskValidationError(f"{where}: 'seeded_bad' must be a non-empty list")
        for index, entry in enumerate(seeded_bad):
            if not isinstance(entry, dict):
                raise TaskValidationError(f"{where}: seeded_bad[{index}] must be an object")
            for field in ("family", "expression", "why"):
                if not isinstance(entry.get(field), str):
                    raise TaskValidationError(
                        f"{where}: seeded_bad[{index}].{field} must be a string"
                    )

    _validate_cases(task["security_cases"], "security_cases", where)
    _validate_cases(task["benign_cases"], "benign_cases", where)


def _require(
    mapping: dict,
    key: str,
    expected_type: type,
    where: str,
    *,
    allow_empty: bool = True,
) -> Any:
    if key not in mapping:
        raise TaskValidationError(f"{where}: missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise TaskValidationError(
            f"{where}: field {key!r} must be {expected_type.__name__}, got "
            f"{type(value).__name__}"
        )
    if not allow_empty and isinstance(value, str) and not value:
        raise TaskValidationError(f"{where}: field {key!r} must be non-empty")
    return value


def _validate_str_list(value: Any, field: str, where: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TaskValidationError(f"{where}: {field!r} must be a list of strings")


def _validate_polarity(polarity: Any, where: str) -> None:
    if not isinstance(polarity, dict):
        raise TaskValidationError(f"{where}: 'polarity' must be an object")
    if not isinstance(polarity.get("fact"), str):
        raise TaskValidationError(f"{where}: polarity.fact must be a string")
    for field in ("correct_marker", "converse_marker"):
        if field in polarity:
            _validate_str_list(polarity[field], f"polarity.{field}", where)


def _validate_cases(cases: list, field: str, where: str) -> None:
    if not cases:
        raise TaskValidationError(f"{where}: {field!r} must be a non-empty list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        label = f"{where}: {field}[{index}]"
        if not isinstance(case, dict):
            raise TaskValidationError(f"{label} must be an object")
        case_id = _require(case, "id", str, label)
        if not case_id:
            raise TaskValidationError(f"{label}.id must be non-empty")
        _require(case, "call", str, label, allow_empty=False)
        expect_error = _require(case, "expect_error", bool, label)
        if not expect_error and "expect" not in case:
            raise TaskValidationError(f"{label}: missing 'expect' for a non-error case")
        if case_id in seen:
            raise TaskValidationError(f"{where}: duplicate case id {case_id!r} in {field}")
        seen.add(case_id)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_candidate_source(task: dict, expression: str) -> str:
    """Substitute the candidate expression for the task's single placeholder."""

    if not isinstance(task, dict):
        raise TaskValidationError("task must be a dict")
    if not isinstance(expression, str):
        raise TypeError(f"expression must be str, not {type(expression).__name__}")

    placeholder = task.get("placeholder")
    source = task.get("source")
    if not isinstance(placeholder, str) or not placeholder:
        raise TaskValidationError("task is missing a non-empty 'placeholder'")
    if not isinstance(source, str) or not source:
        raise TaskValidationError("task is missing a non-empty 'source'")

    occurrences = source.count(placeholder)
    if occurrences != 1:
        raise TaskValidationError(
            "cannot render candidate: the editable region must contain exactly one "
            f"placeholder {placeholder!r}; found {occurrences}"
        )
    return source.replace(placeholder, expression)


# ---------------------------------------------------------------------------
# Case execution (parent side)
# ---------------------------------------------------------------------------


def _infra_result(case: dict, reason: str) -> dict:
    return {
        "id": case.get("id"),
        "ok": False,
        "observed": "",
        "error": f"infrastructure: {reason}",
        "infra": True,
    }


def run_cases(source: str, cases: list[dict], *, timeout_s: float = DEFAULT_CASE_TIMEOUT_S) -> list[dict]:
    """Execute ``cases`` against ``source`` in a throw-away child process.

    Returns one entry per case: ``{"id", "ok", "observed", "error", "infra"}``.
    ``infra`` is an additive flag (not part of the SPEC-documented keys) that
    marks cases that never produced a real result because the worker timed out,
    crashed, or could not load the challenge module.
    """

    if not isinstance(source, str):
        raise TypeError(f"source must be str, not {type(source).__name__}")
    if not isinstance(cases, list):
        raise TypeError("cases must be a list")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if not cases:
        return []

    context = _execution_context()
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_case_worker,
        args=(source, cases, child_conn),
        daemon=True,
    )

    results: list[dict] = []
    try:
        process.start()
        child_conn.close()

        for index, case in enumerate(cases):
            message = _receive(parent_conn, timeout_s)
            if message is None:
                alive = process.is_alive()
                reason = (
                    f"case did not finish within {timeout_s}s (worker killed)"
                    if alive
                    else "case worker process died without returning a result"
                )
                for missing in cases[index:]:
                    results.append(_infra_result(missing, reason))
                break

            kind, payload = message
            if kind == "case" and isinstance(payload, dict):
                results.append(payload)
                continue
            if kind == "setup_error":
                reason = f"challenge module failed to load: {payload}"
                for missing in cases[index:]:
                    results.append(_infra_result(missing, reason))
                break
            if kind == "done":
                reason = "case worker stopped before returning this case"
                for missing in cases[index:]:
                    results.append(_infra_result(missing, reason))
                break
            reason = f"unexpected worker message {kind!r}"
            for missing in cases[index:]:
                results.append(_infra_result(missing, reason))
            break
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5.0)
        parent_conn.close()

    # Defensive: guarantee exactly one result per case.
    while len(results) < len(cases):
        results.append(_infra_result(cases[len(results)], "no result was recorded"))
    return results[: len(cases)]


def _execution_context() -> Any:
    """Pick a multiprocessing context that can start a worker here.

    ``spawn`` is the safe default: a fresh interpreter, no inherited fds or locks.
    It re-imports the caller's ``__main__`` from disk, which is impossible when the
    parent was started from stdin/a REPL (``python -``), so in that corner we fall
    back to ``fork`` on POSIX.  Pytest and the runners run from real files and use
    ``spawn``.
    """

    if sys.platform != "win32":
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if not isinstance(main_file, str) or not Path(main_file).is_file():
            try:
                return multiprocessing.get_context("fork")
            except ValueError:  # pragma: no cover - fork exists on POSIX
                pass
    return multiprocessing.get_context("spawn")


def _receive(conn: Any, timeout_s: float) -> tuple[str, Any] | None:
    try:
        if not conn.poll(timeout_s):
            return None
        return conn.recv()
    except (EOFError, OSError):
        return None


# ---------------------------------------------------------------------------
# Case execution (child side)
# ---------------------------------------------------------------------------


def _case_worker(source: str, cases: list[dict], conn: Any) -> None:
    """Child entry point.  Runs untrusted candidate code with a restricted env."""

    sys.dont_write_bytecode = True
    try:
        namespace = _build_namespace(source)
    except BaseException as exc:  # noqa: BLE001 - child must never leak a hang
        _safe_send(conn, ("setup_error", _format_exc(exc)))
        conn.close()
        return

    try:
        for case in cases:
            try:
                result = _run_case(namespace, case)
            except BaseException as exc:  # noqa: BLE001
                result = {
                    "id": case.get("id"),
                    "ok": False,
                    "observed": "",
                    "error": _format_exc(exc),
                    "infra": False,
                }
            if not _safe_send(conn, ("case", result)):
                return
        _safe_send(conn, ("done", None))
    finally:
        conn.close()


def _safe_send(conn: Any, message: tuple[str, Any]) -> bool:
    try:
        conn.send(message)
        return True
    except (BrokenPipeError, OSError, EOFError):
        return False


def _build_namespace(source: str) -> dict:
    namespace = {
        "__name__": "challenge_module",
        "__doc__": None,
        "__package__": None,
        "__builtins__": _safe_builtins(),
    }
    code = compile(source, "<challenge>", "exec")
    exec(code, namespace)  # noqa: S102 - intentional, restricted builtins
    return namespace


def _safe_import(
    name: str,
    globals: dict | None = None,
    locals: dict | None = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    if level:
        raise ImportError("relative imports are not allowed in the challenge sandbox")
    if name not in SAFE_MODULES:
        raise ImportError(
            f"import of {name!r} is blocked in the challenge sandbox "
            "(no filesystem or network access)"
        )
    return importlib.import_module(name)


def _safe_builtins() -> dict:
    safe: dict[str, Any] = {"__import__": _safe_import, "__build_class__": builtins.__build_class__}
    for name in _ALLOWED_BUILTIN_NAMES:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    for name in _ALLOWED_EXCEPTIONS:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    safe["True"] = True
    safe["False"] = False
    safe["None"] = None
    return safe


def _run_case(namespace: dict, case: dict) -> dict:
    call = case.get("call")
    case_id = case.get("id")
    expect_error = bool(case.get("expect_error", False))

    try:
        value = eval(call, namespace)  # noqa: S307 - isolated child, restricted builtins
    except BaseException as exc:  # noqa: BLE001 - a case may raise anything
        if expect_error:
            return {
                "id": case_id,
                "ok": True,
                "observed": f"raised {type(exc).__name__}",
                "error": None,
                "infra": False,
            }
        return {
            "id": case_id,
            "ok": False,
            "observed": "",
            "error": _format_exc(exc),
            "infra": False,
        }

    observed = _short_repr(value)
    if expect_error:
        return {
            "id": case_id,
            "ok": False,
            "observed": observed,
            "error": f"expected the call to raise, but it returned {observed}",
            "infra": False,
        }

    expected = case.get("expect")
    try:
        if isinstance(expected, bool):
            ok = bool(value) is expected
        else:
            ok = value == expected
    except BaseException:  # noqa: BLE001 - hostile __bool__/__eq__
        ok = False

    return {
        "id": case_id,
        "ok": bool(ok),
        "observed": observed,
        "error": None if ok else f"expected {expected!r}, observed {observed}",
        "infra": False,
    }


def _short_repr(value: Any, limit: int = _MAX_REPR_CHARS) -> str:
    try:
        text = repr(value)
    except BaseException:  # noqa: BLE001
        text = f"<unreprable {type(value).__name__}>"
    return text[:limit] + ("..." if len(text) > limit else "")


def _format_exc(exc: BaseException, limit: int = 300) -> str:
    try:
        text = f"{type(exc).__name__}: {exc}"
    except BaseException:  # noqa: BLE001
        text = type(exc).__name__
    return text[:limit]


# ---------------------------------------------------------------------------
# Patch execution
# ---------------------------------------------------------------------------


def execute_patch(task: dict, expression: str | None) -> ExecutionResult:
    """Render the candidate into the task source and run every case.

    ``expression is None`` means "no candidate" (e.g. the model's output was not
    admitted) and yields ``compiled=False`` without spawning a worker.
    """

    if not isinstance(task, dict):
        raise TaskValidationError("task must be a dict")

    security_cases = list(task.get("security_cases") or [])
    benign_cases = list(task.get("benign_cases") or [])
    security_total = len(security_cases)
    benign_total = len(benign_cases)

    def failure(error: str) -> ExecutionResult:
        return ExecutionResult(
            compiled=False,
            security_passed=0,
            security_total=security_total,
            benign_passed=0,
            benign_total=benign_total,
            detail=[],
            error=error,
        )

    if expression is None:
        return failure("no candidate expression supplied")
    if not isinstance(expression, str):
        return failure(
            f"candidate expression must be str, got {type(expression).__name__}"
        )
    if not expression.strip():
        return failure("candidate expression is empty")

    if not _candidate_parses(expression, task):
        return failure(
            f"candidate expression does not parse as a Python expression: "
            f"{expression.strip()[:120]!r}"
        )

    try:
        source = render_candidate_source(task, expression)
    except TaskError as exc:
        return failure(f"could not render candidate source: {exc}")

    try:
        compile(source, "<candidate>", "exec")
    except (SyntaxError, ValueError) as exc:
        return failure(f"candidate source does not compile: {_format_exc(exc)}")

    detail = run_cases(source, security_cases + benign_cases)
    security_detail = detail[:security_total]
    benign_detail = detail[security_total:]

    security_passed = sum(1 for entry in security_detail if entry.get("ok"))
    benign_passed = sum(1 for entry in benign_detail if entry.get("ok"))

    infra_entries = [entry for entry in detail if entry.get("infra")]
    error: str | None = None
    if infra_entries:
        error = (
            f"infrastructure: {len(infra_entries)} of {len(detail)} case(s) did not "
            "execute"
        )

    return ExecutionResult(
        compiled=True,
        security_passed=security_passed,
        security_total=security_total,
        benign_passed=benign_passed,
        benign_total=benign_total,
        detail=detail,
        error=error,
    )


def _candidate_parses(expression: str, task: dict) -> bool:
    """SPEC requires ``ast.parse(mode="eval")`` for the candidate expression.

    A ``mode == "line"`` task may legitimately submit a single ``return <expr>``
    line; that unwrapped expression is what is checked and executed.
    """

    targets = [expression]
    editable = task.get("editable")
    if isinstance(editable, dict) and editable.get("mode") == "line":
        stripped = expression.strip()
        if stripped.startswith("return "):
            targets.append(stripped[len("return ") :].strip())

    for target in targets:
        if not target.strip():
            continue
        try:
            ast.parse(target, mode="eval")
            return True
        except (SyntaxError, ValueError):
            continue
    return False


__all__ = [
    "DEFAULT_CASE_TIMEOUT_S",
    "ExecutionResult",
    "TaskError",
    "TaskNotFoundError",
    "TaskValidationError",
    "execute_patch",
    "load_task",
    "render_candidate_source",
    "run_cases",
]
