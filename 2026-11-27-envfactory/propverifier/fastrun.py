"""In-process case evaluator for the A1 latency measurement.

The baseline harness runs every candidate in a throw-away child process
(``harness.tasks_io.run_cases``) because it grades untrusted model output.  That
is the right substrate for correctness runs.  It is the wrong substrate for a
10,000-call latency measurement: process spawn dominates, and the measurement
would report the cost of ``multiprocessing`` rather than the cost of the
property tier.

This module therefore executes cases in-process with **exactly** the harness
sandbox: it reuses ``harness.tasks_io._build_namespace`` (same restricted
builtins, same import allowlist, same compile step) and mirrors
``harness.tasks_io._run_case``'s exception handling.  ``tests/test_fastrun.py``
asserts equivalence with the subprocess runner over every case of every task,
so the fast path cannot silently drift from the graded path.

Nothing here is used for correctness verdicts that enter evidence.
"""

from __future__ import annotations

from typing import Any, Iterable

from .harness.tasks_io import _build_namespace, render_candidate_source

#: A case label is either ("value", bool) or ("raise", ExceptionTypeName).
Label = tuple[str, Any]

_VALUE = "value"
_RAISE = "raise"


def compile_candidate(task: dict, expression: str) -> dict:
    """Render + compile a candidate expression into a sandbox namespace.

    Raises whatever ``_build_namespace`` raises (SyntaxError, ImportError, ...),
    which the caller treats as a compile failure -> VIOLATED.
    """

    source = render_candidate_source(task, expression)
    return _build_namespace(source)


def label_calls(namespace: dict, calls: Iterable[str]) -> list[Label]:
    """Evaluate each call expression and reduce it to a comparable label."""

    labels: list[Label] = []
    for call in calls:
        try:
            value = eval(call, namespace)  # noqa: S307 - restricted builtins only
        except BaseException as exc:  # noqa: BLE001 - candidate may raise anything
            labels.append((_RAISE, type(exc).__name__))
        else:
            try:
                labels.append((_VALUE, bool(value)))
            except BaseException:  # noqa: BLE001 - hostile __bool__
                labels.append((_VALUE, False))
    return labels


def label_candidate(task: dict, expression: str, calls: list[str]) -> list[Label] | None:
    """Compile + label one candidate.  ``None`` means the candidate did not compile."""

    try:
        namespace = compile_candidate(task, expression)
    except BaseException:  # noqa: BLE001 - any compile-time failure
        return None
    return label_calls(namespace, calls)


def agrees(labels: list[list[Label]]) -> tuple[list[bool], list[Label | None]]:
    """Strict differential oracle over implementation labels.

    A case is *decided* only when every implementation returns the same label
    **and** that label is a value (not an exception).  A case where any
    implementation raises is excluded rather than adjudicated: the written task
    contracts define admission/refusal, not crash behaviour, so an exception is
    evidence that the case is outside the contract's domain, not that the
    majority is right.
    """

    if not labels:
        return [], []
    width = len(labels[0])
    decided: list[bool] = []
    expected: list[Label | None] = []
    for index in range(width):
        column = [impl[index] for impl in labels]
        first = column[0]
        uniform = all(item == first for item in column)
        ok = uniform and first[0] == _VALUE
        decided.append(ok)
        expected.append(first if ok else None)
    return decided, expected


__all__ = [
    "Label",
    "agrees",
    "compile_candidate",
    "label_calls",
    "label_candidate",
]
