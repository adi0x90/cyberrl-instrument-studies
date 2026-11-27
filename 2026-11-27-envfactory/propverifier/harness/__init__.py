"""Harness for the 2026-11-20 frontier re-experiment.

Modules
-------
models      OpenRouter chat client with a fully offline, deterministic mock mode.
envelopes   Three admission consumers (strict JSON, one ```json fence, tool-call/diff).
tasks_io    Task loading/validation, placeholder substitution, sandboxed case execution.
graders     Deliberately weak baseline grader and the obligation grader.
polarity    Conservative static classification of a guard's polarity.

The public signatures are fixed by ``docs/SPEC.md``; do not change them without
editing that file first.
"""

__all__ = ["models", "envelopes", "tasks_io", "graders", "polarity"]
__version__ = "1.0.0"
