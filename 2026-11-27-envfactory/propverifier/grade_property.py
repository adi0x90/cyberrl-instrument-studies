"""A1 — property-hardened Tier-1 verifier with a non-circular oracle.

Design notes that matter for reading the results:

* **Nothing here is a result.**  This module implements the preregistered
  instrument; ``run_a1.py`` produces the evidence.
* **The oracle is differential, not authored.**  A case is labelled by executing
  independent valid implementations and keeping only cases where every in-oracle
  implementation agrees on a *value*.  Cases where any implementation raises, or
  where implementations disagree, are excluded and counted.  The generator emits
  inputs only, so no single implementation's semantics becomes ground truth.
* **One alternate is always held out.**  ``build_oracle(..., held_out=k)`` never
  sees ``alternate_fixes[k]``; that alternate is then scored blind.  For
  publication the default grading oracle holds out index 2, and the 3-fold
  rotation in ``run_a1.py`` reports each held-out alternate separately.
* **UNKNOWN is not a kill.**  A bad candidate is killed only on ``VIOLATED``.  A
  grader that answers ``UNKNOWN`` everywhere therefore produces zero kills and
  zero decisive coverage, which the accounting in ``run_a1.py`` makes visible.
* **Latency is not multiplied.**  ``grade_property_tier1`` evaluates the full
  generated case set on each call; the reported per-candidate delta is already
  per call and must not be scaled by cases-per-episode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import fastrun, gen
from .harness.graders import HOLDS, UNKNOWN, VIOLATED, grade_obligation
from .harness.tasks_io import ExecutionResult

DEFAULT_CASES = 200
DEFAULT_HELD_OUT = 2


@dataclass(frozen=True)
class Oracle:
    task_id: str
    seed: int
    held_out: int
    calls: list[str]
    decided: list[bool]
    expected: list[Any]
    impls: list[str] = field(default_factory=list)
    build_seconds: float = 0.0

    @property
    def decided_count(self) -> int:
        return sum(1 for flag in self.decided if flag)

    @property
    def ambiguous_fraction(self) -> float:
        if not self.calls:
            return 0.0
        return 1.0 - (self.decided_count / len(self.calls))


def _implementations(task: dict, held_out: int) -> list[tuple[str, str]]:
    impls: list[tuple[str, str]] = [("reference", task["reference_fix"])]
    for index, expression in enumerate(task.get("alternate_fixes") or []):
        if index == held_out:
            continue
        impls.append((f"alternate[{index}]", expression))
    return impls


def build_oracle(
    task: dict,
    seed: int,
    held_out: int = DEFAULT_HELD_OUT,
    n_cases: int = DEFAULT_CASES,
) -> Oracle:
    """Build the differential oracle, never consulting the held-out alternate."""

    started = time.perf_counter()
    calls = gen.generate_calls(task["id"], seed, n_cases)
    impls = _implementations(task, held_out)
    per_impl_labels = [fastrun.label_candidate(task, expression, calls) for _, expression in impls]
    usable = [labels for labels in per_impl_labels if labels is not None]
    if len(usable) != len(impls):
        # An in-oracle implementation that cannot compile is a corpus defect, not
        # a candidate failure.  Refuse to build a silently weaker oracle.
        broken = [name for (name, _), labels in zip(impls, per_impl_labels) if labels is None]
        raise ValueError(f"in-oracle implementation(s) failed to compile: {broken}")
    decided, expected = fastrun.agrees(usable)  # type: ignore[arg-type]
    return Oracle(
        task_id=task["id"],
        seed=seed,
        held_out=held_out,
        calls=calls,
        decided=decided,
        expected=expected,
        impls=[name for name, _ in impls],
        build_seconds=time.perf_counter() - started,
    )


@dataclass
class PropertyStats:
    task_id: str = ""
    seed: int = 0
    cases: int = 0
    decided: int = 0
    ambiguous: int = 0
    ambiguous_fraction: float = 0.0
    property_mismatches: int = 0
    metamorphic_pairs: int = 0
    metamorphic_violations: int = 0
    first_counterexample: str | None = None
    first_metamorphic: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "decided": self.decided,
            "ambiguous": self.ambiguous,
            "ambiguous_fraction": round(self.ambiguous_fraction, 6),
            "property_mismatches": self.property_mismatches,
            "metamorphic_pairs": self.metamorphic_pairs,
            "metamorphic_violations": self.metamorphic_violations,
            "first_counterexample": self.first_counterexample,
            "first_metamorphic": self.first_metamorphic,
        }


def grade_property_tier1(
    task: dict,
    result: ExecutionResult,
    expression: str | None,
    seed: int,
    *,
    oracle: Oracle | None = None,
    n_cases: int = DEFAULT_CASES,
) -> tuple[str, list[str], PropertyStats]:
    """Three-valued property-hardened verdict.

    Combines (a) the unchanged static/behavioural obligation checks over the
    task's finite fixed cases with (b) seeded randomized cases scored against a
    differential oracle and (c) seeded metamorphic pairs that must not change the
    verdict.
    """

    stats = PropertyStats(task_id=task.get("id", ""), seed=seed)

    static_verdict, static_reasons = grade_obligation(task, result, expression)
    if static_verdict == UNKNOWN:
        return UNKNOWN, static_reasons, stats
    if static_verdict == VIOLATED:
        return VIOLATED, static_reasons, stats

    assert expression is not None  # grade_obligation already rejected None
    active_oracle = oracle or build_oracle(task, seed, DEFAULT_HELD_OUT, n_cases)
    stats.cases = len(active_oracle.calls)
    stats.decided = active_oracle.decided_count
    stats.ambiguous = stats.cases - stats.decided
    stats.ambiguous_fraction = active_oracle.ambiguous_fraction

    labels = fastrun.label_candidate(task, expression, active_oracle.calls)
    if labels is None:
        return VIOLATED, ["COMPILE: property case set could not be compiled"], stats

    reasons: list[str] = []
    for index, label in enumerate(labels):
        if not active_oracle.decided[index]:
            continue
        expected = active_oracle.expected[index]
        if label != expected:
            stats.property_mismatches += 1
            if stats.first_counterexample is None:
                stats.first_counterexample = (
                    f"{active_oracle.calls[index]} -> {label[1]!r}, oracle {expected[1]!r}"
                )
    if stats.property_mismatches:
        reasons.append(
            f"PROPERTY: {stats.property_mismatches}/{stats.decided} oracle-decided "
            f"randomized case(s) disagreed (first: {stats.first_counterexample})"
        )

    pairs = gen.METAMORPHIC[task["id"]](seed)
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_labels = fastrun.label_candidate(task, expression, left) or []
    right_labels = fastrun.label_candidate(task, expression, right) or []
    stats.metamorphic_pairs = len(pairs)
    for index in range(min(len(left_labels), len(right_labels))):
        if left_labels[index] != right_labels[index]:
            stats.metamorphic_violations += 1
            if stats.first_metamorphic is None:
                stats.first_metamorphic = (
                    f"{left[index]} -> {left_labels[index][1]!r} vs "
                    f"{right[index]} -> {right_labels[index][1]!r}"
                )
    if stats.metamorphic_violations:
        reasons.append(
            f"METAMORPHIC: {stats.metamorphic_violations}/{stats.metamorphic_pairs} "
            f"invariant pair(s) disagreed (first: {stats.first_metamorphic})"
        )

    if reasons:
        return VIOLATED, reasons, stats
    if stats.decided == 0:
        return UNKNOWN, ["NO_DECIDED_CASES: every generated case was oracle-ambiguous"], stats
    return HOLDS, ["all obligation + property + metamorphic checks satisfied"], stats


# ---------------------------------------------------------------------------
# Confusion / coverage accounting used by the 3x2 matrix
# ---------------------------------------------------------------------------


@dataclass
class Matrix:
    """Full 3-valued verdict matrix over {bad, valid} candidates."""

    rows: dict[str, dict[str, int]] = field(default_factory=lambda: {HOLDS: {}, VIOLATED: {}, UNKNOWN: {}})

    def add(self, truth: str, verdict: str) -> None:
        bucket = self.rows.setdefault(verdict, {})
        bucket[truth] = bucket.get(truth, 0) + 1

    def count(self, truth: str, verdict: str) -> int:
        return self.rows.get(verdict, {}).get(truth, 0)

    def total(self, truth: str) -> int:
        return sum(self.count(truth, verdict) for verdict in (HOLDS, VIOLATED, UNKNOWN))

    @property
    def kills(self) -> int:
        return self.count("bad", VIOLATED)

    @property
    def bad_total(self) -> int:
        return self.total("bad")

    @property
    def kill_rate(self) -> float:
        return self.kills / self.bad_total if self.bad_total else 0.0

    @property
    def decisive_coverage(self) -> float:
        """1 - P(UNKNOWN), over every graded candidate."""

        total = self.total("bad") + self.total("valid")
        unknown = self.count("bad", UNKNOWN) + self.count("valid", UNKNOWN)
        return 1.0 - (unknown / total) if total else 0.0

    @property
    def false_reject_rate(self) -> float:
        valid = self.total("valid")
        return self.count("valid", VIOLATED) / valid if valid else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": {verdict: dict(bucket) for verdict, bucket in self.rows.items()},
            "kills": self.kills,
            "bad_total": self.bad_total,
            "kill_rate": round(self.kill_rate, 6),
            "decisive_coverage": round(self.decisive_coverage, 6),
            "false_reject_rate": round(self.false_reject_rate, 6),
        }


def candidate_truth(candidate: dict) -> str:
    return "valid" if candidate.get("kind") in {"reference", "alternate"} else "bad"


__all__ = [
    "DEFAULT_CASES",
    "DEFAULT_HELD_OUT",
    "Matrix",
    "Oracle",
    "PropertyStats",
    "build_oracle",
    "candidate_truth",
    "grade_property_tier1",
]
