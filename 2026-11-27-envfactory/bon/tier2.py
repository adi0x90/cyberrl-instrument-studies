"""Tier-2 for the expression control: property + metamorphic checks only.

Deliberately **without** the static/finite-fixture arm.  Tier-1 is the task's fixed
cases plus the AST/scope obligations; Tier-2 is the seeded randomized obligations.
Sharing the fixed cases between the two would make "Tier-2 refutes a Tier-1 accept"
partly a restatement of Tier-1.

Independence caveat, stated up front: both tiers were authored by the same person,
so this is a *distribution-disjoint* check, not an independently authored one.  The
disjoint-obligation arm is the native pack, where Tier-2 is the pack author's own
`variants/` artifacts.
"""

from __future__ import annotations

from propverifier import fastrun, gen
from propverifier.grade_property import Oracle, build_oracle

HOLDS = "HOLDS"
VIOLATED = "VIOLATED"
UNKNOWN = "UNKNOWN"


def tier2_expression(task: dict, expression: str | None, oracle: Oracle) -> tuple[str, list[str]]:
    if not expression or not expression.strip():
        return UNKNOWN, ["NO_EXPRESSION"]

    labels = fastrun.label_candidate(task, expression, oracle.calls)
    if labels is None:
        return VIOLATED, ["COMPILE: property case set did not compile"]

    reasons: list[str] = []
    for index, label in enumerate(labels):
        if not oracle.decided[index]:
            continue
        if label != oracle.expected[index]:
            call = oracle.calls[index]
            reasons.append(f"PROPERTY: {call} -> {label[1]!r}, oracle {oracle.expected[index][1]!r}")
            break

    pairs = gen.METAMORPHIC[task["id"]](oracle.seed)
    left = fastrun.label_candidate(task, expression, [pair[0] for pair in pairs]) or []
    right = fastrun.label_candidate(task, expression, [pair[1] for pair in pairs]) or []
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            reasons.append(f"METAMORPHIC: pair {index} disagreed")
            break

    if reasons:
        return VIOLATED, reasons
    if oracle.decided_count == 0:
        return UNKNOWN, ["no decided cases"]
    return HOLDS, ["property + metamorphic satisfied"]


def oracle_for(task: dict, seed: int) -> Oracle:
    return build_oracle(task, seed)


__all__ = ["HOLDS", "UNKNOWN", "VIOLATED", "oracle_for", "tier2_expression"]
