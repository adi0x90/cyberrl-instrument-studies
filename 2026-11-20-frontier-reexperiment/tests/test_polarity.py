"""Unit tests for the conservative polarity analysis in harness.polarity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.polarity import CONVERSE, CORRECT, UNKNOWN, converse_guard, explain  # noqa: E402


def polarity(correct: object, converse: object, fact: str = "test fact") -> dict:
    return {"fact": fact, "correct_marker": correct, "converse_marker": converse}


TENANT = polarity(["=="], ["!="])
NONE = polarity(["is not None"], ["is None"])
MEMBERSHIP = polarity([" in "], ["not in"])
PREFIX = polarity(["startswith"], ["endswith"])


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("req.tenant_id == resource.tenant_id", CORRECT),
        ("req.tenant_id != resource.tenant_id", CONVERSE),
        ("self.tenant_id == resource.tenant_id and self.can_write", CORRECT),
    ],
)
def test_equality_versus_inequality(expression: str, expected: str) -> None:
    assert converse_guard(expression, TENANT) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("value is not None", CORRECT),
        ("value is None", CONVERSE),
        ("obj.value is not None and obj.ready", CORRECT),
    ],
)
def test_identity_versus_negated_identity(expression: str, expected: str) -> None:
    assert converse_guard(expression, NONE) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("user in allowed_users", CORRECT),
        ("user not in allowed_users", CONVERSE),
    ],
)
def test_membership_versus_negated_membership(expression: str, expected: str) -> None:
    assert converse_guard(expression, MEMBERSHIP) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("s.startswith(prefix)", CORRECT),
        ("s.endswith(prefix)", CONVERSE),
        ("not s.startswith(prefix)", CONVERSE),
        ("not s.endswith(prefix)", CORRECT),
    ],
)
def test_startswith_versus_endswith(expression: str, expected: str) -> None:
    assert converse_guard(expression, PREFIX) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "foo(request)",  # a call, but not a declarative predicate
        "x + 1",  # arithmetic only
        "True",  # bare constant
    ],
)
def test_expressions_without_a_declarative_predicate_are_unknown(expression: str) -> None:
    assert converse_guard(expression, TENANT) == UNKNOWN


def test_conflicting_markers_are_unknown() -> None:
    assert converse_guard("x == y or x != z", TENANT) == UNKNOWN


def test_missing_polarity_markers_are_unknown() -> None:
    assert converse_guard("x == y", {}) == UNKNOWN
    assert converse_guard("x == y", {"fact": "no markers"}) == UNKNOWN


@pytest.mark.parametrize("expression", ["x ==", "", "   ", None, 42])
def test_unusable_expressions_are_unknown(expression: object) -> None:
    assert converse_guard(expression, TENANT) == UNKNOWN  # type: ignore[arg-type]


def test_string_literals_do_not_leak_markers() -> None:
    """A marker inside a string literal must not create false evidence."""

    expression = 'user in allowed_users and note != "not in the list"'
    assert converse_guard(expression, MEMBERSHIP) == CORRECT


def test_longest_marker_wins_over_a_substring_marker() -> None:
    # "not in" claims its span before the shorter " in " can match inside it.
    assert converse_guard("user not in allowed_users", MEMBERSHIP) == CONVERSE
    assert converse_guard("value is not None", NONE) == CORRECT


def test_word_markers_do_not_match_inside_identifiers() -> None:
    """A bare "in" marker must match the operator, not ``principal``."""

    markers = polarity(["in"], ["not in"])
    assert converse_guard("op.scope in principal.scopes", markers) == CORRECT
    assert converse_guard("op.scope not in principal.scopes", markers) == CONVERSE


def test_nested_negation_flips_the_inner_polarity() -> None:
    assert converse_guard("not (x == y)", TENANT) == CONVERSE
    assert converse_guard("not (x != y)", TENANT) == CORRECT
    assert converse_guard("not not (x == y)", TENANT) == CORRECT


def test_top_level_negation_of_an_unknown_remains_unknown() -> None:
    assert converse_guard("not foo(x)", TENANT) == UNKNOWN


def test_markers_accept_a_single_string_as_well_as_a_list() -> None:
    markers = polarity("==", "!=")
    assert converse_guard("a == b", markers) == CORRECT
    assert converse_guard("a != b", markers) == CONVERSE


# --- explain ---------------------------------------------------------------


def test_explain_describes_comparisons() -> None:
    notes = explain("req.tenant_id == resource.tenant_id")
    assert isinstance(notes, list)
    assert any("comparison" in note for note in notes)


def test_explain_describes_predicate_calls() -> None:
    notes = explain("s.startswith(prefix)")
    assert any("predicate call" in note for note in notes)


def test_explain_reports_missing_predicates() -> None:
    notes = explain("foo(x)")
    assert any("no declarative comparison" in note for note in notes)


def test_explain_reports_parse_failure() -> None:
    notes = explain("x ==")
    assert any("does not parse" in note for note in notes)


def test_explain_reports_top_level_negation() -> None:
    notes = explain("not (x == y)")
    assert any("top-level negation" in note for note in notes)


@pytest.mark.parametrize("expression", ["", "   ", None, 3])
def test_explain_never_raises(expression: object) -> None:
    notes = explain(expression)  # type: ignore[arg-type]
    assert isinstance(notes, list)
    assert notes


# --- operator-boundary regression (2026-11-20 E3 audit) ---------------------
# The E3 run mislabelled `a >= b` as CONVERSE for a task whose converse marker is
# `>`, because `>` matched inside `>=`. These tests pin the fix.

def test_composite_operators_do_not_match_their_prefix():
    pol = {"correct_marker": ["<="], "converse_marker": [">"]}
    assert converse_guard("a <= b", pol) == "CORRECT"
    assert converse_guard("a >= b", pol) == "UNKNOWN"
    assert converse_guard("a > b", pol) == "CONVERSE"
    assert converse_guard("a < b", pol) == "UNKNOWN"


def test_equality_is_not_matched_by_inequality_markers():
    pol = {"correct_marker": ["=="], "converse_marker": ["!="]}
    assert converse_guard("x == y", pol) == "CORRECT"
    assert converse_guard("x != y", pol) == "CONVERSE"
    assert converse_guard("x >= y", pol) == "UNKNOWN"
