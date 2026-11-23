"""Tests for the three admission consumers in harness.envelopes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.envelopes import Admission, admit_fence, admit_strict, admit_toolcall  # noqa: E402


# Text that the strict consumer must admit.  Used for the monotonicity property.
STRICT_ACCEPTED = [
    '{"expression": "x == y"}',
    '  {"expression": "x == y"}  \n',
    '{"expression": " x == y "}',
    '{"expression": "True and (a == b)"}',
    '{"expression": "s.startswith(\\"a\\")"}',
    '{"expression": "a in b"}',
    '{"expression": "x is not None"}',
    '{"expression": "\\u00e9"}',
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"expression": "x == y"}', "x == y"),
        ('  {"expression": "x == y"}  \n', "x == y"),
        ('{"expression": " x == y "}', "x == y"),
        ('{"expression": "True and (a == b)"}', "True and (a == b)"),
        ('{"expression": "s.startswith(\\"a\\")"}', 's.startswith("a")'),
        ('{"expression": "\\u00e9"}', "\u00e9"),
    ],
)
def test_strict_admits_one_json_object(text: str, expected: str) -> None:
    admission = admit_strict(text)
    assert admission.ok is True
    assert admission.expression == expected
    assert admission.envelope == "strict"
    assert admission.reason


def test_strict_parses_but_does_not_judge_ast_wrapper() -> None:
    """Shape grading belongs to graders; the strict envelope only checks framing."""

    admission = admit_strict('{"expression": "True and (req.tenant_id == resource.tenant_id)"}')
    assert admission.ok is True
    assert admission.expression == "True and (req.tenant_id == resource.tenant_id)"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\t ",
        'Here is the fix: {"expression": "x == y"}',
        '{"expression": "x == y"} and that is all',
        '{"expression": "x"}{"expression": "y"}',
        "```json\n{\"expression\": \"x == y\"}\n```",
        '{"expression": 1}',
        '{"expression": null}',
        '{"expression": ""}',
        '{"expression": "   "}',
        '{"expression": "x", "note": "extra field"}',
        '{"expr": "x"}',
        '["x"]',
        '"just a string"',
        '{"expression": "x", "expression": "y"}',
    ],
)
def test_strict_rejects_everything_else(text: str) -> None:
    admission = admit_strict(text)
    assert admission.ok is False
    assert admission.expression is None
    assert admission.reason
    assert admission.envelope == "strict"


def test_strict_rejects_markdown_fence_specifically() -> None:
    admission = admit_strict('```json\n{"expression": "x == y"}\n```')
    assert admission.ok is False
    assert "fence" in admission.reason


# --- fence -----------------------------------------------------------------


@pytest.mark.parametrize("text", STRICT_ACCEPTED)
def test_strict_admits_imply_fence_admits(text: str) -> None:
    """Monotonicity: every strictly admitted response is fence admitted."""

    strict = admit_strict(text)
    fence = admit_fence(text)
    assert strict.ok is True
    assert fence.ok is True
    assert fence.expression == strict.expression
    assert fence.envelope == "fence"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"expression": "x == y"}\n```', "x == y"),
        ('```json\n{"expression": "True and (a == b)"}\n```', "True and (a == b)"),
        ('```json\r\n{"expression": "x == y"}\r\n```', "x == y"),
    ],
)
def test_fence_admits_one_outer_json_fence(text: str, expected: str) -> None:
    admission = admit_fence(text)
    assert admission.ok is True
    assert admission.expression == expected
    assert admission.envelope == "fence"


def test_fence_rejects_two_fences() -> None:
    text = (
        '```json\n{"expression": "x == y"}\n```\n'
        '```json\n{"expression": "a == b"}\n```'
    )
    admission = admit_fence(text)
    assert admission.ok is False
    assert "more than one code fence" in admission.reason


@pytest.mark.parametrize(
    "text",
    [
        'Here you go:\n```json\n{"expression": "x == y"}\n```',
        '```json\n{"expression": "x == y"}\n```\nDone.',
        "```JSON\n{\"expression\": \"x == y\"}\n```",
        "```\n{\"expression\": \"x == y\"}\n```",
        "```json\nx == y\n```",
        '```json\n{"expression": "x", "note": "y"}\n```',
        '```json\n{"expression": "x"}',
        "x == y",
    ],
)
def test_fence_rejects_non_monotone_shapes(text: str) -> None:
    admission = admit_fence(text)
    assert admission.ok is False
    assert admission.expression is None
    assert admission.reason


def test_fence_does_not_accept_prose_inside_fence() -> None:
    admission = admit_fence('```json\nHere is the fix\n```')
    assert admission.ok is False
    assert "fenced payload rejected" in admission.reason


# --- toolcall --------------------------------------------------------------

UNIFIED_DIFF = """--- a/challenge.py
+++ b/challenge.py
@@ -1,3 +1,3 @@
 def check(req, resource):
-    return REPLACE_ME
+    return req.tenant_id == resource.tenant_id and req.can_write
"""


def test_toolcall_extracts_from_unified_diff() -> None:
    admission = admit_toolcall(UNIFIED_DIFF)
    assert admission.ok is True
    assert admission.expression == "req.tenant_id == resource.tenant_id and req.can_write"
    assert admission.envelope == "toolcall"


def test_toolcall_extracts_from_diff_that_rewrites_the_function() -> None:
    diff = (
        "diff --git a/challenge.py b/challenge.py\n"
        "--- a/challenge.py\n"
        "+++ b/challenge.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def check(req, resource):\n"
        "-    return REPLACE_ME\n"
        "+def check(req, resource):\n"
        "+    return req.can_write and req.tenant_id == resource.tenant_id\n"
    )
    admission = admit_toolcall(diff)
    assert admission.ok is True
    assert admission.expression == "req.can_write and req.tenant_id == resource.tenant_id"


def test_toolcall_ignores_trailing_comment_on_added_line() -> None:
    diff = (
        "@@ -1,2 +1,2 @@\n"
        " def check(req, resource):\n"
        "-    return REPLACE_ME\n"
        "+    return req.tenant_id == resource.tenant_id  # same tenant only\n"
    )
    admission = admit_toolcall(diff)
    assert admission.ok is True
    assert admission.expression == "req.tenant_id == resource.tenant_id"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "```python\ndef check(req, resource):\n    return req.tenant_id == resource.tenant_id\n```",
            "req.tenant_id == resource.tenant_id",
        ),
        ("```python\nreturn req.can_write\n```", "req.can_write"),
        ("```\nreturn x == y\n```", "x == y"),
        ("```python\nx == y\n```", "x == y"),
        (
            '```json\n{"expression": "x == y"}\n```',
            "x == y",
        ),
        (
            "```diff\n-    return REPLACE_ME\n+    return x == y\n```",
            "x == y",
        ),
        (
            "Use this patch:\n```python\nreturn x == y\n```\n",
            "x == y",
        ),
        ('{"expression": "x == y"}', "x == y"),
    ],
)
def test_toolcall_accepts_fenced_blocks_and_tool_arguments(text: str, expected: str) -> None:
    admission = admit_toolcall(text)
    assert admission.ok is True
    assert admission.expression == expected
    assert admission.envelope == "toolcall"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think the fix is to compare the tenant ids.",
        "--- a/challenge.py\n+++ b/challenge.py\n@@ -1 +1 @@\n-    return REPLACE_ME\n",
        "```python\nx = 1\ny = 2\n```",
        "```python\n```",
        '```json\n{"expression": "x", "note": "y"}\n```',
        "```python\nreturn x == y",
        "```python\nreturn x == y\n```\n```python\nreturn a == b\n```",
        "@@ -1,2 +1,2 @@\n-    return REPLACE_ME\n+    return x == y\n+    return a == b\n",
    ],
)
def test_toolcall_rejects_what_it_cannot_extract(text: str) -> None:
    admission = admit_toolcall(text)
    assert admission.ok is False
    assert admission.expression is None
    assert admission.reason
    assert admission.envelope == "toolcall"


def test_toolcall_reports_ambiguity_reason() -> None:
    text = "@@ -1,2 +1,2 @@\n-    return REPLACE_ME\n+    return x == y\n+    return a == b\n"
    admission = admit_toolcall(text)
    assert admission.ok is False
    assert "ambiguous" in admission.reason


def test_admission_is_frozen() -> None:
    admission = Admission(True, "x == y", "ok", "strict")
    with pytest.raises(Exception):
        admission.ok = False  # type: ignore[misc]
