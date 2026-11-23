"""Three admission consumers over the same model output bytes.

The three envelopes are deliberately ordered by permissiveness:

* :func:`admit_strict` -- the whole response must be exactly one JSON object with
  exactly one field, ``expression``, holding a non-empty string.  No prose, no
  markdown, no duplicate keys, no extra fields.  This consumer does **not** judge
  the expression (an AST wrapper such as ``True and (x == y)`` is admitted here);
  shape grading belongs to :mod:`harness.graders`.
* :func:`admit_fence` -- exactly one outer lowercase ```` ```json ```` fence
  wrapped around a strict payload.  It first tries strict admission, so it is
  monotone over :func:`admit_strict` by construction.
* :func:`admit_toolcall` -- the loosest consumer: a bare JSON tool argument, a
  single fenced code block (JSON, Python, or a bare diff hunk), or a unified diff.
  It extracts the replacement expression from an added ``return``/expression line.

Every consumer returns an :class:`Admission`; failures carry a human-readable
reason instead of raising.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

STRICT_ENVELOPE = "strict"
FENCE_ENVELOPE = "fence"
TOOLCALL_ENVELOPE = "toolcall"


@dataclass(frozen=True)
class Admission:
    ok: bool
    expression: str | None
    reason: str
    envelope: str


class _EnvelopeError(ValueError):
    """Internal: an envelope-specific structural failure."""


# ---------------------------------------------------------------------------
# Shared strict payload validation
# ---------------------------------------------------------------------------


class _DuplicateKey(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKey(f"duplicate JSON key(s) in {sorted(keys)}")
    return dict(pairs)


def _admit(payload: object, envelope: str) -> Admission:
    if not isinstance(payload, dict):
        return Admission(
            ok=False,
            expression=None,
            reason=(
                "top-level JSON value must be an object with a single field "
                f"'expression' (got {type(payload).__name__})"
            ),
            envelope=envelope,
        )

    fields = set(payload)
    if fields != {"expression"}:
        missing = sorted({"expression"} - fields)
        extra = sorted(fields - {"expression"})
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unexpected field(s) {extra}")
        return Admission(
            ok=False,
            expression=None,
            reason="JSON object must contain exactly one field, 'expression': "
            + "; ".join(detail),
            envelope=envelope,
        )

    expression = payload["expression"]
    if not isinstance(expression, str):
        return Admission(
            ok=False,
            expression=None,
            reason=f"'expression' must be a string (got {type(expression).__name__})",
            envelope=envelope,
        )
    expression = expression.strip()
    if not expression:
        return Admission(
            ok=False,
            expression=None,
            reason="'expression' must be a non-empty string",
            envelope=envelope,
        )

    return Admission(
        ok=True,
        expression=expression,
        reason="one JSON object with a single non-empty string field 'expression'",
        envelope=envelope,
    )


def _parse_strict(text: str, envelope: str) -> Admission:
    if not isinstance(text, str):
        raise TypeError(f"envelope input must be str, not {type(text).__name__}")
    trimmed = text.strip()
    if not trimmed:
        return Admission(False, None, "empty response", envelope)
    if trimmed.startswith("```"):
        return Admission(
            False,
            None,
            "markdown code fence is not allowed by the strict envelope",
            envelope,
        )
    try:
        payload = json.loads(trimmed, object_pairs_hook=_no_duplicate_keys)
    except _DuplicateKey as exc:
        return Admission(False, None, f"invalid JSON object: {exc}", envelope)
    except json.JSONDecodeError as exc:
        return Admission(
            False,
            None,
            "not exactly one JSON object: "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            envelope,
        )
    return _admit(payload, envelope)


# ---------------------------------------------------------------------------
# Public consumers
# ---------------------------------------------------------------------------


def admit_strict(text: str) -> Admission:
    """Admit exactly one JSON object with a single string field ``expression``."""

    return _parse_strict(text, STRICT_ENVELOPE)


def admit_fence(text: str) -> Admission:
    """Strip exactly one outer lowercase ```` ```json ```` fence, then admit strict.

    Anything :func:`admit_strict` admits is admitted here unchanged, so this
    consumer is monotone over the strict one.
    """

    if not isinstance(text, str):
        raise TypeError(f"envelope input must be str, not {type(text).__name__}")

    strict = _parse_strict(text, STRICT_ENVELOPE)
    if strict.ok:
        return Admission(True, strict.expression, strict.reason, FENCE_ENVELOPE)

    trimmed = text.strip()
    if "```" not in trimmed:
        return Admission(
            False,
            None,
            f"no ```json fence and strict admission failed: {strict.reason}",
            FENCE_ENVELOPE,
        )

    body, error = _strip_single_json_fence(trimmed)
    if error is not None:
        return Admission(False, None, error, FENCE_ENVELOPE)

    inner = _parse_strict(body, STRICT_ENVELOPE)
    if inner.ok:
        return Admission(
            True,
            inner.expression,
            "one ```json fence wrapping " + inner.reason,
            FENCE_ENVELOPE,
        )
    return Admission(False, None, f"fenced payload rejected: {inner.reason}", FENCE_ENVELOPE)


def _strip_single_json_fence(text: str) -> tuple[str | None, str | None]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "```json":
        return None, "outer fence must open with exactly ```json on its own line"
    if len(lines) < 2 or lines[-1].strip() != "```":
        return None, "outer fence must close with exactly ``` and nothing may follow it"

    body_lines = lines[1:-1]
    for line in body_lines:
        if line.strip().startswith("```"):
            return None, "more than one code fence in the response"
    if not "".join(body_lines).strip():
        return None, "fenced payload is empty"
    return "\n".join(body_lines), None


def admit_toolcall(text: str) -> Admission:
    """Admit a JSON tool argument, one fenced code block, or a unified diff.

    The replacement expression is extracted from the block's single expression
    statement / ``return`` value, or from the diff's added lines.  If no
    unambiguous expression can be recovered the admission fails with a reason.
    """

    if not isinstance(text, str):
        raise TypeError(f"envelope input must be str, not {type(text).__name__}")

    trimmed = text.strip()
    if not trimmed:
        return Admission(False, None, "empty response", TOOLCALL_ENVELOPE)

    # 1. A bare JSON tool argument is always acceptable.
    strict = _parse_strict(trimmed, STRICT_ENVELOPE)
    if strict.ok:
        return Admission(
            True,
            strict.expression,
            "JSON tool argument: " + strict.reason,
            TOOLCALL_ENVELOPE,
        )

    # 2. Exactly one fenced code block, optional prose around it.
    try:
        block = _extract_single_fence(trimmed)
    except _EnvelopeError as exc:
        return Admission(False, None, str(exc), TOOLCALL_ENVELOPE)
    if block is not None:
        info, body = block
        return _admit_block(body, info)

    # 3. A raw unified diff with no fence.
    if _looks_like_diff(trimmed):
        expression, error = _expression_from_diff(trimmed)
        if expression is not None:
            return Admission(
                True,
                expression,
                "unified diff: extracted the added replacement expression",
                TOOLCALL_ENVELOPE,
            )
        return Admission(
            False,
            None,
            f"unified diff did not contain an extractable expression: {error}",
            TOOLCALL_ENVELOPE,
        )

    return Admission(
        False,
        None,
        "response is neither a JSON tool argument, a single fenced code block, nor a "
        "unified diff",
        TOOLCALL_ENVELOPE,
    )


# ---------------------------------------------------------------------------
# Fenced block handling
# ---------------------------------------------------------------------------


def _extract_single_fence(text: str) -> tuple[str, str] | None:
    """Return ``(info_string, body)`` for exactly one fenced block, else ``None``.

    Raises ``_EnvelopeError`` when a fence is unterminated or more than one
    fenced block is present, since neither is an unambiguous tool call.
    """

    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("```"):
            index += 1
            continue
        info = stripped[3:].strip()
        close = index + 1
        while close < len(lines) and lines[close].strip() != "```":
            close += 1
        if close >= len(lines):
            raise _EnvelopeError("unterminated fenced code block")
        blocks.append((info, "\n".join(lines[index + 1 : close])))
        index = close + 1

    if not blocks:
        return None
    if len(blocks) > 1:
        raise _EnvelopeError(
            f"expected a single fenced code block, found {len(blocks)}"
        )
    return blocks[0]


def _admit_block(body: str, info: str) -> Admission:
    if not body.strip():
        return Admission(False, None, "fenced code block is empty", TOOLCALL_ENVELOPE)

    # JSON payload inside a fence.
    strict = _parse_strict(body, STRICT_ENVELOPE)
    if strict.ok:
        return Admission(
            True,
            strict.expression,
            f"fenced code block ({info or 'no info string'}): " + strict.reason,
            TOOLCALL_ENVELOPE,
        )

    # A block that is JSON-shaped but not a valid payload must not be silently
    # reinterpreted as a Python dict literal.
    if body.strip().startswith(("{", "[")):
        if not _looks_like_diff(body):
            return Admission(
                False,
                None,
                f"fenced JSON rejected: {strict.reason}",
                TOOLCALL_ENVELOPE,
            )

    if _looks_like_diff(body):
        expression, error = _expression_from_diff(body)
        if expression is not None:
            return Admission(
                True,
                expression,
                f"fenced diff ({info or 'no info string'}): extracted the added "
                "replacement expression",
                TOOLCALL_ENVELOPE,
            )
        return Admission(
            False,
            None,
            f"fenced diff did not contain an extractable expression: {error}",
            TOOLCALL_ENVELOPE,
        )

    expression, error = _expression_from_code(body)
    if expression is None:
        return Admission(
            False,
            None,
            f"fenced code block did not contain an unambiguous expression: {error}",
            TOOLCALL_ENVELOPE,
        )
    return Admission(
        True,
        expression,
        f"fenced code block ({info or 'no info string'}): extracted the replacement "
        "expression",
        TOOLCALL_ENVELOPE,
    )


# ---------------------------------------------------------------------------
# Diff handling
# ---------------------------------------------------------------------------


def _looks_like_diff(text: str) -> bool:
    lines = text.splitlines()
    if any(line.startswith("diff --git ") for line in lines):
        return True
    if any(line.startswith("@@") for line in lines):
        return True
    if any(line.startswith("+++ ") for line in lines) and any(
        line.startswith("--- ") for line in lines
    ):
        return True
    # Fenced bare hunk: at least one removed and one added line.
    has_minus = any(line.startswith("-") and not line.startswith("---") for line in lines)
    has_plus = any(line.startswith("+") and not line.startswith("+++") for line in lines)
    return has_minus and has_plus


def _added_lines(text: str) -> list[str]:
    added: list[str] = []
    for line in text.splitlines():
        if line.startswith(("+++", "---", "@@", "diff --git ", "index ")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
    return added


def _expression_from_diff(text: str) -> tuple[str | None, str | None]:
    added = _added_lines(text)
    if not added:
        return None, "the diff has no added (+) lines"

    added_source = "\n".join(added)
    expression, error = _expression_from_added_source(added_source)
    if expression is not None:
        return expression, None
    if error is not None:
        return None, error

    # Fallback: treat each added line independently (handles ``+    return x``).
    candidates: list[str] = []
    for line in added:
        candidate, _ = _expression_from_code(line)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        return None, "no added line contains an expression"
    if len(candidates) > 1:
        return None, f"ambiguous: {len(candidates)} candidate expressions in added lines"
    return candidates[0], None


def _expression_from_added_source(source: str) -> tuple[str | None, str | None]:
    """Parse a block of added lines as a module, or return ``(None, None)``.

    Returns ``(None, None)`` when the block is not module-parseable so the caller
    can fall back to line handling.
    """

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None, None

    returns = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    distinct_returns = list(dict.fromkeys(returns))
    if len(distinct_returns) == 1:
        return distinct_returns[0], None
    if len(distinct_returns) > 1:
        return None, (
            f"ambiguous: {len(distinct_returns)} distinct return expressions in added lines"
        )

    statements = [node for node in tree.body if isinstance(node, ast.Expr)]
    expressions = [
        ast.unparse(node.value)
        for node in statements
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
    ]
    if len(expressions) == 1:
        return expressions[0], None
    if len(expressions) > 1:
        return None, f"ambiguous: {len(expressions)} expression statements in added lines"
    return None, None


# ---------------------------------------------------------------------------
# Code block handling
# ---------------------------------------------------------------------------


_RETURN_LINE = re.compile(r"^return\b\s*(?P<expr>.+)$")


def _expression_from_code(body: str) -> tuple[str | None, str | None]:
    """Extract one expression from a code block (module, function, or one line)."""

    stripped = body.strip()
    if not stripped:
        return None, "empty code"

    # Whole block parses as a module: use its single return value or expression
    # statement.  This handles full function rewrites and multi-line expressions.
    try:
        tree = ast.parse(stripped)
    except (SyntaxError, ValueError):
        tree = None

    if tree is not None:
        returns = [
            ast.unparse(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Return) and node.value is not None
        ]
        distinct_returns = list(dict.fromkeys(returns))
        if len(distinct_returns) == 1:
            return distinct_returns[0], None
        if len(distinct_returns) > 1:
            return None, f"ambiguous: {len(distinct_returns)} distinct return expressions"

        expressions = [
            ast.unparse(node.value)
            for node in tree.body
            if isinstance(node, ast.Expr)
            and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        ]
        if len(expressions) == 1:
            return expressions[0], None
        if len(expressions) > 1:
            return None, f"ambiguous: {len(expressions)} expression statements"

    # Fall back to line handling: bare ``return <expr>`` lines (not module-legal)
    # or a single bare expression line.
    candidates: list[str] = []
    for raw_line in stripped.splitlines():
        line = _strip_trailing_comment(raw_line).strip()
        if not line or line.startswith("#"):
            continue
        match = _RETURN_LINE.match(line)
        candidate = match.group("expr").strip() if match else line
        if not _is_expression(candidate) or _is_pure_string_literal(candidate):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, f"ambiguous: {len(candidates)} candidate expressions in code block"
    return None, "no expression found"


def _is_expression(text: str) -> bool:
    try:
        ast.parse(text, mode="eval")
        return True
    except (SyntaxError, ValueError):
        return False


def _is_pure_string_literal(text: str) -> bool:
    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return False
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _strip_trailing_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, respecting string literals."""

    if "#" not in line:
        return line
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


__all__ = [
    "Admission",
    "admit_fence",
    "admit_strict",
    "admit_toolcall",
    "FENCE_ENVELOPE",
    "STRICT_ENVELOPE",
    "TOOLCALL_ENVELOPE",
]
