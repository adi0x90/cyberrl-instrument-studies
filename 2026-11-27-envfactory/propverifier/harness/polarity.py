"""Conservative static classification of a guard's polarity.

``converse_guard(expression, polarity)`` answers one question: does the
expression encode the fact as the task states it (``CORRECT``), as its converse
(``CONVERSE``), or can that not be decided statically (``UNKNOWN``)?

The analysis is deliberately conservative:

1. The expression must parse (``ast.parse(mode="eval")``) and contain a
   declarative predicate -- a comparison/membership/comparison-chain, or a
   predicate call such as ``s.startswith(p)`` / ``isinstance(x, T)``.
   Anything else (arithmetic, a bare name, a bare function call) is ``UNKNOWN``.
2. Polarity is assigned by the task's own markers (``correct_marker`` /
   ``converse_marker`` from the task's ``polarity`` block), matched against the
   *code* of the expression with string literals and comments blanked out.
   Markers are matched at word boundaries where they are word-like, so ``"in"``
   matches the membership operator but not the ``in`` inside ``principal``.
   Longer markers claim their span first, so a marker like ``"not in"`` wins over
   a shorter ``"in"``.
3. If markers for both sides match, or none match, the result is ``UNKNOWN``.
   There is no majority vote and no guessing.
4. A top-level ``not <predicate>`` flips an otherwise decided result.

``explain(expression)`` returns human-readable notes about what the analysis saw;
it never raises on malformed input.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

CORRECT = "CORRECT"
CONVERSE = "CONVERSE"
UNKNOWN = "UNKNOWN"

VERDICTS = (CORRECT, CONVERSE, UNKNOWN)

#: Method calls that are declarative predicates even though they are not
#: ``ast.Compare`` nodes.
PREDICATE_METHODS: frozenset[str] = frozenset(
    {
        "startswith",
        "endswith",
        "isalnum",
        "isalpha",
        "isdigit",
        "islower",
        "isnumeric",
        "isspace",
        "istitle",
        "isupper",
    }
)

#: Function calls that are declarative predicates.
PREDICATE_FUNCTIONS: frozenset[str] = frozenset({"isinstance", "issubclass", "hasattr"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def converse_guard(expression: str, polarity: dict) -> str:
    """Return ``CORRECT``, ``CONVERSE`` or ``UNKNOWN`` for ``expression``."""

    if not isinstance(expression, str) or not expression.strip():
        return UNKNOWN
    if not isinstance(polarity, dict):
        return UNKNOWN

    tree = _parse(expression)
    if tree is None:
        return UNKNOWN
    if not _has_declarative_predicate(tree):
        return UNKNOWN

    body = tree.body
    if isinstance(body, ast.UnaryOp) and isinstance(body.op, ast.Not):
        inner_text = ast.unparse(body.operand)
        inner_tree = _parse(inner_text)
        if inner_tree is not None and _has_declarative_predicate(inner_tree):
            inner = converse_guard(inner_text, polarity)
            if inner == CORRECT:
                return CONVERSE
            if inner == CONVERSE:
                return CORRECT
        return UNKNOWN

    return _classify_by_markers(expression, polarity)


def explain(expression: str) -> list[str]:
    """Describe the predicates and negation the analysis found.  Never raises."""

    if not isinstance(expression, str) or not expression.strip():
        return ["empty expression"]

    tree = _parse(expression)
    if tree is None:
        return ["expression does not parse as a single Python expression"]

    notes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            notes.append(f"comparison: {_safe_unparse(node)}")
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in PREDICATE_METHODS:
                notes.append(f"predicate call: {_safe_unparse(node)}")
            elif isinstance(function, ast.Name) and function.id in PREDICATE_FUNCTIONS:
                notes.append(f"predicate call: {_safe_unparse(node)}")

    if isinstance(tree.body, ast.UnaryOp) and isinstance(tree.body.op, ast.Not):
        notes.append("top-level negation: the inner predicate's polarity is flipped")

    if not _has_declarative_predicate(tree):
        notes.append("no declarative comparison or membership test found")

    return list(dict.fromkeys(notes)) or ["no predicate details available"]


# ---------------------------------------------------------------------------
# Parsing / predicate detection
# ---------------------------------------------------------------------------


def _parse(expression: str) -> ast.Expression | None:
    try:
        return ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None


def _has_declarative_predicate(tree: ast.Expression) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute) and function.attr in PREDICATE_METHODS:
                return True
            if isinstance(function, ast.Name) and function.id in PREDICATE_FUNCTIONS:
                return True
    return False


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - diagnostics only
        return type(node).__name__


# ---------------------------------------------------------------------------
# Marker matching
# ---------------------------------------------------------------------------


def _marker_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _classify_by_markers(expression: str, polarity: dict) -> str:
    correct_markers = _marker_list(polarity.get("correct_marker"))
    converse_markers = _marker_list(polarity.get("converse_marker"))
    if not correct_markers and not converse_markers:
        return UNKNOWN

    masked = _code_only(expression)

    # Longest marker first, so "not in" claims its span before "in" sees it.
    candidates = sorted(
        [(marker, CORRECT) for marker in correct_markers]
        + [(marker, CONVERSE) for marker in converse_markers],
        key=lambda item: (-len(item[0]), item[1]),
    )

    claimed: list[tuple[int, int, str]] = []
    sides: set[str] = set()

    for marker, side in candidates:
        if not marker.strip():
            continue
        pattern = re.compile(_marker_pattern(marker))
        for match in pattern.finditer(masked):
            start, end = match.span()
            if start == end:
                continue
            overlaps = any(
                start < other_end and other_start < end
                for other_start, other_end, _ in claimed
            )
            if not overlaps:
                claimed.append((start, end, side))
                sides.add(side)

    if sides == {CORRECT}:
        return CORRECT
    if sides == {CONVERSE}:
        return CONVERSE
    # Mixed evidence, or no marker present: refuse to guess.
    return UNKNOWN


def _marker_pattern(marker: str) -> str:
    """Compile a marker so short words cannot match inside identifiers.

    ``"in"`` must match the membership operator, not the ``in`` inside
    ``principal``; operators (``==``, ``<=``, ``>``) are matched literally.
    Internal whitespace is flexible so ``"not in"`` matches ``not   in``.
    """

    parts = [re.escape(part) for part in marker.split()]
    pattern = r"\s+".join(parts)
    # Composite operators must not be split: ``>`` must not match inside ``>=``,
    # ``<`` must not match inside ``<=``, ``=`` must not match inside ``==``/``!=``.
    operator_chars = "<>=!+-*/%&|^~"
    if all(ch in operator_chars for ch in marker.strip()):
        pattern = r"(?<![<>=!])" + pattern + r"(?![<>=!])"
    if marker[0].isalnum() or marker[0] == "_":
        pattern = r"(?<![A-Za-z0-9_])" + pattern
    if marker[-1].isalnum() or marker[-1] == "_":
        pattern = pattern + r"(?![A-Za-z0-9_])"
    return pattern


def _code_only(expression: str) -> str:
    """Blank out string literals and comments, preserving offsets."""

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
        start = offset(token.start)
        end = offset(token.end)
        for index in range(start, min(end, len(characters))):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


__all__ = [
    "CONVERSE",
    "CORRECT",
    "PREDICATE_FUNCTIONS",
    "PREDICATE_METHODS",
    "UNKNOWN",
    "VERDICTS",
    "converse_guard",
    "explain",
]
