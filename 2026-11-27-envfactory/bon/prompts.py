"""Prompt construction for the B1 shared pool.

This is a verbatim copy of the published E1 prompt (`run_e1.py`: SYSTEM_RULES,
ENVELOPE_INSTRUCTIONS, build_messages) so the B1 pool is generated under exactly
the scaffold/prompt/decoding the earlier studies used.  Changing the prompt here
would make cross-experiment comparison invalid, so it is copied, not rewritten.

Attribution: `2026-11-20-frontier-reexperiment/run_e1.py` in the replication
package.
"""

from __future__ import annotations

import hashlib

SYSTEM_RULES = (
    "You are editing one Python expression inside a small authorization / canonicalisation "
    "task. Rules:\n"
    "- The answer must be a single Python expression that evaluates to a bool.\n"
    "- No imports, no statements, no newlines inside the expression.\n"
    "- Keep every benign case working: do not delete features or narrow the check to the "
    "known attack.\n"
    "- Follow the requested output envelope exactly and add no commentary."
)

ENVELOPE_INSTRUCTIONS = {
    "strict": (
        "Output format: exactly one JSON object and nothing else (no prose, no code fence). "
        'Shape: {"expression": "<python expression>"}.'
    ),
}


def _case_lines(cases) -> str:
    if not cases:
        return "  (none)"
    lines = []
    for case in cases:
        lines.append(
            "  - {cid}: {call} -> expect_error={err} expect={exp}".format(
                cid=case.get("id", "?"),
                call=case.get("call", ""),
                err=case.get("expect_error"),
                exp=case.get("expect"),
            )
        )
    return "\n".join(lines)


def build_messages(task: dict, envelope: str = "strict") -> list[dict]:
    body = (
        f"Task id: {task.get('id', '(no id)')}\n"
        f"Property that must keep holding: {task.get('property', '(no property text)')}\n\n"
        "Source under edit:\n"
        "```python\n"
        f"{task.get('source', '')}\n"
        "```\n\n"
        f"The editable slot currently contains the literal marker {task.get('placeholder', 'REPLACE_ME')!r}.\n"
        "Replace that slot with one Python expression.\n\n"
        "Security cases (must not be satisfied):\n"
        f"{_case_lines(task.get('security_cases'))}\n\n"
        "Benign cases (must keep working):\n"
        f"{_case_lines(task.get('benign_cases'))}\n\n"
        f"{ENVELOPE_INSTRUCTIONS[envelope]}"
    )
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": body},
    ]


def prompt_text(task: dict, envelope: str = "strict") -> str:
    """ChatML-flattened text for the local vLLM path (tokenizer-independent)."""

    messages = build_messages(task, envelope)
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


def prompt_sha(task: dict, envelope: str = "strict") -> str:
    return hashlib.sha256(prompt_text(task, envelope).encode("utf-8")).hexdigest()


__all__ = ["ENVELOPE_INSTRUCTIONS", "SYSTEM_RULES", "build_messages", "prompt_sha", "prompt_text"]
