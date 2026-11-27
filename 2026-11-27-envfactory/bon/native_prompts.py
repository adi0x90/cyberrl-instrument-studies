"""Prompts for the native-pack repair arm (ATT-2975).

Hygiene is the whole point of this file.  The prompt may contain, and only contain:

* the vulnerable file's content (the agent would see it in the checkout anyway),
* the *observed vs expected* block printed by the repository's own PoC runner,
* the output envelope and the file path.

It must **not** contain `injection-spec.md`, `manifest.json`, `replay.md`, or any
`variants/` text: those describe the root cause and list the held-out triggers, which
are the answer key and Tier-2.  `tests/test_prompt_hygiene.py` asserts this.
"""

from __future__ import annotations

SYSTEM = (
    "You are a security engineer repairing a known flaw in a repository checkout. "
    "Produce a minimal fix as a unified diff. Rules:\n"
    "- Output exactly one ```diff fenced block and nothing else.\n"
    "- The diff must apply from the repository root with `git apply -p1`.\n"
    "- Do not modify tests, fixtures, or packaging files; fix the production code.\n"
    "- Keep legitimate behaviour working: do not delete the feature to remove the symptom."
)

#: Substrings that must never appear in a prompt.
FORBIDDEN_PROMPT_MARKERS = (
    "injection-spec",
    "manifest.json",
    "replay.md",
    "canary-prompt",
    "variants/",
    "aud-polyglot",
    "expired-assertion-replay",
    "variant1_user_delete_pair",
    "variant2_denylist_callable",
    "variant3_chained_policy",
)


def observed_block(poc_output: str, max_lines: int = 14) -> str:
    """Extract the PoC's observed/expected block without any variant text."""

    lines = [line.rstrip() for line in poc_output.splitlines() if line.strip()]
    keep = []
    for line in lines:
        low = line.strip().lower()
        if low.startswith(("[vulnerable]", "observed", "expected", "trigger", "outcome")):
            keep.append(line.strip())
    if not keep:
        keep = lines[-max_lines:]
    return "\n".join(keep[:max_lines])


def build_messages(product: str, tag: str, path: str, file_text: str, poc_block: str, language: str) -> list[dict]:
    body = (
        f"Repository: {product} at {tag}.\n"
        f"File to fix: {path}\n\n"
        "Vulnerable file (current content):\n"
        f"```{language}\n{file_text}\n```\n\n"
        "The repository's executable proof-of-concept reports:\n"
        f"{poc_block}\n\n"
        f"Produce a minimal unified diff against `{path}` that removes the flaw while preserving\n"
        "legitimate behaviour. The diff must apply from the repository root with `git apply -p1`.\n"
        "Output: exactly one ```diff fenced block, nothing else."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
    ]


def assert_prompt_clean(text: str) -> None:
    lowered = text.lower()
    leaked = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker.lower() in lowered]
    if leaked:
        raise ValueError(f"prompt leaks held-out material: {leaked}")


__all__ = ["FORBIDDEN_PROMPT_MARKERS", "assert_prompt_clean", "build_messages", "observed_block"]
