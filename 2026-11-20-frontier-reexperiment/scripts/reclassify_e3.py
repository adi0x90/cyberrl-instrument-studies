#!/usr/bin/env python3
"""Re-run the E3 polarity classifier over saved expressions and rewrite the verdicts.

Why this exists
---------------
E3's endpoint is `harness.polarity.converse_guard(raw_expression, task["polarity"])`.
The classifier is lexical, so a classifier bug silently changes an endpoint. The first
version matched the converse marker ``>`` inside the operator ``>=`` and produced four
false ``CONVERSE`` labels. The fix made operators boundary-aware; the four affected
records had to be re-labelled offline, from the expressions already saved in the JSONL,
without spending another model call.

This script is that offline pass, kept as code so the correction is reproducible and so
a future classifier change can be applied to the same evidence the same way. It never
calls a model and never invents data: it recomputes a verdict from
(`raw_expression`, task polarity block) and, only when the recomputed verdict differs
from the recorded one, rewrites it and records the old value.

Usage:
    python scripts/reclassify_e3.py --dry-run      # report, change nothing (safe default to read)
    python scripts/reclassify_e3.py                # rewrite evidence/e3.jsonl in place

Fields
------
* ``raw_expression`` is read, never modified.
* ``verdict`` is replaced only when the recomputation disagrees with it.
* ``verdict_reclassifier_before`` is written at the first change and then **left alone**,
  so repeated passes cannot overwrite the runner's original verdict with an intermediate
  one. A record that never changed never grows the field.
* Everything else (``reasons``, ``raw_reply``, tokens, timestamps) is untouched; ``reasons``
  is the classifier's audit note from admission time, not a verdict.

Idempotence: after the correction has been applied, a second run reports 0 changes and
leaves the file byte-for-byte untouched (no rewrite at all when nothing changes).

Exit codes: 0 ok (changes or not), 2 unusable input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

#: Import the real classifier from the experiment root: the whole point is to run the
#: shipped code over saved data, not a copy of its logic. `scripts/` is not on
#: sys.path[0] when the file is run as `python scripts/reclassify_e3.py`, so add ROOT.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.polarity import converse_guard  # noqa: E402  (needs ROOT on sys.path)

BEFORE_KEY = "verdict_reclassifier_before"


def load_tasks(tasks_dir: Path) -> dict[str, dict]:
    """Map task id (file stem) -> task dict for every ``tasks/*.json``."""

    if not tasks_dir.is_dir():
        raise SystemExit(f"error: tasks directory does not exist: {tasks_dir}")
    tasks: dict[str, dict] = {}
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(f"error: {path} is not valid JSON ({error.msg})") from None
        if not isinstance(task, dict):
            raise SystemExit(f"error: {path} is not a JSON object")
        tasks[path.stem] = task
    return tasks


def read_records(evidence: Path) -> list[dict]:
    """Read the JSONL evidence; every non-blank line must be a JSON object."""

    if not evidence.is_file():
        raise SystemExit(f"error: evidence file does not exist: {evidence}")
    records: list[dict] = []
    with evidence.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"error: {evidence} line {number} is not JSON ({error.msg})"
                ) from None
            if not isinstance(record, dict):
                raise SystemExit(f"error: {evidence} line {number} is not a JSON object")
            records.append(record)
    return records


def reclassify(records: list[dict], tasks: dict[str, dict]) -> list[dict]:
    """Return the change records; ``records`` is mutated in place."""

    changes: list[dict] = []
    for record in records:
        expression = record.get("raw_expression")
        task = tasks.get(record.get("task_id"))
        if not isinstance(expression, str) or not expression.strip():
            record["_skip"] = "no raw_expression"  # internal tally, stripped before output
            continue
        if task is None:
            record["_skip"] = "no matching task"
            continue
        recomputed = converse_guard(expression, task.get("polarity", {}))
        previous = record.get("verdict")
        if recomputed == previous:
            continue
        changes.append(
            {
                "unit": record.get("unit"),
                "task_id": record.get("task_id"),
                "model": record.get("model"),
                "variant": record.get("variant"),
                "decoding": record.get("decoding"),
                "before": previous,
                "after": recomputed,
                "expression": expression,
            }
        )
        record.setdefault(BEFORE_KEY, previous)
        record["verdict"] = recomputed
    return changes


def write_records(evidence: Path, records: list[dict]) -> None:
    """Rewrite ``evidence`` atomically in the runners' canonical JSONL format."""

    temporary = evidence.parent / (evidence.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                clean = {k: v for k, v in record.items() if k != "_skip"}
                handle.write(json.dumps(clean, sort_keys=True) + "\n")
        os.replace(temporary, evidence)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute E3 verdicts with harness.polarity.converse_guard.",
    )
    parser.add_argument(
        "--evidence", default="evidence/e3.jsonl", help="E3 JSONL (default: evidence/e3.jsonl)"
    )
    parser.add_argument("--tasks", default="tasks", help="task corpus directory (default: tasks)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change; write nothing",
    )
    args = parser.parse_args(argv)

    evidence = Path(args.evidence)
    tasks = load_tasks(Path(args.tasks))
    records = read_records(evidence)

    before = Counter(str(record.get("verdict")) for record in records)
    changes = reclassify(records, tasks)
    after = Counter(str(record.get("verdict")) for record in records)

    considered = sum(1 for record in records if "_skip" not in record)
    skipped = Counter(
        record["_skip"] for record in records if record.get("_skip")  # noqa: SIM118
    )

    print(f"evidence: {evidence} ({len(records)} records)")
    print(f"tasks: {len(tasks)} task files")
    print(f"recomputed: {considered} record(s) with a raw_expression and a matching task")
    for reason, count in sorted(skipped.items()):
        print(f"skipped: {count} record(s) — {reason}")
    print("verdict before: " + _counts(before))
    print("verdict after : " + _counts(after))
    print(f"changed: {len(changes)}")

    for change in changes:
        print(
            f"  {change['unit']}: {change['before']} -> {change['after']}"
            f"  [{_short(change['expression'])}]"
        )

    if not changes:
        print("no change — file left untouched (idempotent)")
        return 0

    if args.dry_run:
        print(f"dry run: {evidence} not modified")
        return 0

    write_records(evidence, records)
    print(f"rewrote {evidence} ({len(changes)} verdict(s) updated)")
    return 0


def _counts(counter: Counter) -> str:
    if not counter:
        return "(none)"
    return ", ".join(f"{name} {count}" for name, count in sorted(counter.items()))


def _short(text: str, width: int = 72) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
