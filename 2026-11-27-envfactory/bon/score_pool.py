"""B1 — score every completion in the shared pool under all three rewards.

One row per completion:

* ``admitted`` / ``expression`` — the strict JSON envelope result (P(admitted) plumbing)
* ``weak``  — ``grade_weak`` (PoC/security-only reward)
* ``t1``    — ``grade_obligation`` (static obligation reward)
* ``t2``    — property + metamorphic only (independent Tier-2)

Selection never happens here; this file only produces verdicts on the whole pool so
that ``select_drift.py`` can rescore the *same* pool at every N.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bon.tier2 import oracle_for, tier2_expression
from propverifier.harness.envelopes import admit_strict
from propverifier.harness.graders import grade_obligation, grade_weak
from propverifier.harness.tasks_io import execute_patch, load_task

TASK_ROOT = Path(__file__).resolve().parent.parent / "propverifier"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args()

    tasks = {path.stem: load_task(path.stem, TASK_ROOT) for path in (TASK_ROOT / "tasks").glob("*.json")}
    oracles = {task_id: oracle_for(task, args.seed) for task_id, task in tasks.items()}

    rows = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    written = 0
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in rows:
            task = tasks[row["task_id"]]
            admission = admit_strict(row["raw_reply"])
            expression = admission.expression if admission.ok else None
            record = {
                "task_id": row["task_id"],
                "model": row["model"],
                "sample": row["sample"],
                "admitted": bool(admission.ok),
                "admission_reason": admission.reason,
                "expression": expression,
                "weak": "UNKNOWN",
                "t1": "UNKNOWN",
                "t2": "UNKNOWN",
            }
            if expression:
                result = execute_patch(task, expression)
                record["weak"] = grade_weak(task, result)
                record["t1"], record["t1_reasons"] = grade_obligation(task, result, expression)
                record["t2"], record["t2_reasons"] = tier2_expression(task, expression, oracles[row["task_id"]])
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
            if written % 64 == 0:
                print(f"[score] {written}/{len(rows)}", flush=True, file=sys.stderr)

    print(f"[score] wrote {args.out} ({written} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
