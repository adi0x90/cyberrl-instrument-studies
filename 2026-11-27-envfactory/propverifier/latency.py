"""A1 latency arm — paired per-candidate cost, measured not modelled.

Reports, per task and over ``n`` calls:

* ``static_ms``   — run the task's fixed cases + ``grade_obligation``
* ``property_ms`` — run the generated case set + metamorphic pairs + ``grade_property_tier1``
* ``delta_ms``    — the paired difference, i.e. what the property tier costs per candidate

The delta is **per call**.  It is not multiplied by cases-per-episode: each call
already evaluates the whole generated case set, and the preregistration forbids
that double count.

Oracle construction is a per-task artefact, not a per-candidate cost, so it is
timed and reported separately (``oracle_build_ms``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .harness.tasks_io import execute_patch, load_task
from .run_a1 import BASE_SEED, latency_pair, task_ids

TASK_ROOT = Path(__file__).resolve().parent


def _one(task_id: str, n: int) -> dict:
    task = load_task(task_id, TASK_ROOT)
    return latency_pair(task, BASE_SEED, n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--out", default="evidence/a1/a1_latency.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", default=None, help="debug: run one task id")
    args = parser.parse_args()

    ids = [args.only] if args.only else task_ids()
    started = time.time()
    results: list[dict] = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_one, task_id, args.n): task_id for task_id in ids}
            for future in futures:
                task_id = futures[future]
                result = future.result()
                results.append(result)
                print(f"[latency] {task_id}: delta_p50={result['delta_ms']['p50']} ms", flush=True)
    else:
        for task_id in ids:
            result = _one(task_id, args.n)
            results.append(result)
            print(f"[latency] {task_id}: delta_p50={result['delta_ms']['p50']} ms", flush=True)

    results.sort(key=lambda row: row["task_id"])
    deltas = [row["delta_ms"]["p50"] for row in results]
    payload = {
        "n_per_task": args.n,
        "tasks": results,
        "aggregate": {
            "delta_p50_ms_mean_across_tasks": round(statistics.fmean(deltas), 4),
            "delta_p50_ms_max": max(deltas),
            "wall_seconds": round(time.time() - started, 1),
            "note": "per-candidate delta; already includes the full generated case set per call",
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[latency] wrote {args.out} ({payload['aggregate']['wall_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
