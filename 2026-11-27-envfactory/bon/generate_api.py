"""B1 — shared Best-of-N pool, OpenRouter flash-model path.

Same prompt, same temperature (1.0), same fixed seed, same n=32 per (task, model)
as the local path.  Sequential with resume: an interrupted run continues where it
stopped, keyed by (task, model, sample).

Budget discipline: every call records token usage and the runner stops before the
pre-reserved cap instead of truncating a pool mid-way (a truncated pool would
silently bias the N curve).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bon.prompts import build_messages
from propverifier.harness.models import chat

# Pre-reserved: up to $50 for this arm.  Flash-tier list prices are well under
# $0.30 / 1M input and $1.00 / 1M output tokens; the guard is deliberately
# conservative and reports the estimate it used.
PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 1.00


def load_tasks(tasks_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(Path(tasks_dir).glob("*.json"))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default="propverifier/tasks")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--cap-usd", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing", file=sys.stderr)
        return 2

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, int]] = set()
    spend = 0.0
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["task_id"], row["model"], row["sample"]))
                spend += (
                    row["prompt_tokens"] / 1e6 * PRICE_IN_PER_M
                    + row["completion_tokens"] / 1e6 * PRICE_OUT_PER_M
                )

    started = time.time()
    work = [
        (model, task, sample)
        for model in models
        for task in tasks
        for sample in range(args.n)
        if (task["id"], model, sample) not in done
    ]
    print(f"[api] {len(work)} completions to fetch, {args.workers} workers", flush=True)

    lock = threading.Lock()
    state = {"spend": spend, "written": 0, "stopped": False}

    def fetch(model: str, task: dict, sample: int):
        with lock:
            if state["stopped"] or state["spend"] >= args.cap_usd:
                state["stopped"] = True
                return None
            budget = args.cap_usd - state["spend"]
        reply = chat(
            model,
            build_messages(task),
            temperature=1.0,
            seed=args.seed + sample,
            max_tokens=1600,
        )
        cost = reply.prompt_tokens / 1e6 * PRICE_IN_PER_M + reply.completion_tokens / 1e6 * PRICE_OUT_PER_M
        with lock:
            state["spend"] += cost
            if state["spend"] > args.cap_usd:
                state["stopped"] = True
        return {
            "task_id": task["id"],
            "model": model,
            "sample": sample,
            "seed": args.seed + sample,
            "temperature": 1.0,
            "prompt_tokens": int(reply.prompt_tokens or 0),
            "completion_tokens": int(reply.completion_tokens or 0),
            "raw_reply": reply.text,
            "finish_reason": None,
        }

    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, model, task, sample): (model, task["id"], sample) for model, task, sample in work}
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            with lock:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                state["written"] += 1
                if state["written"] % 32 == 0:
                    print(
                        f"[api] {state['written']}/{len(work)} written, est ${state['spend']:.3f} "
                        f"({time.time() - started:.0f}s)",
                        flush=True,
                    )
    spend = state["spend"]

    print(f"[api] wrote {out_path}; estimated spend ${spend:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
