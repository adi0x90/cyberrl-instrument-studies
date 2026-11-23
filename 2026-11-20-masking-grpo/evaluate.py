#!/usr/bin/env python3
"""Held-out evaluation of one saved adapter (the co-primary endpoint).

Protocol: ``docs/SPEC.md`` ("Endpoints", co-primary 1-2).  The four held-out
tasks are the lexicographically **last** four of the 12-task corpus; the adapter
is the final update of one arm (no checkpoint selection).  Each task is decoded
``--decodings`` independent times at the training decoding (temperature 1.0,
top_p 0.95, max 48 new tokens); Amendment v2 fixes that count at **16 per task**,
so the endpoint has 4 x 16 = 64 draws, each scored exactly as in training:
``admit_strict`` -> ``execute_patch`` + ``grade_obligation`` -> reward 1.0 iff
HOLDS.  The draw count is never hardcoded: it is the ``--decodings`` flag, and it
is recorded in the eval JSON as ``decodings``.

Reported: held-out joint success (HOLDS / draws) with a Wilson 95% interval,
held-out admission rate with a Wilson 95% interval, joint success conditional on
admission, and the per-task breakdown.  The intervals are computed here (standard
library) rather than by a third-party package, so the file can be rebuilt on a
bare interpreter.

``--dry-run`` evaluates the deterministic canned model instead of an adapter
(no torch, no GPU, no network).  Dry-run numbers are not evidence.

Usage:
    python evaluate.py --arm control --adapter results/adapter_control
    python evaluate.py --arm masked --adapter results/adapter_masked \
        --out results/eval_masked.json
    python evaluate.py --arm control --dry-run --out results/dryrun/eval_control.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import grpo
import model_utils
import rollout

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DRYRUN_SUBDIR = "dryrun"

#: ``docs/SPEC.md`` Amendment v2: 16 decodings x 4 held-out tasks = 64 draws per
#: adapter.  Overridable with ``--decodings``; the effective count is recorded in
#: the eval JSON.
DECODINGS = 16
MAX_MINUTES = 30.0
#: Evaluation draws use group indices far above any training group, so the
#: per-group sampling seed can never collide with a training draw.
HELD_OUT_INDEX_BASE = 100_000


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base_dir(dry_run: bool) -> Path:
    return DEFAULT_RESULTS / DRYRUN_SUBDIR if dry_run else DEFAULT_RESULTS


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    ``(0.0, 0.0)`` for ``n == 0``: there is no interval, and callers render that
    as ``n/a`` rather than inventing a number.
    """

    if n <= 0:
        return (0.0, 0.0)
    successes = max(0, min(int(successes), int(n)))
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _default_out(arm: str, dry_run: bool) -> Path:
    return _base_dir(dry_run) / f"eval_{arm}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one saved LoRA adapter on the 4 held-out tasks (docs/SPEC.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm", required=True, choices=list(grpo.ARMS), help="which arm's adapter this is")
    parser.add_argument("--adapter", default=None, help="path to the saved LoRA adapter directory")
    parser.add_argument("--out", default=None, help="eval JSON path")
    parser.add_argument("--decodings", type=int, default=DECODINGS, help="independent decodings per task")
    parser.add_argument("--seed", type=int, default=20261120, help="sampling seed")
    parser.add_argument("--model", default=model_utils.MODEL_ID, help="base policy model id")
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "pinned model revision/commit; falls back to $"
            f"{model_utils.REVISION_ENV}. The resolved commit is recorded either way."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=model_utils.DTYPE,
        choices=["auto", "bfloat16", "float16", "float32"],
        help="model dtype",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate the canned deterministic model (no adapter, no GPU)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help=(
            "forbid the Hugging Face Hub: the model and adapter must already be cached. "
            "The harness makes no network call at all when this is set."
        ),
    )
    parser.add_argument(
        "--tasks-dir",
        default=str(rollout.DEFAULT_TASKS_DIR),
        help="directory holding the 12 corpus task files",
    )
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES, help="wall-clock budget")
    return parser


def build_backend(args: argparse.Namespace, tasks):
    if args.dry_run:
        return model_utils.DryRunModel(
            rollout.dry_run_pools(tasks), seed=args.seed, arm=args.arm, model_id=args.model
        )

    if not args.adapter:
        raise model_utils.ModelError(
            "--adapter is required for a real evaluation (pass --dry-run for the offline path)"
        )
    adapter = Path(args.adapter)
    if not adapter.is_dir():
        raise model_utils.ModelError(f"adapter directory not found: {adapter}")
    revision, pinned = model_utils.requested_revision(args.model_revision)
    if not pinned:
        print(
            "warning: no model revision pinned; pass --model-revision <commit> or set "
            f"${model_utils.REVISION_ENV}. The resolved commit is recorded either way.",
            file=sys.stderr,
        )
    return model_utils.load_for_evaluation(
        args.model,
        revision if pinned else None,
        adapter_path=str(adapter),
        dtype=args.dtype,
        local_files_only=bool(args.local_files_only),
    )


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    started_utc = utc_now()

    _, held_out_tasks = rollout.split_train_heldout(args.tasks_dir)
    out_path = Path(args.out) if args.out else _default_out(args.arm, args.dry_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args, held_out_tasks)
    cache = rollout.GraderCache()

    records: list[dict] = []
    per_task: list[dict] = []
    stopped_early = False

    for task in held_out_tasks:
        task_records: list[dict] = []
        prompt = rollout.build_prompt(task)
        for decoding in range(max(1, int(args.decodings))):
            if (time.monotonic() - started) / 60.0 > args.max_minutes:
                stopped_early = True
                break
            seed = rollout.group_seed(args.seed, str(task["id"]), HELD_OUT_INDEX_BASE + decoding)
            generations = backend.generate(prompt, 1, seed)
            if len(generations) != 1:
                raise RuntimeError(f"generate returned {len(generations)} completions, expected 1")
            generation = generations[0]
            score = rollout.score_completion(task, generation.text, cache, arm=args.arm)
            record = {
                "task_id": str(task["id"]),
                "decoding": decoding,
                "seed": seed,
                "completion": generation.text,
                "admitted": score.admitted,
                "admission_reason": score.admission_reason,
                "expression": score.expression,
                "verdict": score.verdict,
                "reasons": list(score.reasons),
                "reward": score.reward,
                "prompt_tokens": int(generation.prompt_tokens),
                "completion_tokens": int(generation.completion_tokens),
            }
            records.append(record)
            task_records.append(record)
        if stopped_early:
            break

        admitted = sum(1 for record in task_records if record["admitted"])
        holds = sum(1 for record in task_records if record["verdict"] == rollout.HOLDS)
        per_task.append(
            {
                "task_id": str(task["id"]),
                "draws": len(task_records),
                "admitted": admitted,
                "holds": holds,
                "admission_rate": (admitted / len(task_records)) if task_records else None,
                "joint_success": (holds / len(task_records)) if task_records else None,
                "joint_success_given_admitted": (holds / admitted) if admitted else None,
            }
        )

    draws = len(records)
    admitted_total = sum(1 for record in records if record["admitted"])
    holds_total = sum(1 for record in records if record["verdict"] == rollout.HOLDS)

    joint_low, joint_high = wilson_ci(holds_total, draws)
    admission_low, admission_high = wilson_ci(admitted_total, draws)

    wall_time_s = time.monotonic() - started
    payload = {
        "experiment": "2026-11-20-masking-grpo",
        "arm": args.arm,
        "dry_run": bool(args.dry_run),
        "status": "incomplete" if stopped_early else "complete",
        "git_sha": rollout.git_sha(ROOT),
        "spec_sha256": rollout.spec_sha256(ROOT / "docs" / "SPEC.md"),
        "utc_start": started_utc,
        "utc_end": utc_now(),
        "wall_time_s": round(wall_time_s, 3),
        "argv": sys.argv[1:],
        "seed": int(args.seed),
        "model": {
            **backend.description(),
            "adapter_path": None if args.dry_run else str(args.adapter),
        },
        "decodings": int(args.decodings),
        "decoding": {
            "temperature": rollout.TEMPERATURE,
            "top_p": rollout.TOP_P,
            "max_new_tokens": rollout.MAX_NEW_TOKENS,
            "decodings_per_task": int(args.decodings),
        },
        "held_out_tasks": [str(task["id"]) for task in held_out_tasks],
        "token_accounting": "estimated" if args.dry_run else "tokenizer",
        "counts": {
            "draws": draws,
            "admitted": admitted_total,
            "holds": holds_total,
        },
        "endpoints": {
            "joint_success": (holds_total / draws) if draws else None,
            "joint_success_wilson95": [joint_low, joint_high],
            "admission_rate": (admitted_total / draws) if draws else None,
            "admission_rate_wilson95": [admission_low, admission_high],
            "joint_success_given_admitted": (holds_total / admitted_total) if admitted_total else None,
            "draws": draws,
        },
        "per_task": per_task,
        "records": records,
    }

    temporary = out_path.with_name(out_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(out_path)

    print(
        f"[eval] arm={args.arm} draws={draws} admitted={admitted_total} holds={holds_total} "
        f"joint={100.0 * holds_total / draws if draws else 0.0:.1f}% "
        f"(95% {100.0 * joint_low:.1f}-{100.0 * joint_high:.1f}%) "
        f"admission={100.0 * admitted_total / draws if draws else 0.0:.1f}% "
        f"wall={wall_time_s:.1f}s",
        file=sys.stderr,
    )
    print(f"[eval] wrote {out_path}", file=sys.stderr)
    return 0 if not stopped_early else 3


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.decodings < 1:
        parser.error("--decodings must be >= 1")
    if args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    try:
        return run(args)
    except rollout.CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except model_utils.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[eval] interrupted", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
