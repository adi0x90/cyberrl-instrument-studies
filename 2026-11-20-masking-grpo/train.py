#!/usr/bin/env python3
"""One arm of the preregistered masking experiment.

Protocol: ``docs/SPEC.md`` (including "Amendment v2").  This script runs **one
arm** (``control``, ``masked`` or ``penalty``) and nothing else, so the arms
cannot drift apart in model, prompts, decoding or optimizer.  The arm decides two
things and only two things:

* the reward a rejected completion carries (``grpo.invalid_reward``: ``0.0`` for
  control/masked, ``-1.0`` for penalty), applied in ``rollout.score_completion``;
* the baseline population (``grpo.group_advantage``: all G for control and
  penalty, admitted samples only for masked).

Everything else below is shared:

* tasks: the 12-task corpus of record, split lexicographically into the 8 train
  tasks and the 4 held-out tasks;
* budget: ``--effective-updates`` **effective** gradient steps (a group whose arm
  treatment yields a nonzero-variance baseline), hard-capped at
  ``--max-groups`` groups sampled.  An arm stops at whichever comes first and the
  shortfall is recorded.  Rollouts consumed are a reported cost, not a hidden
  confound;
* per group: sample ``--g`` completions at temperature 1.0 / top_p 0.95 / 48 new
  tokens, admit with ``admit_strict``, grade with ``grade_obligation``, reward
  1.0 iff HOLDS, compute the arm's advantages with :mod:`grpo`, and take **one**
  optimizer step on ``-mean(advantage * sum_logprob)`` over the arm's treated
  samples.  A zero-variance group takes no step at all (a step with a zero
  gradient would still move the weights under AdamW's weight decay).

Outputs (paths overridable):

``results/rollouts_<arm>.jsonl``
    One record per attempted completion, failures included.
``results/metrics_<arm>.json``
    The manifest (model, revision, GPU, git SHA, argv, task hashes), the stop
    reason, the counts, the endpoint aggregates, and one metrics record per group.
``results/adapter_<arm>/``
    The saved LoRA adapter, final update only (no checkpoint selection).

``--dry-run`` swaps the policy for ``model_utils.DryRunModel``: canned
completions derived from the task files, no torch, no GPU, no network.  The dry
run exercises every branch of the loop (admission, HOLDS, zero-variance groups,
skipped groups) and writes to ``results/dryrun/`` by default.  Dry-run artefacts
are **not evidence**.

Usage:
    python train.py --arm control
    python train.py --arm masked --effective-updates 8 --max-groups 40
    python train.py --arm penalty --dry-run --effective-updates 2 --max-groups 6
"""

from __future__ import annotations

import argparse
import json
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

#: Fixed by the protocol; not tunable from the CLI.
EFFECTIVE_UPDATES = 24
MAX_GROUPS = 96
PROMPTS_PER_UPDATE = 4
SEED = 20261120
MAX_MINUTES = 90.0
GPU_BUDGET_CAP_USD = 3.00

#: Stopping rules ("Abort an arm ... if > 10 consecutive update exceptions, OOM twice").
MAX_CONSECUTIVE_EXCEPTIONS = 10
MAX_OOM = 2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _task_sha256(tasks) -> dict[str, str]:
    return {str(task["id"]): rollout.sha256_text(json.dumps(task, sort_keys=True)) for task in tasks}


def _base_dir(dry_run: bool) -> Path:
    return DEFAULT_RESULTS / DRYRUN_SUBDIR if dry_run else DEFAULT_RESULTS


def _default_out(arm: str, dry_run: bool) -> Path:
    return _base_dir(dry_run) / f"rollouts_{arm}.jsonl"


def _default_metrics(arm: str, dry_run: bool) -> Path:
    return _base_dir(dry_run) / f"metrics_{arm}.json"


def _default_adapter(arm: str) -> Path:
    return DEFAULT_RESULTS / f"adapter_{arm}"


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "outofmemory" in name or "out of memory" in text or "cuda oom" in text


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verdict_counts(records) -> dict[str, int]:
    counts = {"HOLDS": 0, "VIOLATED": 0, "UNKNOWN": 0}
    for record in records:
        verdict = str(record.get("verdict"))
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one arm of the invalid-action masking experiment (docs/SPEC.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm", required=True, choices=list(grpo.ARMS), help="which treatment to run")
    parser.add_argument(
        "--effective-updates",
        type=int,
        default=EFFECTIVE_UPDATES,
        help="stop after this many effective (nonzero-variance) gradient steps",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=MAX_GROUPS,
        help="hard cap on groups sampled; the arm stops here if it has not reached the target",
    )
    parser.add_argument(
        "--prompts-per-update",
        type=int,
        default=PROMPTS_PER_UPDATE,
        help="groups sampled per round (one optimizer step per group)",
    )
    parser.add_argument("--g", type=int, default=rollout.G, help="completions per prompt")
    parser.add_argument("--out", default=None, help="rollout JSONL path")
    parser.add_argument("--metrics", default=None, help="metrics JSON path")
    parser.add_argument(
        "--adapter-out",
        default=None,
        help="directory for the saved LoRA adapter (ignored by --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="canned completions, no model, no GPU; writes under results/dryrun/",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="sampling seed")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=MAX_MINUTES,
        help="wall-clock budget; the arm aborts and reports when it is exceeded",
    )
    parser.add_argument("--model", default=model_utils.MODEL_ID, help="policy model id")
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
        "--local-files-only",
        action="store_true",
        help=(
            "forbid the Hugging Face Hub: the model must already be cached. The harness "
            "makes no network call at all when this is set."
        ),
    )
    parser.add_argument(
        "--tasks-dir",
        default=str(rollout.DEFAULT_TASKS_DIR),
        help="directory holding the 12 corpus task files",
    )
    parser.add_argument(
        "--hourly-rate",
        type=float,
        default=0.0,
        help="pod hourly rate in USD, for the spend estimate (never read from a file)",
    )
    return parser


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def build_backend(args: argparse.Namespace, tasks):
    """Real policy, or the deterministic canned model for ``--dry-run``."""

    if args.dry_run:
        return model_utils.DryRunModel(
            rollout.dry_run_pools(tasks),
            seed=args.seed,
            arm=args.arm,
            model_id=args.model,
        )
    revision, pinned = model_utils.requested_revision(args.model_revision)
    if not pinned:
        print(
            "warning: no model revision pinned. Pass --model-revision <commit> or set "
            f"${model_utils.REVISION_ENV}. The resolved commit will still be recorded, but the "
            "manifest will say pinned=false.",
            file=sys.stderr,
        )
    return model_utils.TorchPolicy.load(
        args.model,
        revision if pinned else None,
        dtype=args.dtype,
        local_files_only=bool(args.local_files_only),
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    started_utc = utc_now()

    train_tasks, held_out_tasks = rollout.split_train_heldout(args.tasks_dir)
    all_tasks = train_tasks + held_out_tasks

    out_path = Path(args.out) if args.out else _default_out(args.arm, args.dry_run)
    metrics_path = Path(args.metrics) if args.metrics else _default_metrics(args.arm, args.dry_run)
    adapter_path = Path(args.adapter_out) if args.adapter_out else _default_adapter(args.arm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    cache = rollout.GraderCache()
    backend = build_backend(args, all_tasks)
    description = backend.description()

    if args.max_groups < args.effective_updates:
        print(
            f"warning: --max-groups ({args.max_groups}) < --effective-updates "
            f"({args.effective_updates}); the arm will hit the group cap first and report a "
            "shortfall.",
            file=sys.stderr,
        )

    manifest = {
        "experiment": "2026-11-20-masking-grpo",
        "arm": args.arm,
        "arms": list(grpo.ARMS),
        "arm_treatment": {
            "invalid_reward": grpo.invalid_reward(args.arm),
            "baseline_population": (
                "all-G" if args.arm in grpo.ALL_SAMPLE_ARMS else "admitted-only"
            ),
            "skip_if_fewer_than_two_admitted": bool(args.arm == "masked"),
        },
        "dry_run": bool(args.dry_run),
        "git_sha": rollout.git_sha(ROOT),
        "spec_sha256": rollout.spec_sha256(ROOT / "docs" / "SPEC.md"),
        "utc_start": started_utc,
        "argv": sys.argv[1:],
        "seed": int(args.seed),
        "model": description,
        "decoding": {
            "temperature": rollout.TEMPERATURE,
            "top_p": rollout.TOP_P,
            "max_new_tokens": rollout.MAX_NEW_TOKENS,
            "g": int(args.g),
        },
        "budget": {
            "target_effective_updates": int(args.effective_updates),
            "max_groups": int(args.max_groups),
            "prompts_per_update": int(args.prompts_per_update),
            "max_minutes": float(args.max_minutes),
            "gpu_budget_cap_usd": GPU_BUDGET_CAP_USD,
            "hourly_rate_usd": float(args.hourly_rate),
        },
        "tasks": {
            "train": [str(task["id"]) for task in train_tasks],
            "held_out": [str(task["id"]) for task in held_out_tasks],
            "sha256": _task_sha256(all_tasks),
        },
        "token_accounting": "estimated" if args.dry_run else "tokenizer",
        "status": "running",
        "records_written": 0,
        "groups": [],
    }
    _write_json(metrics_path, manifest)

    groups: list[dict] = []
    groups_sampled = 0
    effective_updates = 0
    groups_skipped = 0
    groups_zero_variance = 0
    groups_errored = 0
    rollouts_consumed = 0
    records_written = 0
    admitted_total = 0
    holds_total = 0
    admitted_advantage_total = 0.0
    admitted_advantage_count = 0
    variance_total = 0.0

    consecutive_exceptions = 0
    oom_count = 0
    stop_reason: Optional[str] = None
    abort_reason: Optional[str] = None
    status = "complete"

    handle = out_path.open("w", encoding="utf-8")
    try:
        while stop_reason is None:
            for _ in range(max(1, int(args.prompts_per_update))):
                if effective_updates >= args.effective_updates:
                    stop_reason = "effective-target"
                    break
                if groups_sampled >= args.max_groups:
                    stop_reason = "group-cap"
                    break
                if (time.monotonic() - started) / 60.0 > args.max_minutes:
                    stop_reason = "wall-time"
                    status = "aborted"
                    abort_reason = "wall-time-exceeded"
                    break

                group_index = groups_sampled
                task = train_tasks[group_index % len(train_tasks)]
                group_started = time.monotonic()

                try:
                    group = rollout.run_group(
                        task,
                        backend.generate,
                        group_index=group_index,
                        g=args.g,
                        seed=args.seed,
                        arm=args.arm,
                        cache=cache,
                    )
                    # One group is one update.  The arm treatment decides the
                    # baseline population and which samples carry gradient.  The
                    # step is inside this guard on purpose: an OOM during a
                    # backward pass must feed the same abort accounting as any
                    # other group failure, not crash the arm.
                    advantages, effective = grpo.group_advantage(
                        group.rewards, group.valid_mask, args.arm
                    )
                    stats = grpo.group_stats(group.rewards, group.valid_mask, args.arm)
                    treated = grpo.treated_mask(group.valid_mask, args.arm)

                    loss = 0.0
                    sum_logprobs: list[Optional[float]] = [None] * len(group.completions)
                    gpu_mem = backend.gpu_mem_mb()
                    if effective:
                        update = backend.update(
                            group.prompt, group.completions, advantages, treated
                        )
                        loss = float(update.loss)
                        sum_logprobs = list(update.sum_logprobs)
                        gpu_mem = update.gpu_mem_mb
                except Exception as exc:  # noqa: BLE001 - one bad group must not kill the arm
                    groups_sampled += 1
                    groups_errored += 1
                    consecutive_exceptions += 1
                    if _is_oom(exc):
                        oom_count += 1
                    groups.append(
                        {
                            "group_index": group_index,
                            "group_id": f"g{group_index:04d}",
                            "task_id": str(task["id"]),
                            "effective": False,
                            "skipped": False,
                            "zero_variance": True,
                            "mean_reward": None,
                            "admission_rate": None,
                            "group_std": None,
                            "group_variance": None,
                            "valid_count": 0,
                            "mean_abs_advantage_admitted": None,
                            "loss": 0.0,
                            "optimizer_step": False,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "wall_time_s": round(time.monotonic() - group_started, 3),
                            "gpu_mem_mb": backend.gpu_mem_mb(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(
                        f"[train] group {group_index} failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if consecutive_exceptions > MAX_CONSECUTIVE_EXCEPTIONS:
                        status, abort_reason, stop_reason = (
                            "aborted",
                            "consecutive-exceptions",
                            "exception-abort",
                        )
                        break
                    if oom_count >= MAX_OOM:
                        status, abort_reason, stop_reason = "aborted", "oom", "oom-abort"
                        break
                    continue

                # A group that ran without raising breaks any exception streak.
                consecutive_exceptions = 0
                if effective:
                    effective_updates += 1

                for record, logprob in zip(group.records, sum_logprobs):
                    record["sum_logprob"] = logprob
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    records_written += 1
                handle.flush()

                skipped = args.arm == "masked" and stats.n_valid < 2
                if skipped:
                    groups_skipped += 1
                if not effective:
                    groups_zero_variance += 1

                verdicts = _verdict_counts(group.records)
                admitted = sum(1 for flag in group.valid_mask if flag)
                holds = sum(1 for score in group.scores if score.verdict == rollout.HOLDS)
                admitted_total += admitted
                holds_total += holds
                if stats.mean_abs_advantage_admitted is not None and stats.n_valid:
                    admitted_advantage_total += stats.mean_abs_advantage_admitted * stats.n_valid
                    admitted_advantage_count += stats.n_valid
                variance_total += stats.variance

                groups_sampled += 1
                rollouts_consumed += len(group.completions)
                groups.append(
                    {
                        "group_index": group_index,
                        "group_id": group.group_id,
                        "task_id": group.task_id,
                        "effective": bool(effective),
                        "skipped": bool(skipped),
                        "zero_variance": bool(not effective),
                        "mean_reward": round(sum(group.rewards) / len(group.rewards), 6),
                        "admission_rate": round(admitted / len(group.rewards), 6),
                        "valid_count": int(stats.n_valid),
                        "group_std": round(stats.std, 8),
                        "group_variance": round(stats.variance, 8),
                        "mean_abs_advantage_admitted": (
                            None
                            if stats.mean_abs_advantage_admitted is None
                            else round(stats.mean_abs_advantage_admitted, 8)
                        ),
                        "loss": round(loss, 8),
                        "optimizer_step": bool(effective),
                        "prompt_tokens": int(group.prompt_tokens),
                        "completion_tokens": int(group.completion_tokens),
                        "verdicts": verdicts,
                        "wall_time_s": round(time.monotonic() - group_started, 3),
                        "gpu_mem_mb": gpu_mem,
                        "error": None,
                    }
                )
                print(
                    f"[train] {args.arm} group {group_index:3d} "
                    f"task={group.task_id:<34s} effective={str(effective):5s} "
                    f"admitted={admitted}/{len(group.rewards)} "
                    f"holds={holds} loss={loss:+.4f} "
                    f"eff_updates={effective_updates}/{args.effective_updates}",
                    file=sys.stderr,
                )
    finally:
        handle.close()

    wall_time_s = time.monotonic() - started

    adapter_dir: Optional[str] = None
    if not args.dry_run:
        try:
            backend.save_adapter(adapter_path)
            adapter_dir = str(adapter_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[train] warning: could not save the adapter: {exc}", file=sys.stderr)

    if stop_reason is None:
        stop_reason = "effective-target" if effective_updates >= args.effective_updates else "group-cap"

    groups_effective_rate = (
        (effective_updates / groups_sampled) if groups_sampled else None
    )
    metrics = dict(manifest)
    metrics.update(
        {
            "status": status,
            "abort_reason": abort_reason,
            "stop_reason": stop_reason,
            "utc_end": utc_now(),
            "wall_time_s": round(wall_time_s, 3),
            "rollouts_path": str(out_path),
            "adapter_path": adapter_dir,
            "records_written": records_written,
            "grader_cache": {"entries": len(cache), "hits": cache.hits, "misses": cache.misses},
            "counts": {
                "groups_sampled": groups_sampled,
                "groups_effective": effective_updates,
                "effective_updates": effective_updates,
                "groups_skipped": groups_skipped,
                "groups_zero_variance": groups_zero_variance,
                "groups_errored": groups_errored,
                "rollouts_consumed": rollouts_consumed,
                "optimizer_steps": effective_updates,
                "records_written": records_written,
                "admitted": admitted_total,
                "holds": holds_total,
                "shortfall_effective_updates": max(0, args.effective_updates - effective_updates),
            },
            "endpoints": {
                "effective_group_rate": groups_effective_rate,
                "effective_updates": effective_updates,
                "target_effective_updates": int(args.effective_updates),
                "shortfall_effective_updates": max(0, args.effective_updates - effective_updates),
                "groups_sampled": groups_sampled,
                "groups_skipped": groups_skipped,
                "groups_zero_variance": groups_zero_variance,
                "rollouts_consumed": rollouts_consumed,
                "rollouts_per_effective_update": (
                    (rollouts_consumed / effective_updates) if effective_updates else None
                ),
                "mean_group_variance": (variance_total / groups_sampled) if groups_sampled else None,
                "mean_abs_advantage_admitted": (
                    (admitted_advantage_total / admitted_advantage_count)
                    if admitted_advantage_count
                    else None
                ),
                "admission_rate": (admitted_total / records_written) if records_written else None,
                "train_joint_success": (holds_total / records_written) if records_written else None,
                "train_joint_success_given_admitted": (
                    (holds_total / admitted_total) if admitted_total else None
                ),
            },
            "spend": {
                "hourly_rate_usd": float(args.hourly_rate),
                "wall_hours": round(wall_time_s / 3600.0, 6),
                "estimated_spend_usd": round(float(args.hourly_rate) * wall_time_s / 3600.0, 6),
                "cap_usd": GPU_BUDGET_CAP_USD,
            },
            "groups": groups,
        }
    )
    _write_json(metrics_path, metrics)

    print(
        f"[train] arm={args.arm} status={status} stop={stop_reason} "
        f"effective={effective_updates}/{args.effective_updates} "
        f"groups={groups_sampled}/{args.max_groups} rollouts={rollouts_consumed} "
        f"records={records_written} wall={wall_time_s:.1f}s",
        file=sys.stderr,
    )
    print(f"[train] rollouts -> {out_path}", file=sys.stderr)
    print(f"[train] metrics  -> {metrics_path}", file=sys.stderr)
    if adapter_dir:
        print(f"[train] adapter  -> {adapter_dir}", file=sys.stderr)

    return 0 if status == "complete" else 3


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.g < 2:
        parser.error("--g must be >= 2 (a group of one sample has no variance)")
    if args.effective_updates < 1:
        parser.error("--effective-updates must be >= 1")
    if args.max_groups < 1:
        parser.error("--max-groups must be >= 1")
    if args.prompts_per_update < 1:
        parser.error("--prompts-per-update must be >= 1")
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
        print("[train] interrupted", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001 - a crash must print, not swallow, the traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
