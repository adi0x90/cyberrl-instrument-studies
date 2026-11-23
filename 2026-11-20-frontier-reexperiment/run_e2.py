#!/usr/bin/env python3
"""E2 — grader adequacy: what the weak grader accepts and the obligation grader rejects.

Question: on a fixed candidate corpus, how often does the *cheap* check (test cases
only: grade_weak) call a patch correct while the *obligation* check (no-op / feature
deletion / symptom suppression / AST wrapper / scope) calls it wrong — and does the
obligation grader ever reject a genuinely correct fix?

Candidates are built locally; E2 makes no model calls. For every task:

  seeded_bad entry          -> 1 candidate (the seeded entry itself)
                            -> 3 programmatic mutations of that entry, kept only if the
                               result still parses:
                                 swap_op            swap the first comparison operator
                                 invert_membership  invert the first membership test
                                 weaken_guard       wrap the expression and add `or True`
  reference_fix             -> 1 candidate (positive control)
  alternate_fixes[]         -> n candidates (positive controls)
  adversarial               -> hand-written "obligations pass, root cause intact" cases,
                               reported in their own section (never pooled)
  admitted E1 patches       -> every admitted E1 patch from an E1 JSONL file, if present

Endpoints recomputed in analysis/summarize.py:
  * weak false-accept rate per seeded-bad family (weak HOLDS while obligation VIOLATED)
  * obligation false-reject rate on reference / alternate fixes
  * paired McNemar exact between the two graders on identical candidates
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.graders import grade_obligation, grade_weak
from harness.tasks_io import execute_patch, load_task

# --- shared runner scaffolding -------------------------------------------
# NOTE: this block is intentionally kept identical in run_e1/run_e2/run_e3 so each
# runner script is self-contained. tests/test_runners_dryrun.py asserts all three
# expose the same CLI / JSONL / manifest / resume contract.

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"

DEFAULT_MODELS = (
    "tencent/hy4-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash-1017",
)

BASE_SEED = 20261120

# Stopping rule 2 (docs/DESIGN.md): a run that raises on ten consecutive units is
# broken, not unlucky -- stop instead of burning the call budget.
ABORT_AFTER_CONSECUTIVE_ERRORS = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = proc.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_file_map() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(TASKS_DIR.glob("*.json"))}


def config_drift(
    previous_manifest: Path, *, seed: int, models: list[str], mock: bool
) -> list[str]:
    """Seed / model-list / mock-mode mismatches against the previous run's manifest.

    Resuming with a different configuration would skip units recorded under different
    conditions and pool two incompatible samples, so the runner says so out loud.
    """
    if not previous_manifest.exists():
        return []
    try:
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken manifest must not block a run
        return []
    drift: list[str] = []
    if previous.get("seed") != seed:
        drift.append(f"seed {previous.get('seed')} -> {seed}")
    if list(previous.get("models") or []) != list(models):
        drift.append(f"models {previous.get('models')} -> {list(models)}")
    if bool(previous.get("mock")) != bool(mock):
        drift.append(f"mock {previous.get('mock')} -> {bool(mock)}")
    return drift


def task_sha_drift(previous_manifest: Path) -> list[str]:
    """Task files whose sha256 differs from the previous run's manifest.

    Mixed-shas are not pooled (see docs/DESIGN.md, stopping rules): the runner warns,
    records the drift in the manifest, and leaves the decision to the operator.
    """
    if not previous_manifest.exists():
        return []
    try:
        previous = json.loads(previous_manifest.read_text(encoding="utf-8")).get("task_sha256") or {}
    except Exception:  # noqa: BLE001 - a broken manifest must not block a run
        return []
    current = task_file_map()
    return sorted(
        name for name in set(previous) | set(current) if previous.get(name) != current.get(name)
    )


def parse_models(spec: str) -> list[str]:
    models = [chunk.strip() for chunk in str(spec).split(",") if chunk.strip()]
    if not models:
        raise SystemExit("--models resolved to an empty list")
    return models


def build_parser(description: str, default_out: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use deterministic mocked model replies (no network, no key)",
    )
    parser.add_argument("--dry-run", dest="mock", action="store_true", help="alias for --mock")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N tasks (sorted by task id)",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma separated model ids")
    parser.add_argument("--out", default=str(default_out), help="JSONL output path")
    parser.add_argument("--resume", action="store_true", help="skip units already present in --out")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="base seed; decoding d uses seed+d")
    return parser


def read_done_keys(path: Path) -> set[str]:
    """Unit keys already present in the out file (torn trailing lines are ignored)."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("unit")
            if isinstance(key, str):
                done.add(key)
    return done


def emit(handle, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def manifest_path(out: Path) -> Path:
    return out.with_name(out.stem + ".manifest.json")


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


_LOADED: dict[str, dict] = {}


def load_tasks(limit: int | None) -> list[tuple[str, dict]]:
    files = sorted(TASKS_DIR.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return [(path.stem, _load_one(path.stem)) for path in files]


def _load_one(task_id: str) -> dict:
    """Load one task through harness.tasks_io.load_task.

    tasks_io resolves ``<root>/tasks/<task_id>.json``, so the experiment root is the
    documented root. The tasks directory itself is tried as a fallback in case that
    contract moves; the loader stays harness-owned and nothing is reimplemented here.
    """
    if task_id in _LOADED:
        return _LOADED[task_id]
    last: Exception | None = None
    for root in (ROOT, TASKS_DIR):
        try:
            task = load_task(task_id, root)
        except Exception as exc:  # noqa: BLE001 - surface the real loader error
            last = exc
            continue
        _LOADED[task_id] = task
        return task
    raise RuntimeError(f"load_task failed for {task_id!r}: {last!r}")


def require_key_unless_mock(mock: bool) -> None:
    """Fail fast in real mode when the documented env var is absent.

    Presence check only: the value is never read, copied, logged, or written.
    """
    if mock:
        return
    if "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit(
            "real run requires OPENROUTER_API_KEY in the environment "
            "(use --mock / --dry-run to run offline)"
        )


def warn_unknown_models(models: list[str], mock: bool) -> None:
    """Warn (never substitute) if a requested id is not in the harness model set."""
    try:
        from harness.models import available_models
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] harness.models unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    try:
        known = set(available_models(mock=mock))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] available_models() unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return
    unknown = [model for model in models if model not in known]
    if unknown:
        print(f"[warn] not returned by available_models(): {unknown}", file=sys.stderr)


@dataclass(frozen=True)
class Sampling:
    """One decoding: temperature plus the seed that makes it replayable."""

    decoding: int
    temperature: float
    seed: int


def sampling_table(base_seed: int, temperatures: dict[int, float]) -> dict[int, Sampling]:
    return {
        decoding: Sampling(decoding=decoding, temperature=float(temp), seed=base_seed + decoding)
        for decoding, temp in temperatures.items()
    }


def experiment_argv() -> list[str]:
    return [Path(sys.argv[0]).name] + list(sys.argv[1:])


def write_run_manifest(
    path: Path,
    *,
    experiment: str,
    argv: list[str],
    models: list[str],
    seed: int,
    mock: bool,
    limit: int | None,
    out: Path,
    samplings: dict[int, Sampling],
    extra: dict | None = None,
    status: str = "running",
    records_written: int | None = None,
    records_skipped: int | None = None,
) -> None:
    payload: dict = {
        "experiment": experiment,
        "git_sha": git_sha(),
        "utc": utc_now(),
        "models": models,
        "task_sha256": task_file_map(),
        "argv": argv,
        "seed": seed,
        "mock": bool(mock),
        "limit": limit,
        "out": str(out),
        "decodings": {
            str(decoding): {"temperature": s.temperature, "seed": s.seed}
            for decoding, s in sorted(samplings.items())
        },
        "python": sys.version.split()[0],
        "status": status,
    }
    if extra:
        payload.update(extra)
    if records_written is not None:
        payload["records_written"] = records_written
    if records_skipped is not None:
        payload["records_skipped"] = records_skipped
    write_manifest(path, payload)


# --- E2 specifics ---------------------------------------------------------

EXPERIMENT = "E2"
DEFAULT_OUT = RESULTS_DIR / "e2.jsonl"

# E2 samples nothing: every candidate is graded deterministically. The single
# pseudo-decoding keeps the record schema uniform across the three experiments.
DECODING_TEMPERATURES = {0: 0.0}

LOCAL_MODEL_TAG = "none"  # no model produced these candidates

# Candidate kinds. Only the first four are pooled into the per-family false-accept
# rate; the rest are reported separately (see analysis/summarize.py).
KIND_SEEDED = "seeded"
KIND_SWAP_OP = "mutation:swap_op"
KIND_INVERT = "mutation:invert_membership"
KIND_WEAKEN = "mutation:weaken_guard"
KIND_REFERENCE = "reference"
KIND_ALTERNATE = "alternate"
KIND_ADVERSARIAL = "adversarial"
KIND_E1_PATCH = "e1_patch"

POOLED_KINDS = (KIND_SEEDED, KIND_SWAP_OP, KIND_INVERT, KIND_WEAKEN)
NEGATIVE_KINDS = POOLED_KINDS + (KIND_E1_PATCH,)

_OPERATOR_PATTERN = re.compile(r"==|!=|<=|>=|<|>")
_OPERATOR_SWAPS = {
    "==": "!=",
    "!=": "==",
    "<": ">=",
    ">": "<=",
    "<=": ">",
    ">=": "<",
}
_WORD_PATTERN = re.compile(r"\b(not\s+in|is\s+not|in|is)\b")
_WORD_SWAPS = {
    "not in": "in",
    "is not": "is",
    "in": "not in",
    "is": "is not",
}


def parses_as_expression(expression: str) -> bool:
    """True when ``expression`` is a single parseable Python expression.

    This is a *generation-time* filter (may I keep this mutation?), not a
    reimplementation of harness execution: pass/fail on the task's cases is still
    decided by harness.tasks_io.execute_patch.
    """
    if not expression or not expression.strip():
        return False
    try:
        ast.parse(expression, mode="eval")
        return True
    except SyntaxError:
        return False


def swap_comparison(expression: str) -> str | None:
    match = _OPERATOR_PATTERN.search(expression)
    if not match:
        return None
    token = match.group(0)
    return expression[: match.start()] + _OPERATOR_SWAPS[token] + expression[match.end():]


def invert_membership(expression: str) -> str | None:
    match = _WORD_PATTERN.search(expression)
    if not match:
        return None
    token = re.sub(r"\s+", " ", match.group(0))
    return expression[: match.start()] + _WORD_SWAPS[token] + expression[match.end():]


def weaken_guard(expression: str) -> str:
    return f"({expression}) or True"


MUTATORS = (
    (KIND_SWAP_OP, "swap the first comparison operator", swap_comparison),
    (KIND_INVERT, "invert the first membership / identity test", invert_membership),
    (KIND_WEAKEN, "weaken the guard with an `or True` clause", weaken_guard),
)


def expression_digest(expression: str) -> str:
    return hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Candidate:
    kind: str
    seeded_family: str
    expression: str
    detail: str = ""
    model: str = LOCAL_MODEL_TAG
    source: str = "local"


def _why(entry: dict) -> str:
    return str(entry.get("why", "") or "")


def seeded_candidates(task: dict) -> list[Candidate]:
    """Seeded entry + up to three parseable programmatic mutations per seeded family."""
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()

    def add(candidate: Candidate) -> None:
        key = (candidate.kind, candidate.expression)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    for entry in task.get("seeded_bad") or []:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family", "unknown"))
        expression = str(entry.get("expression", "") or "")
        if not expression:
            continue
        add(Candidate(KIND_SEEDED, family, expression, detail=_why(entry)))
        for kind, description, mutator in MUTATORS:
            mutated = mutator(expression)
            if mutated is None or mutated == expression:
                continue
            if not parses_as_expression(mutated):
                continue
            add(Candidate(kind, family, mutated, detail=f"{description} (from {family})"))
    return candidates


def control_candidates(task: dict) -> list[Candidate]:
    """Reference and alternate fixes: the obligation grader must not reject these."""
    candidates: list[Candidate] = []
    reference = task.get("reference_fix")
    if isinstance(reference, str) and reference.strip():
        candidates.append(Candidate(KIND_REFERENCE, "reference", reference, detail="reference fix"))
    for index, alternate in enumerate(task.get("alternate_fixes") or []):
        if isinstance(alternate, str) and alternate.strip():
            candidates.append(
                Candidate(KIND_ALTERNATE, "reference", alternate, detail=f"alternate fix #{index + 1}")
            )
    return candidates


def _adversarial_expressions(value) -> list[tuple[str, str]]:
    """Normalise the optional task-level ``adversarial`` field.

    Accepted shapes: a string expression, ``{"expression": ..., "why": ...}``, or a
    list of either. Unknown shapes are skipped rather than guessed at.
    """
    items = value if isinstance(value, list) else [value]
    out: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append((item, ""))
        elif isinstance(item, dict):
            expression = item.get("expression")
            if isinstance(expression, str) and expression.strip():
                out.append((expression, str(item.get("why", "") or "")))
    return out


def adversarial_candidates(task: dict) -> list[Candidate]:
    candidates: list[Candidate] = []
    for expression, why in _adversarial_expressions(task.get("adversarial")):
        candidates.append(
            Candidate(
                KIND_ADVERSARIAL,
                "adversarial",
                expression,
                detail=why or "hand-written: obligations may pass, root cause intact",
            )
        )
    return candidates


def load_e1_patches(paths: list[Path], allowed_tasks: set[str]) -> list[dict]:
    """Every admitted E1 patch, deduplicated by (task, model, expression)."""
    for path in paths:
        if not path or not path.exists():
            continue
        patches: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("experiment") not in (None, "E1"):
                    continue
                if not record.get("admitted"):
                    continue
                expression = record.get("raw_expression")
                task_id = record.get("task_id")
                if not isinstance(expression, str) or not expression.strip():
                    continue
                if not isinstance(task_id, str) or task_id not in allowed_tasks:
                    continue
                model = str(record.get("model") or LOCAL_MODEL_TAG)
                key = (task_id, model, expression)
                if key in seen:
                    continue
                seen.add(key)
                patches.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "expression": expression,
                        "source_unit": record.get("unit"),
                    }
                )
        print(f"[e2] imported {len(patches)} admitted E1 patches from {path}", file=sys.stderr)
        return patches
    return []


def e1_patch_candidates(patches: list[dict]) -> list[Candidate]:
    return [
        Candidate(
            KIND_E1_PATCH,
            "e1_patch",
            patch["expression"],
            detail=f"admitted E1 patch (unit {patch.get('source_unit')})",
            model=patch["model"],
            source="e1",
        )
        for patch in patches
    ]


def resolve_e1_paths(explicit: str | None, mock: bool) -> list[Path]:
    if explicit:
        return [Path(explicit)]
    candidates = [RESULTS_DIR / "e1.jsonl", RESULTS_DIR / "mock" / "e1.jsonl"]
    if mock:
        candidates = [RESULTS_DIR / "mock" / "e1.jsonl", RESULTS_DIR / "e1.jsonl"]
    return candidates


def grade_candidate(task: dict, candidate: Candidate) -> dict:
    result = execute_patch(task, candidate.expression)
    weak = str(grade_weak(task, result))
    verdict, reasons = grade_obligation(task, result, candidate.expression)
    return {
        "weak_verdict": weak,
        "verdict": str(verdict),
        "reasons": [str(reason) for reason in (reasons or [])],
        "compiled": bool(getattr(result, "compiled", False)),
        "detail": candidate.detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(
        "E2: grade a fixed candidate corpus with the weak and obligation graders.",
        DEFAULT_OUT,
    )
    parser.add_argument(
        "--e1",
        default=None,
        help="E1 JSONL to import admitted patches from (default: results/e1.jsonl, else results/mock/e1.jsonl)",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    # E2 makes no model calls, so it needs no key even in real mode.
    models = parse_models(args.models)
    warn_unknown_models(models, bool(args.mock))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(out)
    samplings = sampling_table(args.seed, DECODING_TEMPERATURES)

    drift = task_sha_drift(mpath) if args.resume else []
    if drift:
        print(
            f"[warn] task files changed since the previous run of this out file: {drift}. "
            "Mixed git-sha/task-sha results must not be pooled: use a fresh --out directory.",
            file=sys.stderr,
        )
    config_drifted = (
        config_drift(mpath, seed=args.seed, models=models, mock=bool(args.mock))
        if args.resume
        else []
    )
    if config_drifted:
        print(
            f"[warn] configuration changed since the previous run of this out file: "
            f"{'; '.join(config_drifted)}. Resume assumes identical conditions; use a fresh "
            "--out directory if this is a different sample.",
            file=sys.stderr,
        )

    done = read_done_keys(out) if args.resume else set()
    if not args.resume and out.exists():
        out.write_text("", encoding="utf-8")

    tasks = load_tasks(args.limit)
    if not tasks:
        print(f"[warn] no task files under {TASKS_DIR}", file=sys.stderr)
    allowed_tasks = {task_id for task_id, _ in tasks}
    e1_paths = resolve_e1_paths(args.e1, bool(args.mock))
    patches_by_task: dict[str, list[dict]] = {}
    for patch in load_e1_patches(e1_paths, allowed_tasks):
        patches_by_task.setdefault(patch["task_id"], []).append(patch)

    # Flat, ordered candidate list: task -> seeded + mutations + controls + adversarial + E1 patches.
    units: list[tuple[str, dict, Candidate]] = []
    for task_id, task in tasks:
        for candidate in (
            seeded_candidates(task)
            + control_candidates(task)
            + adversarial_candidates(task)
            + e1_patch_candidates(patches_by_task.get(task_id, []))
        ):
            units.append((task_id, task, candidate))

    base_manifest = {
        "experiment": EXPERIMENT,
        "argv": experiment_argv(),
        "models": models,
        "seed": args.seed,
        "mock": bool(args.mock),
        "limit": args.limit,
        "out": out,
        "samplings": samplings,
        "extra": {
            "task_sha256_drift": drift,
            "config_drift": config_drifted,
            "e1_inputs_considered": [str(path) for path in e1_paths],
            "e1_patches_imported": sum(len(items) for items in patches_by_task.values()),
            "candidate_kinds": [
                KIND_SEEDED,
                KIND_SWAP_OP,
                KIND_INVERT,
                KIND_WEAKEN,
                KIND_REFERENCE,
                KIND_ALTERNATE,
                KIND_ADVERSARIAL,
                KIND_E1_PATCH,
            ],
        },
    }
    write_run_manifest(mpath, status="running", records_written=0, records_skipped=0, **base_manifest)

    attempted = written = skipped = 0
    consecutive_errors = 0
    last_error = ""
    abort_reason: str | None = None
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        for task_id, task, candidate in units:
            attempted += 1
            unit = (
                f"{task_id}|{candidate.kind}|{candidate.seeded_family}|"
                f"{candidate.model}|{expression_digest(candidate.expression)}"
            )
            if unit in done:
                skipped += 1
                continue
            record = {
                "unit": unit,
                "experiment": EXPERIMENT,
                "task_id": task_id,
                "model": candidate.model,
                "decoding": None,
                "temperature": None,
                "seed": args.seed,
                "ts_utc": utc_now(),
                "admitted": False,
                "admission_reason": "",
                "raw_expression": candidate.expression,
                "verdict": "UNKNOWN",
                "reasons": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "http_status": 0,
                "weak_verdict": "UNKNOWN",
                "candidate_kind": candidate.kind,
                "family": str(task.get("family", "unknown")),
                "seeded_family": candidate.seeded_family,
                "source": candidate.source,
                "detail": candidate.detail,
                "compiled": False,
            }
            try:
                graded = grade_candidate(task, candidate)
                record.update(graded)
                record["admitted"] = bool(graded["compiled"] and graded["weak_verdict"] != "UNKNOWN")
                record["admission_reason"] = "compiled" if graded["compiled"] else "did not compile"
                consecutive_errors = 0
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the run
                consecutive_errors += 1
                last_error = f"{type(exc).__name__}: {exc}"
                record["verdict"] = "UNKNOWN"
                record["weak_verdict"] = "UNKNOWN"
                record["reasons"] = [last_error]
                record["error"] = last_error
            emit(handle, record)
            done.add(unit)
            written += 1
            if consecutive_errors >= ABORT_AFTER_CONSECUTIVE_ERRORS:
                abort_reason = (
                    f"{consecutive_errors} consecutive candidate exceptions; last: {last_error}"
                )
                break
            if written % 50 == 0:
                print(f"[e2] {written} records written (last unit {unit})", file=sys.stderr)

    status = "aborted_consecutive_errors" if abort_reason else "complete"
    base_manifest["extra"]["abort_reason"] = abort_reason
    write_run_manifest(
        mpath,
        status=status,
        records_written=written,
        records_skipped=skipped,
        **base_manifest,
    )
    print(
        f"[e2] {status}: {written} written, {skipped} skipped, {attempted} candidates considered "
        f"-> {out} (+ {mpath.name})",
        file=sys.stderr,
    )
    if abort_reason:
        print(f"[e2] stopped: {abort_reason}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
