#!/usr/bin/env python3
"""E3 — polarity sensitivity: does stating the polarity fact change the expression?

This is the converse-guard experiment. A "converse guard" is an expression that
implements the opposite of the stated property (e.g. ``return True only when the
tenant matches`` answered as ``self.tenant_id != resource.tenant_id``). It can pass
weak case checks while encoding the wrong policy.

Loop: polarity-sensitive task x variant x model x 3 decodings, where
  explicit = the prompt states the task's polarity fact verbatim
  neutral  = the same prompt with the polarity fact removed
Every returned expression is admitted (strict -> fence -> toolcall, first success),
classified with harness.polarity.converse_guard against the task's polarity block,
and logged in full (raw_reply included) for audit.

The polarity *markers* are never shown to the model: showing them would turn the
measurement into a prompt-copying test. Only the natural-language fact is stated.

Endpoints recomputed in analysis/summarize.py:
  * converse-guard rate per (model, variant)
  * Fisher exact (explicit vs neutral) and paired McNemar exact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.envelopes import admit_fence, admit_strict, admit_toolcall
from harness.models import available_models, chat
from harness.polarity import converse_guard, explain
from harness.tasks_io import load_task

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
        help="only the first N polarity-sensitive tasks (sorted by task id)",
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


# --- E3 specifics ---------------------------------------------------------

EXPERIMENT = "E3"
DEFAULT_OUT = RESULTS_DIR / "e3.jsonl"

# 3 decodings: greedy, warm, hot -- each with a fixed recorded seed.
DECODING_TEMPERATURES = {0: 0.0, 1: 0.3, 2: 0.7}

VARIANTS = ("explicit", "neutral")

ADMITTORS = (admit_strict, admit_fence, admit_toolcall)

SYSTEM_RULES = (
    "You are editing one Python expression inside a small security check. Rules:\n"
    "- The answer must be a single Python expression that evaluates to a bool.\n"
    "- No imports, no statements, no newlines inside the expression.\n"
    "- Implement the stated property; do not simply block the example cases.\n"
    "- Output format: exactly one JSON object and nothing else: "
    '{"expression": "<python expression>"}'
)


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


def build_messages(task: dict, variant: str) -> list[dict]:
    """Deterministic prompt; the neutral variant removes only the polarity fact."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    polarity = task.get("polarity") or {}
    property_text = task.get("property", "(no property text)")
    lines = [
        f"Task id: {task.get('id', '(no id)')}",
        f"Security property: {property_text}",
    ]
    if variant == "explicit":
        fact = str(polarity.get("fact", "") or "").strip()
        lines.append(f"Polarity fact (state this direction exactly, do not invert it): {fact}")
    lines.extend(
        [
            "",
            "Source under edit:",
            "```python",
            str(task.get("source", "")),
            "```",
            "",
            f"The editable slot currently contains the literal marker {task.get('placeholder', 'REPLACE_ME')!r}.",
            "Replace that slot with one Python expression that implements the security property.",
            "",
            "Security cases (must not be satisfied):",
            _case_lines(task.get("security_cases")),
            "",
            "Benign cases (must keep working):",
            _case_lines(task.get("benign_cases")),
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": "\n".join(lines)},
    ]


def extract_expression(text: str) -> tuple[str | None, str, str]:
    """Admit through the three consumers in order; first success wins.

    Returns (expression, admission_reason, admitting_envelope).
    """
    for admittor in ADMITTORS:
        try:
            admission = admittor(text)
        except Exception:  # noqa: BLE001 - an envelope consumer may reject loudly
            continue
        if getattr(admission, "ok", False) and getattr(admission, "expression", None):
            return (
                str(admission.expression),
                str(getattr(admission, "reason", "")),
                str(getattr(admission, "envelope", "unknown")),
            )
    try:
        last = admit_toolcall(text)
        reason = str(getattr(last, "reason", "")) or "no envelope admitted the reply"
    except Exception as exc:  # noqa: BLE001
        reason = f"no envelope admitted the reply ({type(exc).__name__}: {exc})"
    return None, reason, "none"


def run_unit(task: dict, variant: str, model: str, sampling: Sampling, mock: bool) -> dict:
    messages = build_messages(task, variant)
    reply = chat(
        model,
        messages,
        temperature=sampling.temperature,
        seed=sampling.seed,
        mock=mock,
    )
    body: dict = {
        "prompt_tokens": int(getattr(reply, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(reply, "completion_tokens", 0) or 0),
        "http_status": int(getattr(reply, "http_status", 0) or 0),
        "raw_reply": getattr(reply, "text", ""),
    }
    expression, reason, envelope = extract_expression(body["raw_reply"])
    body["admitted"] = expression is not None
    body["admission_reason"] = reason
    body["admission_envelope"] = envelope
    body["raw_expression"] = expression
    if expression is None:
        body["verdict"] = "UNKNOWN"
        body["reasons"] = [f"not admitted: {reason}"]
        return body
    polarity = task.get("polarity") or {}
    verdict = str(converse_guard(expression, polarity))
    try:
        reasons = [str(item) for item in (explain(expression) or [])]
    except Exception:  # noqa: BLE001 - explain() is diagnostic only
        reasons = []
    body["verdict"] = verdict
    body["reasons"] = reasons
    return body


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(
        "E3: polarity-sensitive tasks x prompt variant x model x decoding, classified by converse_guard.",
        DEFAULT_OUT,
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    require_key_unless_mock(bool(args.mock))
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

    # --limit counts polarity-sensitive tasks, so a dry run stays small even if the
    # corpus grows: filter first, then slice.
    selected: list[tuple[str, dict]] = []
    skipped_not_sensitive: list[str] = []
    for task_id, task in load_tasks(None):
        if not task.get("polarity_sensitive"):
            skipped_not_sensitive.append(task_id)
            continue
        if not isinstance(task.get("polarity"), dict) or not task["polarity"]:
            print(f"[warn] {task_id}: polarity_sensitive but no polarity block; skipped", file=sys.stderr)
            continue
        selected.append((task_id, task))
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        print(f"[warn] no polarity-sensitive tasks under {TASKS_DIR}", file=sys.stderr)

    units = [
        (task_id, task, variant, model, decoding, samplings[decoding])
        for task_id, task in selected
        for variant in VARIANTS
        for model in models
        for decoding in sorted(samplings)
    ]

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
            "variants": list(VARIANTS),
            "tasks_selected": [task_id for task_id, _ in selected],
            "tasks_skipped_not_polarity_sensitive": skipped_not_sensitive,
        },
    }
    write_run_manifest(mpath, status="running", records_written=0, records_skipped=0, **base_manifest)

    attempted = written = skipped = 0
    consecutive_errors = 0
    last_error = ""
    abort_reason: str | None = None
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        for task_id, task, variant, model, decoding, sampling in units:
            unit = f"{task_id}|{model}|{variant}|{decoding}"
            attempted += 1
            if unit in done:
                skipped += 1
                continue
            record = {
                "unit": unit,
                "experiment": EXPERIMENT,
                "task_id": task_id,
                "model": model,
                "variant": variant,
                "decoding": decoding,
                "temperature": sampling.temperature,
                "seed": sampling.seed,
                "ts_utc": utc_now(),
                "admitted": False,
                "admission_reason": "",
                "raw_expression": None,
                "verdict": "UNKNOWN",
                "reasons": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "http_status": 0,
                "raw_reply": "",
            }
            try:
                record.update(run_unit(task, variant, model, sampling, bool(args.mock)))
                consecutive_errors = 0
            except Exception as exc:  # noqa: BLE001 - one bad unit must not kill the run
                consecutive_errors += 1
                last_error = f"{type(exc).__name__}: {exc}"
                record["verdict"] = "UNKNOWN"
                record["reasons"] = [last_error]
                record["error"] = last_error
            emit(handle, record)
            done.add(unit)
            written += 1
            if consecutive_errors >= ABORT_AFTER_CONSECUTIVE_ERRORS:
                abort_reason = (
                    f"{consecutive_errors} consecutive unit exceptions; last: {last_error}"
                )
                break
            if written % 25 == 0:
                print(f"[e3] {written} records written (last unit {unit})", file=sys.stderr)

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
        f"[e3] {status}: {written} written, {skipped} skipped, {attempted} units considered "
        f"-> {out} (+ {mpath.name})",
        file=sys.stderr,
    )
    if abort_reason:
        print(f"[e3] stopped: {abort_reason}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
