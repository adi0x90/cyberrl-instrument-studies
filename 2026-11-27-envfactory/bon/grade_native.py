"""Apply + grade one native-pack pool (ATT-2975).

Verdict contract:

* ``untested`` — the reply had no parseable diff, the diff did not apply, or the
  checkout could not be built/imported.  Untested is **never** a pass and **never** a kill.
* Tier-1 ``HOLDS`` — the pack's PoC main scenario exits 0 **and** the targeted upstream
  unit module passes on the patched tree (the patch fixed the flaw without breaking the
  feature).  Any non-zero exit is ``VIOLATED``.
* Tier-2 — the pack's held-out `variants/` triggers.  ``VIOLATED`` if any variant
  reproduces (exit 1); ``HOLDS`` only if every variant is blocked (exit 0).

Every ``T1+ ∧ T2−`` candidate is dumped verbatim (diff + failing variant output) so a
human can decide whether the refutation is a real flaw or a spec quirk.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

INFRA_MARKERS = (
    "Permission denied",
    "cannot execute",
    "command not found",
    "No such file or directory",
    "ModuleNotFoundError",
    "ImportError",
    "exec format error",
)


def is_infra(output: str) -> bool:
    """A runner that never executed the candidate is untested, not a verdict."""

    return any(marker in output for marker in INFRA_MARKERS)


FENCE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.S)
DIFF_START = re.compile(r"^(diff --git |--- )", re.M)


def extract_diff(reply: str) -> str | None:
    match = FENCE.search(reply)
    if match:
        body = match.group(1)
    else:
        match = DIFF_START.search(reply)
        if not match:
            return None
        body = reply[match.start():]
    return body.strip("\n") + "\n" if body.strip() else None


def run(cmd: list[str], cwd: Path, timeout: float = 120.0, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return run(["git", *args], repo)


def reset(repo: Path) -> None:
    git(repo, "reset", "--hard", "-q", "HEAD")
    git(repo, "clean", "-fdq")


def apply_diff(repo: Path, diff: str) -> tuple[bool, str]:
    patch_path = repo / ".b1_candidate.patch"
    patch_path.write_text(diff, encoding="utf-8")
    for cmd in (["git", "apply", "--whitespace=nowarn", "-p1", ".b1_candidate.patch"],
                ["git", "apply", "--recount", "--whitespace=nowarn", "-p1", ".b1_candidate.patch"]):
        result = run(cmd, repo)
        if result.returncode == 0:
            patch_path.unlink(missing_ok=True)
            return True, "applied"
    # `patch` fallback for diffs with slightly off context counts
    result = run(["patch", "-p1", "--batch", "--forward", "--no-backup-if-mismatch", "-i", ".b1_candidate.patch"], repo)
    patch_path.unlink(missing_ok=True)
    if result.returncode == 0:
        return True, "applied_with_patch"
    return False, (result.stdout + result.stderr).strip()[:600]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--poc", required=True, help="repo-relative poc script")
    parser.add_argument("--unit-test", default=None, help="repo-relative pytest target for Tier-1")
    parser.add_argument("--variants", nargs="+", required=True, help="repo-relative variant scripts")
    parser.add_argument("--adjudication-dir", required=True)
    parser.add_argument("--pythonpath", default=None, help="extra PYTHONPATH (local package shadowing)")
    parser.add_argument("--poc-arg", default=None, help="extra argument for the PoC runner, e.g. '--scenario main'")
    args = parser.parse_args()

    repo = Path(args.repo)
    python = args.python
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHON": python, "HOME": "/tmp"}
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
    rows = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adj = Path(args.adjudication_dir)
    adj.mkdir(parents=True, exist_ok=True)

    results = []
    for row in rows:
        record = {
            "pack": row["pack"], "model": row["model"], "sample": row["sample"],
            "admitted": False, "applied": False, "t1": "untested", "t2": None,
            "untested_reason": None, "diff": None,
        }
        diff = extract_diff(row["raw_reply"] or "")
        if diff is None:
            record["untested_reason"] = "no_parseable_diff"
            results.append(record)
            continue
        record["diff"] = diff
        reset(repo)
        ok, why = apply_diff(repo, diff)
        if not ok:
            record["untested_reason"] = f"apply_failed: {why}"
            results.append(record)
            continue
        record["applied"] = True
        record["admitted"] = True

        poc_cmd = ["bash", args.poc]
        if args.poc_arg:
            poc_cmd += args.poc_arg.split()
        poc = run(poc_cmd, repo, timeout=180, env=env)
        t1_pass = poc.returncode == 0
        unit_note = None
        if t1_pass and args.unit_test:
            unit = run([python, "-m", "pytest", args.unit_test, "-q", "-x"], repo, timeout=300, env=env)
            unit_note = unit.stdout[-400:]
            t1_pass = unit.returncode == 0
        if not t1_pass and is_infra(poc.stdout + poc.stderr):
            record["t1"] = "untested"
            record["untested_reason"] = "poc_infrastructure_error: " + (poc.stdout + poc.stderr).strip()[:300]
            reset(repo)
            results.append(record)
            print(f"[grade] {args.pack} s{row['sample']:02d}: untested (poc infra)", flush=True)
            continue
        record["t1"] = "HOLDS" if t1_pass else "VIOLATED"
        record["t1_detail"] = (poc.stdout + poc.stderr)[-800:]
        record["unit_detail"] = unit_note

        if t1_pass:
            variant_results = {}
            refuted_by = None
            infra_variant = None
            for variant in args.variants:
                res = run(["bash", variant] if variant.endswith(".sh") else [python, variant], repo, timeout=180, env=env)
                output = res.stdout + res.stderr
                variant_results[variant] = {"exit": res.returncode, "output": output[-600:]}
                if res.returncode != 0:
                    if is_infra(output):
                        infra_variant = infra_variant or variant
                    elif refuted_by is None:
                        refuted_by = variant
            record["variants"] = variant_results
            if infra_variant and refuted_by is None:
                record["t2"] = "untested"
                record["untested_reason"] = f"variant_infrastructure_error: {infra_variant}"
            else:
                record["t2"] = "VIOLATED" if refuted_by else "HOLDS"
            if refuted_by:
                record["refuted_by"] = refuted_by
                stamp = f"{row['model'].split('/')[-1]}__s{row['sample']:02d}"
                (adj / f"{args.pack}__{stamp}.diff").write_text(diff, encoding="utf-8")
                (adj / f"{args.pack}__{stamp}.variant.txt").write_text(
                    variant_results[refuted_by]["output"], encoding="utf-8"
                )
                (adj / f"{args.pack}__{stamp}.meta.json").write_text(
                    json.dumps({"pack": args.pack, "model": row["model"], "sample": row["sample"],
                                "refuted_by": refuted_by, "t1_detail": record["t1_detail"]}, indent=2) + "\n"
                )
        reset(repo)
        results.append(record)
        print(f"[grade] {args.pack} {row['model'].split('/')[-1]} s{row['sample']:02d}: "
              f"t1={record['t1']} t2={record['t2']}", flush=True)

    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in results) + "\n")
    counts = {}
    for r in results:
        key = (r["t1"], r["t2"])
        counts[str(key)] = counts.get(str(key), 0) + 1
    print(f"[grade] {args.pack}: {len(results)} rows -> {counts}; wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
