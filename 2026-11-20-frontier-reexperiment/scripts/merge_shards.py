#!/usr/bin/env python3
"""Merge the per-model E1 / E3 JSONL shards into one authoritative JSONL + manifest.

The runners cannot run three models at once on the VPS without paying the wall-clock
cost three times, so each model was run as its own shard:

    results/e1_ds41.jsonl   results/e1_glm.jsonl   results/e1_hy4.jsonl
    results/e3_ds41.jsonl   results/e3_glm.jsonl   results/e3_hy4.jsonl

This script concatenates the shards for one experiment into the single file the
analyser reads, and writes the sibling manifest that makes the merged file
identifiable. It touches no record content: the merged file is exactly the shard
lines in sorted-shard order, so the study's measured data is unchanged by merging.

Usage (from the experiment root, on the VPS or afterwards):

    python scripts/merge_shards.py --exp e1
    python scripts/merge_shards.py --exp e3 --results-dir evidence --out evidence/e3.jsonl

What the merged manifest is
---------------------------
The first shard's manifest is the template (it already carries the seed, the decoding
table, the task hashes and the run's git sha). These fields are then overwritten so the
manifest describes the *merged* run rather than shard one:

    experiment, models   the three SPEC model ids, in SPEC order
    out                  the merged JSONL path
    argv                 one-line note naming this merge
    records_written      lines written to the merged file
    shards               the shard paths that went in, in merge order
    records_skipped      sum of the shards' skipped counts

Everything else (seed, decodings, task_sha256, git_sha, utc, ...) is inherited from the
first shard. Deliberately deterministic: no clock is read, so the same shards always
produce the same manifest. The inherited `utc` is the first shard's start time, not the
merge time; the merge itself is not an observation.

Exit codes: 0 merged, 2 unusable input (no shards, missing shard manifest, unparseable
line, --out would overwrite a shard).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: The three model ids fixed by docs/SPEC.md. Never substitute another id: the
#: merged manifest must name exactly the models the study claims to have run.
SPEC_MODELS: tuple[str, ...] = (
    "tencent/hy4-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash-1017",
)

EXPERIMENTS = ("e1", "e3")


class MergeError(Exception):
    """Unusable input; reported on stderr with exit code 2."""


def shard_paths(exp: str, results_dir: Path) -> list[Path]:
    """The per-model shards for ``exp``, in deterministic (sorted) order.

    ``<exp>_*.jsonl`` deliberately cannot match the merged ``<exp>.jsonl`` or a
    shard's ``.manifest.json``, so a merge never consumes its own output.
    """

    return sorted(results_dir.glob(f"{exp}_*.jsonl"))


def sibling_manifest(jsonl_path: Path) -> Path:
    """``results/e1_ds41.jsonl`` -> ``results/e1_ds41.manifest.json``."""

    return jsonl_path.parent / (jsonl_path.name[: -len(".jsonl")] + ".manifest.json")


def merge_jsonl(shards: list[Path], out: Path) -> int:
    """Concatenate ``shards`` into ``out``; return the record count.

    Every non-blank line must be a JSON object before it is written: a torn trailing
    line from a killed run is a fault to fix at the source, not something to smuggle
    into the authoritative file.
    """

    temporary = out.parent / (out.name + ".tmp")
    written = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for shard in shards:
                with shard.open("r", encoding="utf-8") as source:
                    for number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise MergeError(
                                f"{shard} line {number} is not JSON "
                                f"({error.msg}); fix or re-run that shard before merging"
                            ) from None
                        if not isinstance(record, dict):
                            raise MergeError(f"{shard} line {number} is not a JSON object")
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        written += 1
        os.replace(temporary, out)
    finally:
        if temporary.exists():
            temporary.unlink()
    return written


def merged_manifest(
    exp: str,
    shards: list[Path],
    out: Path,
    first_manifest: Path,
    records_written: int,
) -> dict:
    """Build the merged manifest from the first shard's manifest."""

    template = json.loads(first_manifest.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise MergeError(f"{first_manifest} is not a JSON object")

    records_skipped = 0
    for shard in shards:
        path = sibling_manifest(shard)
        if not path.exists():
            continue
        shard_manifest = json.loads(path.read_text(encoding="utf-8"))
        records_skipped += int(shard_manifest.get("records_skipped") or 0)

    template.update(
        {
            "experiment": exp.upper(),
            "models": list(SPEC_MODELS),
            "out": str(out),
            "argv": [
                f"scripts/merge_shards.py --exp {exp} "
                f"(merged {len(shards)} per-model shards)"
            ],
            "records_written": records_written,
            "records_skipped": records_skipped,
            "shards": [str(shard) for shard in shards],
        }
    )
    template.setdefault("status", "complete")
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-model JSONL shards into results/<exp>.jsonl + manifest.",
    )
    parser.add_argument("--exp", required=True, choices=EXPERIMENTS, help="experiment to merge")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="directory holding <exp>_*.jsonl shards (default: results)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="merged JSONL path (default: <results-dir>/<exp>.jsonl)",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / f"{args.exp}.jsonl"

    if not results_dir.is_dir():
        print(f"error: results directory does not exist: {results_dir}", file=sys.stderr)
        return 2

    shards = shard_paths(args.exp, results_dir)
    if not shards:
        print(
            f"error: no {args.exp}_*.jsonl shards under {results_dir}"
            " (expected one per model: _ds41, _glm, _hy4)",
            file=sys.stderr,
        )
        return 2

    resolved_out = out.resolve()
    for shard in shards:
        if shard.resolve() == resolved_out:
            print(f"error: --out {out} is one of the shards; choose another path", file=sys.stderr)
            return 2

    first_manifest = sibling_manifest(shards[0])
    if not first_manifest.exists():
        print(
            f"error: shard manifest missing: {first_manifest}"
            " (the shard must finish before it can be merged)",
            file=sys.stderr,
        )
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        records_written = merge_jsonl(shards, out)
        manifest = merged_manifest(args.exp, shards, out, first_manifest, records_written)
    except MergeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    manifest_path = sibling_manifest(out)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"merged {len(shards)} shard(s) -> {out}")
    for shard in shards:
        print(f"  + {shard}")
    print(f"{records_written} records; manifest -> {manifest_path}")
    print("models: " + ", ".join(SPEC_MODELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
