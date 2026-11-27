#!/usr/bin/env python3
"""Export the five pinned native packs into transportable, verifiable snapshots.

Why snapshots instead of patches: the packs live on local-only branches, so the
remote host cannot fetch them.  A ``git diff`` would depend on patch context
surviving a re-clone; a byte-exact snapshot of the files that differ from the
upstream pin does not.  Each snapshot records the git blob SHA of every file, so
the staged checkout can be verified file-by-file after transfer.

What is exported per pack (under ``evidence/packs/<pack>/``):

* ``provenance.json``  — pin, branch tips, derived injection/metadata commits, blob
  SHAs, and the commit-classification evidence used to find them.
* ``files/<variant>/…`` — every non-``pack/`` file that differs from the pin, for
  ``vulnerable`` / ``reference`` / ``alternate``.
* ``meta/…``           — the ``pack/`` metadata tree (manifest, injection-spec,
  poc, variants, replay) from the vulnerable tip.
* ``manifest.json``    — copy of the pack manifest for human reading.

The dual-commit contract from the Part 3 spec is enforced here: the *source*
injection commit (what the agent may see) and the *metadata* commit (trusted
auditor only) are recorded separately, never conflated.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/adi/targets")
OUT = Path(__file__).resolve().parent.parent / "evidence" / "packs"

PACKS = [
    {
        "pack": "etcd-revocation-skew",
        "repo": "etcd",
        "upstream": "https://github.com/etcd-io/etcd",
        "tag": "v3.5.36",
        "pin": "a5f37e35a47bf71c01a1708deaba210374335f1b",
        "meta_dir": "pack/etcd-revocation-skew",
        "branches": {
            "vulnerable": "pack/etcd-revocation-skew-vulnerable",
            "reference": "pack/etcd-revocation-skew-reference-fix",
            "alternate": "pack/etcd-revocation-skew-alternate-fix",
        },
        "build": {"kind": "go", "dir": "server", "test": "go test ./auth/ -count=1"},
    },
    {
        "pack": "authlib-jwt-dupkey-canonicalization",
        "repo": "authlib",
        "upstream": "https://github.com/lepture/authlib",
        "tag": "v1.9.10",
        "pin": "d17ab9579f762195c20541577badd151e9a14efc",
        "meta_dir": "pack",
        "branches": {
            "vulnerable": "pack/authlib-jwt-dupkey-vulnerable",
            "reference": "pack/authlib-jwt-dupkey-reference-fix",
            "alternate": "pack/authlib-jwt-dupkey-alternate-fix",
        },
        "build": {"kind": "python", "test": "python -m pytest tests/core/test_oauth2 -q"},
    },
    {
        "pack": "coremqtt-publish-varint-payloadlen",
        "repo": "coreMQTT",
        "upstream": "https://github.com/FreeRTOS/coreMQTT",
        "tag": "v5.0.2",
        "pin": "04845c6a8e5f9cf2d232f1c6e80baeb81302e690",
        "meta_dir": "pack/publish-varint-payloadlen",
        "branches": {
            "vulnerable": "pack/publish-varint-payloadlen-vulnerable",
            "reference": "pack/publish-varint-payloadlen-reference-fix",
            "alternate": "pack/publish-varint-payloadlen-alternate-fix",
        },
        "build": {"kind": "c-cmake", "test": "cmake --build build -j8"},
    },
    {
        "pack": "fastmcp-canonical-authz-alias",
        "repo": "fastmcp",
        "upstream": "https://github.com/jlowin/fastmcp",
        "tag": "v4.2.3",
        "pin": "c0185fcf3a346c0d11b69562aaa6c971f82c5c20",
        "meta_dir": "pack/fastmcp-canonical-authz-alias",
        "branches": {
            "vulnerable": "pack/fastmcp-canonical-authz-alias-vulnerable",
            "reference": "pack/fastmcp-canonical-authz-alias-reference-fix",
            "alternate": "pack/fastmcp-canonical-authz-alias-alternate-fix",
        },
        "build": {"kind": "python", "test": "python -m pytest tests/server/middleware -q"},
    },
    {
        "pack": "redis-setrange-grow-undercount",
        "repo": "redis",
        "upstream": "https://github.com/redis/redis",
        "tag": "8.0.6",
        "pin": "f49e5cb2b161418f92f2095c532b85ad24d80819",
        "meta_dir": "pack/setrange-grow-undercount",
        "branches": {
            "vulnerable": "pack/setrange-grow-undercount-vulnerable",
            "reference": "pack/setrange-grow-undercount-reference-fix",
            "alternate": "pack/setrange-grow-undercount-alternate-fix",
        },
        "build": {"kind": "c-make", "test": "make -C src -j8"},
    },
]


def git(repo: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT / repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def commits_between(repo: str, tip: str, pin: str) -> list[str]:
    return [line for line in git(repo, "rev-list", tip, "--not", pin).splitlines() if line]


def changed_paths(repo: str, commit: str) -> list[str]:
    out = git(repo, "show", "--name-only", "--format=", commit)
    return [line for line in out.splitlines() if line.strip()]


def classify(repo: str, tip: str, pin: str) -> dict:
    """Find the source-injection commit and the metadata commit(s) from git alone."""

    history = commits_between(repo, tip, pin)
    injection = None
    evidence = []
    for commit in history:  # newest first
        paths = changed_paths(repo, commit)
        non_pack = [p for p in paths if not p.startswith("pack/")]
        evidence.append(
            {
                "commit": commit,
                "subject": git(repo, "log", "-1", "--format=%s", commit).strip(),
                "non_pack_paths": non_pack,
                "kind": "source" if non_pack else "metadata",
            }
        )
        if non_pack:
            injection = commit
            break
    if injection is None:
        raise RuntimeError(f"no source commit found on {tip} above {pin}")

    newer = history[: history.index(injection)]
    for commit in newer:
        paths = changed_paths(repo, commit)
        strays = [p for p in paths if not p.startswith("pack/")]
        if strays:
            raise RuntimeError(
                f"commit {commit} is newer than the injection but touches non-pack paths {strays}"
            )
    return {
        "source_injection_sha": injection,
        "metadata_commit_sha": tip,
        "commits_newer_than_injection": newer,
        "classification": evidence,
    }


def diff_names(repo: str, pin: str, branch: str) -> list[tuple[str, str]]:
    out = git(repo, "diff", "--name-status", pin, branch, "--", ".", ":(exclude)pack/")
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0].strip(), parts[-1].strip()))
    return rows


def blob_sha(repo: str, revision: str, path: str) -> str:
    return git(repo, "rev-parse", f"{revision}:{path}").strip()


def export_pack(spec: dict) -> dict:
    repo = spec["repo"]
    pack_dir = OUT / spec["pack"]
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    (pack_dir / "files").mkdir(parents=True)

    tips = {variant: git(repo, "rev-parse", branch).strip() for variant, branch in spec["branches"].items()}
    classified = classify(repo, tips["vulnerable"], spec["pin"])

    files: dict[str, list[dict]] = {}
    for variant, branch in spec["branches"].items():
        rows = diff_names(repo, spec["pin"], branch)
        files[variant] = []
        for status, path in rows:
            record = {"path": path, "status": status}
            if status.startswith("D"):
                record["blob"] = None
            else:
                record["blob"] = blob_sha(repo, branch, path)
                target = pack_dir / "files" / variant / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    subprocess.run(
                        ["git", "-C", str(ROOT / repo), "show", f"{branch}:{path}"],
                        capture_output=True,
                        check=True,
                    ).stdout
                )
            files[variant].append(record)

    # metadata tree (trusted auditor only)
    meta_tree = git(repo, "ls-tree", "-r", "--name-only", tips["vulnerable"], "--", spec["meta_dir"] + "/")
    meta_files = []
    for path in meta_tree.splitlines():
        if not path.strip():
            continue
        target = pack_dir / "meta" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.run(
                ["git", "-C", str(ROOT / repo), "show", f'{tips["vulnerable"]}:{path}'],
                capture_output=True,
                check=True,
            ).stdout
        )
        meta_files.append({"path": path, "blob": blob_sha(repo, tips["vulnerable"], path)})

    manifest_path = pack_dir / "meta" / spec["meta_dir"] / "manifest.json"
    if manifest_path.is_file():
        shutil.copy2(manifest_path, pack_dir / "manifest.json")

    provenance = {
        "pack": spec["pack"],
        "repo": spec["repo"],
        "upstream": spec["upstream"],
        "tag": spec["tag"],
        "pin": spec["pin"],
        "branches": spec["branches"],
        "tips": tips,
        "source_injection_sha": classified["source_injection_sha"],
        "metadata_commit_sha": classified["metadata_commit_sha"],
        "commits_newer_than_injection": classified["commits_newer_than_injection"],
        "classification": classified["classification"],
        "files": files,
        "meta_files": meta_files,
        "build": spec["build"],
        "export_note": (
            "files/<variant>/ contains every non-pack/ file that differs from the upstream "
            "pin, byte-exact, with its git blob SHA. Stage by cloning the pin and overwriting."
        ),
    }
    (pack_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for spec in PACKS:
        provenance = export_pack(spec)
        total = sum(len(items) for items in provenance["files"].values())
        print(
            f"[export] {spec['pack']}: injection={provenance['source_injection_sha'][:9]} "
            f"meta={provenance['metadata_commit_sha'][:9]} files={total} "
            f"meta_files={len(provenance['meta_files'])}"
        )
        index.append(
            {
                "pack": spec["pack"],
                "source_injection_sha": provenance["source_injection_sha"],
                "metadata_commit_sha": provenance["metadata_commit_sha"],
                "files": {variant: [item["path"] for item in items] for variant, items in provenance["files"].items()},
            }
        )
    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
