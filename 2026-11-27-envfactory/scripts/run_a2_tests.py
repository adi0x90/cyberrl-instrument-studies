#!/usr/bin/env python3
"""A2 defect-injection tests for `envcert`.

Three injected defects, each with an expected exact failure:

* **D1** leaked pack metadata inside the untrusted snapshot -> C2 FAIL naming the file.
* **D2** weak (PoC-only) verifier swapped in under the same audit -> C4 FAIL.
* **D3** stale / drifted deployment against a policy_tested stamp -> `verify` exit 2
  naming the drift trigger; a missing deployment must also fail closed.

Plus the clean-pack sweep: all five packs must reach ``pack_integrity`` with zero
false FAILs.

This is a *test fixture* run: the observation file is synthetic, so any
``policy_tested`` stamp produced here is a test artifact, never a certification.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/envfactory")
EVID = ROOT / "evidence"
PACKS = [
    "etcd-revocation-skew",
    "authlib-jwt-dupkey-canonicalization",
    "coremqtt-publish-varint-payloadlen",
    "fastmcp-canonical-authz-alias",
    "redis-setrange-grow-undercount",
]


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "envcert.cli", *args], cwd=ROOT, capture_output=True, text=True)


def audit(pack_dir: Path, out: Path, *, verifier: str = "property", deployment: Path | None = None) -> subprocess.CompletedProcess:
    provenance = json.loads((pack_dir / "provenance.json").read_text())
    args = [
        "audit", str(pack_dir),
        "--source-sha", provenance["source_injection_sha"],
        "--meta-sha", provenance["metadata_commit_sha"],
        "--out", str(out),
        "--a1-evidence", str(EVID / "a1" / "a1_verdicts.jsonl"),
        "--verifier", verifier,
        "--observations", str(EVID / "a2" / "observations.jsonl"),
    ]
    if deployment:
        args += ["--deployment", str(deployment)]
    return run(args)


def main() -> int:
    results: dict = {"clean": [], "defects": {}, "ok": True}
    base_deployment = json.loads((EVID / "a2" / "deployment.json").read_text())
    out_dir = EVID / "a2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- clean packs ---------------------------------------------------
    for pack in PACKS:
        pack_dir = EVID / "packs" / pack
        out = out_dir / f"audit-{pack}.json"
        pack_deployment = dict(base_deployment)
        prov = json.loads((pack_dir / "provenance.json").read_text())
        pack_deployment["source_sha"] = prov["source_injection_sha"]
        pack_deployment["meta_sha"] = prov["metadata_commit_sha"]
        dep_path = Path("/tmp") / f"deploy-{pack}.json"
        dep_path.write_text(json.dumps(pack_deployment, indent=2) + "\n")
        proc = audit(pack_dir, out, deployment=dep_path)
        doc = json.loads(out.read_text()) if out.is_file() else {}
        failing = [name for name, item in doc.get("checks", {}).items() if item["status"] == "FAIL"]
        results["clean"].append(
            {
                "pack": pack,
                "exit": proc.returncode,
                "integrity_pass": doc.get("integrity_pass"),
                "failing": failing,
                "checks": {name: item["status"] for name, item in doc.get("checks", {}).items()},
            }
        )
        if doc.get("integrity_pass") is not True:
            results["ok"] = False

    # ---- D1: leaked metadata in the untrusted snapshot ------------------
    d1_dir = Path("/tmp/d1-leak")
    if d1_dir.exists():
        shutil.rmtree(d1_dir)
    shutil.copytree(EVID / "packs" / "etcd-revocation-skew", d1_dir)
    leaked = d1_dir / "files" / "vulnerable" / "pack" / "etcd-revocation-skew" / "manifest.json"
    leaked.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d1_dir / "meta" / "pack" / "etcd-revocation-skew" / "manifest.json", leaked)
    d1_out = out_dir / "audit-d1.json"
    d1_proc = audit(d1_dir, d1_out, deployment=Path("/tmp/deploy-etcd-revocation-skew.json"))
    d1_doc = json.loads(d1_out.read_text())
    c2 = d1_doc["checks"]["C2"]
    results["defects"]["D1"] = {
        "exit": d1_proc.returncode,
        "C2_status": c2["status"],
        "names_exact_file": "pack/etcd-revocation-skew/manifest.json" in json.dumps(c2),
        "detail": c2["detail"][:300],
        "caught": c2["status"] == "FAIL" and d1_proc.returncode == 2,
    }
    results["ok"] = results["ok"] and results["defects"]["D1"]["caught"]

    # ---- D2: weak verifier under the same audit ------------------------
    d2_out = out_dir / "audit-d2.json"
    d2_proc = audit(EVID / "packs" / "etcd-revocation-skew", d2_out, verifier="weak", deployment=Path("/tmp/deploy-etcd-revocation-skew.json"))
    d2_doc = json.loads(d2_out.read_text())
    c4 = d2_doc["checks"]["C4"]
    results["defects"]["D2"] = {
        "exit": d2_proc.returncode,
        "C4_status": c4["status"],
        "kill_rate": c4.get("kill_rate"),
        "caught": c4["status"] == "FAIL",
        "detail": c4["detail"][:200],
    }
    results["ok"] = results["ok"] and results["defects"]["D2"]["caught"]

    # ---- D3: stale / drifted deployment --------------------------------
    stamp_path = out_dir / "stamp-verify-fixture.json"
    stamp_proc = run(
        [
            "stamp", str(EVID / "packs" / "etcd-revocation-skew"),
            "--audit", str(out_dir / "audit-etcd-revocation-skew.json"),
            "--policy-eval", str(EVID / "a2" / "observations.jsonl"),
            "--min-audited", "30",
            "--deployment", "/tmp/deploy-etcd-revocation-skew.json",
            "--out", str(stamp_path),
        ]
    )
    stamp_doc = json.loads(stamp_path.read_text())

    drifted = json.loads(Path("/tmp/deploy-etcd-revocation-skew.json").read_text())
    drifted["scaffold"] = "multi-turn-tool-loop"
    drifted["attempt_budget"] = 32
    drifted_path = out_dir / "deployment-drifted.json"
    drifted_path.write_text(json.dumps(drifted, indent=2) + "\n")

    hi_pass = []
    for index in range(40):
        hi_pass.append({"rollout_id": f"h{index}", "admitted": True, "holds": index % 10 != 0, "stage": "rollout" if index < 32 else "audit"})
    hi_pass_path = out_dir / "observations-high-pass.jsonl"
    hi_pass_path.write_text("\n".join(json.dumps(row) for row in hi_pass) + "\n")

    verify_proc = run(
        ["verify", "--stamp", str(stamp_path), "--deployment", str(drifted_path), "--observations", str(hi_pass_path)]
    )
    missing_deploy = run(["verify", "--stamp", str(stamp_path)])
    results["defects"]["D3"] = {
        "stamp_level": stamp_doc["level"],
        "verify_exit": verify_proc.returncode,
        "cites_drift_trigger": "pass_rate_jump" in verify_proc.stderr,
        "cites_deployment_drift": "deployment drifted" in verify_proc.stderr,
        "missing_deployment_exit": missing_deploy.returncode,
        "stderr": verify_proc.stderr.strip().splitlines()[:4],
        "caught": verify_proc.returncode == 2 and "pass_rate_jump" in verify_proc.stderr and missing_deploy.returncode == 2,
    }
    results["ok"] = results["ok"] and results["defects"]["D3"]["caught"]
    results["stamp_fixture_note"] = (
        "stamp produced from a synthetic observation file; it exists to exercise verify(), "
        "not to certify a policy"
    )

    out = out_dir / "a2_tests.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: results["defects"][key].get("caught") for key in results["defects"]}, indent=2))
    print(f"clean_packs_ok={all(row['integrity_pass'] for row in results['clean'])} ok={results['ok']}")
    print(f"wrote {out}")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
