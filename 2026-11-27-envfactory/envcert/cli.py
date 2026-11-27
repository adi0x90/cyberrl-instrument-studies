"""envcert — two-level environment certification for pinned cyber-RL packs.

CLI contract (docs/SPEC.md §2 A2)::

    envcert audit  <pack-dir> --source-sha S --meta-sha M --out audit.json [--profile strict]
    envcert stamp  <pack-dir> --audit audit.json [--policy-eval eval.jsonl --min-audited 30] --out stamp.json
    envcert verify --stamp stamp.json --deployment deploy.json --observations obs.jsonl

The two-level rule is the point of the tool:

* ``pack_integrity``  — the *artifact* is what it claims to be (pins, no bundle leak,
  launcher-level egress containment, reset equivalence, disclosure plumbing).
* ``policy_tested``   — the *policy* has been tested against that artifact, by real
  generated candidates from a named (model, scaffold, prompt, decoding, budget),
  with at least ``min_audited`` audited accepts.

Passing the integrity checks never yields a training-ready stamp.  A stamp that
cannot be justified is not issued: the tool returns ``NOT_CERTIFIED`` and exits 2,
which is also the fail-closed behaviour for a stale or drifted stamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HOLDS = "HOLDS"
VIOLATED = "VIOLATED"
UNKNOWN = "UNKNOWN"

PASS = "PASS"
FAIL = "FAIL"
NA = "not_applicable"

EXIT_OK = 0
EXIT_FAIL = 2

DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

HOST_PATH_MARKERS = ("/Users/", "/home/", "/root/", "C:\\Users\\")
DIFF_MARKERS = ("<<<<<<<", ">>>>>>>", "=======\n")

REQUIRED_DEPLOYMENT_FIELDS = (
    "image_digest",
    "source_sha",
    "meta_sha",
    "model_id",
    "scaffold",
    "prompt_sha",
    "decoding",
    "attempt_budget",
    "seed",
)


# ---------------------------------------------------------------------------
# canonical json / hashing
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    else:
        data = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


class Audit:
    def __init__(self, pack_dir: Path, profile: str = "standard") -> None:
        self.pack_dir = Path(pack_dir)
        self.profile = profile
        self.checks: dict[str, dict[str, Any]] = {}
        self.provenance = read_json(self.pack_dir / "provenance.json")

    # -- helpers ---------------------------------------------------------
    def record(self, check: str, status: str, detail: str, **extra: Any) -> None:
        self.checks[check] = {"status": status, "detail": detail, **extra}

    def result(self) -> dict[str, Any]:
        integrity_order = ["C1", "C2", "C3", "C6", "C8", "C9", "C10"]
        policy_order = ["C4", "C5"]
        integrity = [self.checks.get(name, {"status": NA, "detail": "check not run"}) for name in integrity_order]
        policy = [self.checks.get(name, {"status": NA, "detail": "check not run"}) for name in policy_order]
        return {
            "tool": "envcert",
            "tool_version": "1.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pack": self.provenance["pack"],
            "profile": self.profile,
            "provenance_sha256": sha256_of(self.provenance),
            "spec_sha256": _spec_sha(),
            "checks": self.checks,
            "integrity_checks": integrity,
            "policy_checks": policy,
            "integrity_pass": all(item["status"] in {PASS, NA} for item in integrity),
            "policy_pass": all(item["status"] in {PASS, NA} for item in policy),
        }

    # -- C1 --------------------------------------------------------------
    def check_digests(self, deployment: dict | None) -> None:
        problems: list[str] = []
        provenance = self.provenance
        if not SHA_RE.match(str(provenance.get("pin", ""))):
            problems.append("pin is not a 40-hex sha")
        if not SHA_RE.match(str(provenance.get("source_injection_sha", ""))):
            problems.append("source_injection_sha missing/invalid")
        if not SHA_RE.match(str(provenance.get("metadata_commit_sha", ""))):
            problems.append("metadata_commit_sha missing/invalid")
        if deployment is not None:
            for field in ("image_digest", "source_sha", "meta_sha", "seed"):
                if not deployment.get(field):
                    problems.append(f"deployment.{field} not declared")
            image = str(deployment.get("image_digest", ""))
            if image and not DIGEST_RE.match(image):
                problems.append("deployment.image_digest is not repo@sha256:<64 hex>")
            if deployment.get("source_sha") and deployment["source_sha"] != provenance["source_injection_sha"]:
                problems.append(
                    "deployment.source_sha != provenance.source_injection_sha "
                    f"({deployment['source_sha'][:12]} != {provenance['source_injection_sha'][:12]})"
                )
            if deployment.get("meta_sha") and deployment["meta_sha"] != provenance["metadata_commit_sha"]:
                problems.append("deployment.meta_sha != provenance.metadata_commit_sha")
        if problems:
            self.record("C1", FAIL, "; ".join(problems), problems=problems)
        else:
            self.record(
                "C1",
                PASS,
                "pin + dual commit shas pinned"
                + ("; image digest and seed pinned" if deployment else "; deployment not supplied"),
            )

    # -- C2 --------------------------------------------------------------
    def check_bundle_leak(self) -> None:
        untrusted = self.pack_dir / "files" / "vulnerable"
        leaks: list[str] = []
        scanned = 0
        if not untrusted.is_dir():
            self.record("C2", FAIL, "no untrusted snapshot to scan (files/vulnerable missing)")
            return
        for path in sorted(untrusted.rglob("*")):
            if not path.is_file():
                continue
            scanned += 1
            relative = path.relative_to(untrusted).as_posix()
            if relative.startswith("pack/") or "/pack/" in relative:
                leaks.append(f"metadata path present in untrusted tree: {relative}")
            if ".git" in path.parts:
                leaks.append(f"git metadata in untrusted tree: {relative}")
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            for marker in HOST_PATH_MARKERS:
                if marker in text:
                    leaks.append(f"build-host path {marker!r} in {relative}")
            for marker in DIFF_MARKERS:
                if marker in text:
                    leaks.append(f"unresolved diff marker in {relative}")
        # declared metadata files must be absent from the untrusted mount
        for item in self.provenance.get("meta_files", []):
            candidate = untrusted / item["path"]
            if candidate.exists():
                leaks.append(f"metadata file visible to agent: {item['path']}")
        if leaks:
            self.record("C2", FAIL, "; ".join(sorted(set(leaks))[:8]), leaks=sorted(set(leaks)))
        else:
            self.record("C2", PASS, f"scanned {scanned} file(s); no metadata, git, host-path or diff leaks")

    # -- C3 --------------------------------------------------------------
    def check_egress(self, image: str | None, canary_port: int = 18080) -> None:
        """Launcher-path egress containment.

        Runs through ``docker run`` with the deployment's network policy, not a
        bespoke probe process.  A canary listener is proven alive by a positive
        control on the default bridge; the negative probe runs on an ``--internal``
        network and must fail.  An independent structural observation (no default
        route in the negative container) explains the denial, so a dead canary
        cannot masquerade as containment.
        """

        if not image:
            self.record("C3", NA, "no deployment image supplied; egress containment not exercised")
            return
        try:
            if not _docker_ok():
                self.record("C3", NA, "docker unavailable on this host")
                return
            internal = "envcert-internal"
            _run(["docker", "network", "rm", internal], check=False)
            _run(["docker", "network", "create", "--internal", internal], check=True)
            # The host firewall (ufw) defaults to DROP on INPUT and knows nothing
            # about docker bridge traffic, so a container->host canary probe would
            # be denied by the firewall rather than by the launch network.  Open a
            # scoped rule for the probe, prove the canary is alive *through* it, and
            # remove it afterwards.  The rule is recorded in the evidence so the
            # containment claim is never read as "no rule existed".
            subnet = _docker_bridge_subnet()
            firewall_note = None
            if subnet and _ufw_active():
                added = _run(["ufw", "allow", "from", subnet, "to", "any", "port", str(canary_port), "proto", "tcp"], check=False)
                firewall_note = {"rule": f"allow from {subnet} to any port {canary_port} tcp", "added": added.returncode == 0}
            canary = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(canary_port), "--bind", "0.0.0.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(1.5)
                gateway = _docker_gateway()
                probe = (
                    "import socket,sys\n"
                    f"s=socket.socket(); s.settimeout(4)\n"
                    f"try:\n s.connect(('{gateway}',{canary_port})); print('REACHED')\n"
                    "except Exception as e:\n print('DENIED', type(e).__name__)\n"
                )
                positive = _run(
                    ["docker", "run", "--rm", "--network", "bridge", image, "python", "-c", probe],
                    check=False,
                )
                negative = _run(
                    ["docker", "run", "--rm", "--network", internal, image, "python", "-c", probe],
                    check=False,
                )
                routes = _run(
                    ["docker", "run", "--rm", "--network", internal, image, "sh", "-c", "ip route || true"],
                    check=False,
                )
                mounts = _run(
                    [
                        "docker", "run", "--rm", "-v", f"{self.pack_dir}:/workspace:ro",
                        image, "sh", "-c", "echo x > /workspace/.envcert_probe || echo READONLY",
                    ],
                    check=False,
                )
            finally:
                if firewall_note and firewall_note.get("added"):
                    _run(["ufw", "delete", "allow", "from", subnet, "to", "any", "port", str(canary_port), "proto", "tcp"], check=False)
                canary.terminate()
                try:
                    canary.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    canary.kill()

            positive_ok = "REACHED" in positive.stdout
            negative_denied = "DENIED" in negative.stdout
            route_isolated = "default" not in routes.stdout
            read_only = "READONLY" in mounts.stdout
            verdicts = {
                "firewall_rule": firewall_note,
                "canary_alive_positive_control": positive_ok,
                "negative_probe_denied": negative_denied,
                "no_default_route": route_isolated,
                "workspace_mount_read_only": read_only,
            }
            if all(verdicts.values()):
                self.record(
                    "C3",
                    PASS,
                    "canary reachable from bridge (alive), denied from --internal launch network, "
                    "no default route, workspace mount read-only",
                    evidence=verdicts,
                )
            else:
                self.record(
                    "C3",
                    FAIL,
                    "launcher-path containment not established",
                    evidence=verdicts,
                    stdout={"positive": positive.stdout[-400:], "negative": negative.stdout[-400:]},
                )
        except Exception as exc:  # noqa: BLE001 - a broken probe must not look like a pass
            self.record("C3", FAIL, f"egress probe errored: {exc}")

    # -- C4/C5 -----------------------------------------------------------
    def check_verifier(self, a1_evidence: Path | None, arm: str = "property") -> None:
        if a1_evidence is None or not Path(a1_evidence).is_file():
            self.record("C4", NA, "no A1 evidence supplied")
            self.record("C5", NA, "no A1 evidence supplied")
            return
        rows = read_jsonl(Path(a1_evidence))
        if arm not in rows[0]:
            self.record("C4", NA, f"arm {arm!r} absent from evidence")
            self.record("C5", NA, f"arm {arm!r} absent from evidence")
            return

        def count(truth: str, verdict: str) -> int:
            return sum(1 for row in rows if row["truth"] == truth and row[arm] == verdict)

        bad = sum(1 for row in rows if row["truth"] == "bad")
        valid = sum(1 for row in rows if row["truth"] == "valid")
        kills = count("bad", VIOLATED)
        false_rejects = count("valid", VIOLATED)
        unknown = sum(1 for row in rows if row[arm] == UNKNOWN)
        total = bad + valid
        kill_rate = kills / bad if bad else 0.0
        coverage = 1.0 - (unknown / total) if total else 0.0
        matrix = {
            truth: {verdict: count(truth, verdict) for verdict in (HOLDS, VIOLATED, UNKNOWN)}
            for truth in ("bad", "valid")
        }
        detail = {
            "arm": arm,
            "kill_rate": round(kill_rate, 6),
            "false_rejects": false_rejects,
            "decisive_coverage": round(coverage, 6),
            "matrix": matrix,
            "rows": total,
            "source_sha256": sha256_of(Path(a1_evidence).read_bytes()),
        }
        if kill_rate >= 0.90 and false_rejects == 0:
            self.record("C4", PASS, f"kill_rate={kill_rate:.3f} false_rejects={false_rejects}", **detail)
        else:
            self.record("C4", FAIL, f"kill_rate={kill_rate:.3f} (needs >=0.90), false_rejects={false_rejects}", **detail)
        if coverage >= 0.95 and false_rejects == 0:
            self.record("C5", PASS, f"decisive_coverage={coverage:.3f} false_rejects=0", **detail)
        else:
            self.record(
                "C5",
                FAIL,
                f"decisive_coverage={coverage:.3f} (needs >=0.95) false_rejects={false_rejects}",
                **detail,
            )

    # -- C6 --------------------------------------------------------------
    def check_reset(self, reset_evidence: Path | None) -> None:
        if reset_evidence is None or not Path(reset_evidence).is_file():
            self.record("C6", NA, "no reset-benchmark evidence supplied for this target")
            return
        payload = read_json(Path(reset_evidence))
        rows = payload if isinstance(payload, list) else payload.get("targets", [])
        target = self.provenance["pack"]
        match = [row for row in rows if row.get("pack") == target or row.get("target") == target]
        if not match:
            self.record("C6", NA, f"reset benchmark contains no entry for {target}")
            return
        row = match[0]
        if row.get("state_equivalence") is not True:
            self.record("C6", FAIL, "state-equivalence gate not passed for this target", detail=row)
            return
        self.record(
            "C6",
            PASS,
            f"state-equivalence passed; admitted reset strategies: {row.get('admitted_strategies')}",
            detail=row,
        )

    # -- C8/C9/C10 -------------------------------------------------------
    def check_disclosure(self, deployment: dict | None) -> None:
        if deployment is None:
            for name in ("C8", "C9", "C10"):
                self.record(name, NA, "no deployment metadata supplied")
            return
        missing = [field for field in REQUIRED_DEPLOYMENT_FIELDS if not deployment.get(field)]
        if missing:
            self.record("C8", FAIL, f"deployment omits disclosure fields: {missing}", missing=missing)
        else:
            self.record("C8", PASS, "budget, scaffold, prompt, decoding, seed, model disclosed")

    def check_log_plumbing(self, observations: Path | None) -> None:
        if observations is None or not Path(observations).is_file():
            self.record("C9", NA, "no observations supplied")
            self.record("C10", NA, "no observations supplied")
            return
        rows = read_jsonl(Path(observations))
        if not rows:
            self.record("C9", FAIL, "observations file is empty")
            self.record("C10", FAIL, "observations file is empty")
            return
        admitted = [row for row in rows if row.get("admitted")]
        holds = [row for row in admitted if row.get("holds")]
        p_admitted = len(admitted) / len(rows)
        p_holds_given_admitted = len(holds) / len(admitted) if admitted else None
        missing_fields = [row for row in rows if "admitted" not in row]
        if missing_fields:
            self.record("C9", FAIL, f"{len(missing_fields)} observation(s) lack the admitted flag")
        else:
            self.record(
                "C9",
                PASS,
                f"P(admitted)={p_admitted:.3f} ({len(admitted)}/{len(rows)}); "
                f"P(holds|admitted)={p_holds_given_admitted if p_holds_given_admitted is None else round(p_holds_given_admitted, 3)}",
                p_admitted=round(p_admitted, 6),
                p_holds_given_admitted=None if p_holds_given_admitted is None else round(p_holds_given_admitted, 6),
                n=len(rows),
            )

        stages = sorted({str(row.get("stage", "")) for row in rows})
        # cross-tab: stage x admission outcome.  A cell with zero rows is missing
        # coverage, and must be declared rather than inferred as "not a problem".
        cells = [f"{stage}|{outcome}" for stage in stages for outcome in ("admitted", "refused")]
        observed = {f"{row.get('stage','')}|{'admitted' if row.get('admitted') else 'refused'}" for row in rows}
        empty_cells = [cell for cell in cells if cell not in observed]
        declared = deployment_empty_cells()
        undeclared = [cell for cell in empty_cells if cell not in declared]
        if len(stages) < 2:
            self.record("C10", NA, f"single-stage log ({stages}); no cross-tab to build")
        elif undeclared:
            self.record("C10", FAIL, f"empty cross-tab cells not declared: {undeclared}")
        else:
            self.record(
                "C10",
                PASS,
                f"stages observed: {stages}; empty cells {empty_cells or 'none'} all declared",
                empty_cells=empty_cells,
            )

    # -- orchestrate -----------------------------------------------------
    def run(
        self,
        deployment: dict | None = None,
        a1_evidence: Path | None = None,
        verifier_arm: str = "property",
        reset_evidence: Path | None = None,
        observations: Path | None = None,
    ) -> dict[str, Any]:
        self.check_digests(deployment)
        self.check_bundle_leak()
        self.check_egress(deployment.get("image_digest") if deployment else None)
        self.check_verifier(a1_evidence, verifier_arm)
        self.check_reset(reset_evidence)
        self.check_disclosure(deployment)
        self.check_log_plumbing(observations)
        return self.result()


def deployment_empty_cells() -> list[str]:
    value = os.environ.get("ENVCERT_DECLARED_EMPTY_CELLS", "")
    return [item for item in value.split(",") if item]


def stage_cells_undeclared(stages: list[str], declared: list[str]) -> bool:
    return any(stage not in declared for stage in stages)


def _spec_sha() -> str:
    path = Path(__file__).resolve().parent.parent / "docs" / "SPEC.sha256"
    try:
        return path.read_text(encoding="utf-8").split()[0]
    except OSError:
        return ""


def _run(argv: list[str], check: bool) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=check, timeout=180)


def _docker_ok() -> bool:
    try:
        return _run(["docker", "info", "--format", "{{.ServerVersion}}"], check=False).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _ufw_active() -> bool:
    proc = _run(["ufw", "status"], check=False)
    return "Status: active" in proc.stdout


def _docker_bridge_subnet() -> str | None:
    proc = _run(["docker", "network", "inspect", "bridge", "--format", "{{range .IPAM.Config}}{{.Subnet}}{{end}}"], check=False)
    return proc.stdout.strip() or None


def _docker_gateway() -> str:
    out = _run(["docker", "network", "inspect", "bridge", "--format", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"], check=False)
    return out.stdout.strip() or "172.17.0.1"


# ---------------------------------------------------------------------------
# stamp
# ---------------------------------------------------------------------------

POLICY_REQUIRED = (
    "model_id",
    "scaffold",
    "prompt_sha",
    "decoding",
    "attempt_budget",
)


def stamp(
    audit: dict[str, Any],
    deployment: dict | None,
    observations: Path | None,
    min_audited: int,
) -> tuple[dict[str, Any], int]:
    checks = audit["checks"]
    policy_checks = audit.get("policy_checks", [])
    integrity_only = audit.get("integrity_pass", False)

    if not integrity_only:
        failing = [name for name, item in checks.items() if item["status"] == FAIL]
        return _not_certified(audit, f"integrity checks failed: {failing}"), EXIT_FAIL

    policy_tested = (
        all(item["status"] == PASS for item in policy_checks)
        and deployment is not None
        and observations is not None
    )
    audited = 0
    if observations is not None and Path(observations).is_file():
        audited = sum(1 for row in read_jsonl(Path(observations)) if row.get("admitted"))

    missing_policy = [field for field in POLICY_REQUIRED if not (deployment or {}).get(field)]
    if policy_tested and not missing_policy and audited >= min_audited:
        level = "policy_tested"
    else:
        level = "pack_integrity"

    body: dict[str, Any] = {
        "stamp_version": "1.0",
        "level": level,
        "pack": audit["pack"],
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expires": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30 * 86400)),
        "provenance_sha256": audit["provenance_sha256"],
        "audit_sha256": sha256_of(audit),
        "spec_sha256": audit.get("spec_sha256", ""),
        "integrity_checks_sha256": sha256_of(audit.get("integrity_checks")),
        "policy_checks_sha256": sha256_of(policy_checks),
        "min_audited": min_audited,
        "audited_accepts": audited,
        "deployment": {field: (deployment or {}).get(field) for field in REQUIRED_DEPLOYMENT_FIELDS},
        "certified_pass_band": (deployment or {}).get("certified_pass_band"),
        "drift_triggers": (deployment or {}).get("drift_triggers") or {"pass_rate_jump": 0.15},
    }
    if level == "pack_integrity":
        body["refusal_reason"] = (
            "policy_tested requires real generated candidates under C4/C5 with "
            f"audited_accepts>={min_audited} (got {audited}) and a named "
            f"(model, scaffold, prompt, decoding, budget); missing={missing_policy or 'none'}"
        )
    body["stamp_sha256"] = sha256_of(body)
    return body, EXIT_OK


def _not_certified(audit: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "stamp_version": "1.0",
        "level": "NOT_CERTIFIED",
        "pack": audit.get("pack"),
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "refusal_reason": reason,
        "audit_sha256": sha256_of(audit),
    }


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify(stamp_doc: dict[str, Any], deployment: dict | None, observations: Path | None) -> tuple[list[str], list[str]]:
    """Return (errors, notes).  Any error means exit 2 (fail closed)."""

    errors: list[str] = []
    notes: list[str] = []

    recorded = stamp_doc.get("stamp_sha256")
    body = {key: value for key, value in stamp_doc.items() if key != "stamp_sha256"}
    if recorded != sha256_of(body):
        errors.append("stamp body hash mismatch (stamp edited after issue)")

    if stamp_doc.get("level") not in {"pack_integrity", "policy_tested"}:
        errors.append(f"stamp level {stamp_doc.get('level')!r} is not deployable")

    expires = stamp_doc.get("expires")
    if not expires:
        errors.append("stamp has no expiry (fail closed)")
    else:
        try:
            if time.strptime(expires, "%Y-%m-%dT%H:%M:%SZ") < time.gmtime():
                errors.append(f"stamp expired at {expires}")
        except ValueError:
            errors.append(f"stamp expiry unparseable: {expires!r}")

    if deployment is None:
        errors.append("no deployment metadata supplied (fail closed)")
        return errors, notes

    stamped = stamp_doc.get("deployment") or {}
    mismatches = []
    for field in ("image_digest", "source_sha", "meta_sha", "model_id", "scaffold", "prompt_sha", "decoding", "attempt_budget"):
        if stamped.get(field) != deployment.get(field):
            mismatches.append(f"{field}: stamp={stamped.get(field)!r} deployment={deployment.get(field)!r}")
    if mismatches:
        errors.append("deployment drifted from certified configuration: " + "; ".join(mismatches))

    triggers = stamp_doc.get("drift_triggers") or {}
    jump = float(triggers.get("pass_rate_jump", 0.15))
    if observations is not None and Path(observations).is_file():
        rows = read_jsonl(Path(observations))
        admitted = [row for row in rows if row.get("admitted")]
        if admitted:
            pass_rate = sum(1 for row in admitted if row.get("holds")) / len(admitted)
            band = stamp_doc.get("certified_pass_band")
            if isinstance(band, dict) and band.get("pass_rate") is not None:
                delta = abs(pass_rate - float(band["pass_rate"]))
                if delta >= jump:
                    errors.append(
                        f"drift trigger fired: pass_rate_jump={delta:.3f} >= {jump} "
                        f"(certified {band['pass_rate']}, observed {pass_rate:.3f})"
                    )
                else:
                    notes.append(f"pass-rate delta {delta:.3f} below trigger {jump}")
            else:
                notes.append("stamp carries no certified_pass_band; drift check skipped")
    if stamp_doc.get("level") == "pack_integrity":
        notes.append("pack_integrity is not training-ready; policy_tested required before deployment")
    return errors, notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="envcert")
    sub = parser.add_subparsers(dest="command", required=True)

    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("pack_dir")
    audit_parser.add_argument("--source-sha", default=None)
    audit_parser.add_argument("--meta-sha", default=None)
    audit_parser.add_argument("--out", required=True)
    audit_parser.add_argument("--profile", default="standard", choices=["standard", "strict"])
    audit_parser.add_argument("--deployment", default=None)
    audit_parser.add_argument("--a1-evidence", default="evidence/a1/a1_verdicts.jsonl")
    audit_parser.add_argument("--verifier", default="property", choices=["property", "static", "weak"])
    audit_parser.add_argument("--reset-evidence", default=None)
    audit_parser.add_argument("--observations", default=None)

    stamp_parser = sub.add_parser("stamp")
    stamp_parser.add_argument("pack_dir")
    stamp_parser.add_argument("--audit", required=True)
    stamp_parser.add_argument("--policy-eval", default=None)
    stamp_parser.add_argument("--min-audited", type=int, default=30)
    stamp_parser.add_argument("--deployment", default=None)
    stamp_parser.add_argument("--out", required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--stamp", required=True)
    verify_parser.add_argument("--deployment", default=None)
    verify_parser.add_argument("--observations", default=None)

    args = parser.parse_args(argv)

    if args.command == "audit":
        pack_dir = Path(args.pack_dir)
        run = Audit(pack_dir, args.profile)
        provenance = run.provenance
        if args.source_sha and args.source_sha != provenance["source_injection_sha"]:
            print(f"envcert: --source-sha does not match provenance ({args.source_sha[:12]})", file=sys.stderr)
            return EXIT_FAIL
        if args.meta_sha and args.meta_sha != provenance["metadata_commit_sha"]:
            print(f"envcert: --meta-sha does not match provenance ({args.meta_sha[:12]})", file=sys.stderr)
            return EXIT_FAIL
        deployment = read_json(Path(args.deployment)) if args.deployment else None
        result = run.run(
            deployment=deployment,
            a1_evidence=Path(args.a1_evidence) if args.a1_evidence else None,
            verifier_arm=args.verifier,
            reset_evidence=Path(args.reset_evidence) if args.reset_evidence else None,
            observations=Path(args.observations) if args.observations else None,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        failing = [name for name, item in result["checks"].items() if item["status"] == FAIL]
        print(f"envcert audit: {result['pack']} integrity_pass={result['integrity_pass']} failing={failing}")
        return EXIT_OK if result["integrity_pass"] else EXIT_FAIL

    if args.command == "stamp":
        audit_doc = read_json(Path(args.audit))
        deployment = read_json(Path(args.deployment)) if args.deployment else None
        doc, code = stamp(audit_doc, deployment, Path(args.policy_eval) if args.policy_eval else None, args.min_audited)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"envcert stamp: level={doc['level']} reason={doc.get('refusal_reason','')}")
        return code

    if args.command == "verify":
        doc = read_json(Path(args.stamp))
        deployment = read_json(Path(args.deployment)) if args.deployment else None
        errors, notes = verify(doc, deployment, Path(args.observations) if args.observations else None)
        for note in notes:
            print(f"note: {note}")
        for error in errors:
            print(f"DRIFT: {error}", file=sys.stderr)
        return EXIT_FAIL if errors else EXIT_OK

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
