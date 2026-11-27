"""A1 native-pack case — held-out valid repair on a real repository.

The expression control uses hand-written implements; this case uses the real
authlib RFC 7523 grant (`authlib-jwt-dupkey-canonicalization`, pinned v1.9.10 +
injection commit).

Cheap Tier-1 (what the pack ships): the three PoC scenarios (``main``, ``aud``,
``exp``) — finite fixtures.

Property-hardened Tier-1: generated assertions whose duplicated member varies in
name (``scope`` / ``aud`` / ``exp`` / ``sub``), value, and ordering, checked
against the written contract:

* the authorization decision may never act on a *different* view of the
  assertion than the one that was validated;
* therefore a duplicate that changes the effective ``scope`` / ``aud`` / ``exp``
  / ``sub`` must either be refused, or have the first-occurrence view enforced.

Candidates are applied by exact textual patches to the vulnerable file, so the
repair is a patch a human could review, not a monkeypatch.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/root/work/authlib")
PACK = Path("/root/envfactory/evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack")
VULNERABLE_FILE = "authlib/oauth2/rfc7523/jwt_bearer.py"

CANONICAL_BLOCK = """        # claims are validated against their canonical, duplicate-free
        # projection, so validation sees a deterministic form of the
        # assertion regardless of duplicate JSON member names
        canonical = self.canonical_assertion_claims(self.extract_raw_payload(assertion))
        self.verify_claims(canonical)
        self._verify_asserted_scope(canonical, client)
        self.canonical_claims = canonical
        return token.claims"""

BASE_SEED = 20261127


def _insert_strict_parser(source: str) -> str:
    helper = '''    @classmethod
    def _strict_assertion_claims(cls, payload: bytes) -> dict:
        """Reject duplicate member names instead of picking a winner.

        The assertion is attacker-controlled input; two views of the same claim
        are not a formatting detail, they are two different claims.  Fail closed.
        """
        def _pairs(pairs):
            seen = set()
            for name, _value in pairs:
                if name in seen:
                    raise InvalidGrantError(
                        description="Assertion carries duplicate member names"
                    )
                seen.add(name)
            return dict(pairs)

        try:
            return json.loads(payload, object_pairs_hook=_pairs)
        except InvalidGrantError:
            raise
        except ValueError:
            raise InvalidGrantError(description="Invalid JWT assertion") from None

'''
    marker = "    def process_assertion_claims(self, assertion):"
    return source.replace(marker, helper + marker, 1)


def _reject_duplicates_for(source: str, members: tuple[str, ...]) -> str:
    helper = '''    @classmethod
    def _strict_assertion_claims(cls, payload: bytes) -> dict:
        """Reject duplicate member names for a fixed subset of claims."""
        watched = set(%r)

        def _pairs(pairs):
            seen = set()
            for name, _value in pairs:
                if name in seen and name in watched:
                    raise InvalidGrantError(
                        description="Assertion carries duplicate member names"
                    )
                seen.add(name)
            return dict(pairs)

        try:
            return json.loads(payload, object_pairs_hook=_pairs)
        except InvalidGrantError:
            raise
        except ValueError:
            raise InvalidGrantError(description="Invalid JWT assertion") from None

''' % (members,)
    marker = "    def process_assertion_claims(self, assertion):"
    return source.replace(marker, helper + marker, 1)


def apply_variant(variant: str, target: Path) -> None:
    source = target.read_text(encoding="utf-8")
    if variant == "vulnerable":
        pass
    elif variant == "heldout_strict":
        source = _insert_strict_parser(source)
        source = source.replace(
            "        canonical = self.canonical_assertion_claims(self.extract_raw_payload(assertion))",
            "        canonical = self._strict_assertion_claims(self.extract_raw_payload(assertion))",
        )
        source = source.replace(
            '        if self.canonical_claims.get("scope") is not None:',
            '        if self.canonical_claims.get("scope") is not None:',
        )
    elif variant == "partial_scope_only":
        source = source.replace(
            '            self.requested_scope = claims.get("scope")',
            '            self.requested_scope = self.canonical_claims.get("scope")',
        )
    elif variant == "partial_sub_raw":
        source = _reject_duplicates_for(source, ("scope", "aud", "exp"))
        source = source.replace(
            "        canonical = self.canonical_assertion_claims(self.extract_raw_payload(assertion))",
            "        canonical = self._strict_assertion_claims(self.extract_raw_payload(assertion))",
        )
        # identity still read from the raw, last-wins view
    elif variant == "shortcut_admin":
        source = source.replace(
            '            self.requested_scope = claims.get("scope")',
            '            self.requested_scope = claims.get("scope")\n'
            '            if "admin" in (self.requested_scope or ""):\n'
            '                raise InvalidScopeError()',
        )
    else:
        raise SystemExit(f"unknown variant {variant}")
    target.write_text(source, encoding="utf-8")


def stage_variant(variant: str, workspace: Path) -> Path:
    target = workspace / f"authlib-{variant}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(REPO, target, ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc"))
    apply_variant(variant, target / VULNERABLE_FILE)
    return target


# ---------------------------------------------------------------------------
# cheap tier: the pack's own scenarios
# ---------------------------------------------------------------------------


def _interpreter() -> str:
    """The editable-install venv lives in the base checkout; each variant is
    staged as a source tree and imported ahead of site-packages by harness.py."""

    return str(REPO / ".venv/bin/python")


def cheap_verdict(workspace: Path) -> dict:
    scenarios = {}
    for scenario in ("main", "aud", "exp"):
        proc = subprocess.run(
            [_interpreter(), "pack/poc/poc.py", "--scenario", scenario],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=180,
        )
        scenarios[scenario] = {"exit": proc.returncode, "passed": proc.returncode == 0}
    return {"scenarios": scenarios, "passed": all(item["passed"] for item in scenarios.values())}


# ---------------------------------------------------------------------------
# property tier: generated duplicate-member assertions
# ---------------------------------------------------------------------------


PROPERTY_SCRIPT = r"""
import json, random, sys, time
sys.path.insert(0, sys.argv[1] + "/pack/poc")
import harness
seed = int(sys.argv[2]); trials = int(sys.argv[3])
rng = random.Random(seed); now = int(time.time())
violations = []; decided = 0
for _ in range(trials):
    kind = rng.choice(["scope", "aud", "exp", "sub"])
    if kind == "scope":
        safe = rng.choice(["openid", "openid profile", "openid email"])
        escalated = safe + " " + rng.choice(["admin", "profile admin", "email admin"])
        payload = ('{"iss":"client-id","aud":"https://as.example.com/token",'
                   f'"exp":{now + 3600},"iat":{now},'
                   f'"scope":"{safe}","scope":"{escalated}"}}')
        status, body = harness.run_grant(payload, scope_param="openid")
        if status == 200:
            issued = (harness.token_claims(body).get("scope") or "")
            if not set(issued.split()) <= set(safe.split()):
                decided += 1; violations.append({"kind": kind, "issued": issued, "policy": safe})
    elif kind == "aud":
        payload = ('{"iss":"client-id","aud":"https://as.example.com/token",'
                   f'"exp":{now + 3600},"iat":{now},'
                   '"aud":"https://attacker.example"}')
        status, _ = harness.run_grant(payload); decided += 1
        if status == 200: violations.append({"kind": kind, "status": status})
    elif kind == "exp":
        payload = ('{"iss":"client-id","aud":"https://as.example.com/token",'
                   f'"exp":{now + 3600},"iat":{now},'
                   f'"exp":{now - rng.randrange(60, 7200)}}}')
        status, _ = harness.run_grant(payload); decided += 1
        if status == 200: violations.append({"kind": kind, "status": status})
    else:
        payload = ('{"iss":"client-id","aud":"https://as.example.com/token",'
                   f'"exp":{now + 3600},"iat":{now},'
                   '"sub":"svc-web","sub":"admin"}')
        status, body = harness.run_grant(payload); decided += 1
        if status == 200:
            claims = harness.token_claims(body)
            if claims.get("sub") == "admin" or claims.get("user_id") == "admin":
                violations.append({"kind": kind, "claims": {k: claims.get(k) for k in ("sub", "user_id")}})
print(json.dumps({"decided": decided, "violations": len(violations), "passed": not violations, "examples": violations[:3]}))
"""


def property_verdict(workspace: Path, seed: int, trials: int = 24) -> dict:
    """Run the generated assertions in a fresh interpreter per variant.

    A fresh process is deliberate: authlib, joserfc and the harness all cache
    modules, so in-process re-imports would silently keep the previous variant's
    code and make every candidate look identical.
    """

    proc = subprocess.run(
        [_interpreter(), "-c", PROPERTY_SCRIPT, str(workspace), str(seed), str(trials)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return {"decided": 0, "violations": 0, "passed": False, "error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/envfactory/evidence/a1_native.json")
    parser.add_argument("--workspace", default="/root/work")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    variants = ["vulnerable", "heldout_strict", "partial_scope_only", "partial_sub_raw", "shortcut_admin"]
    truth = {
        "vulnerable": "bad",
        "heldout_strict": "valid",
        "partial_scope_only": "bad",
        "partial_sub_raw": "bad",
        "shortcut_admin": "bad",
    }
    rows = []
    for variant in variants:
        staged = stage_variant(variant, workspace)
        venv_ok = (staged / ".venv").exists() or True
        cheap = cheap_verdict(staged)
        prop = property_verdict(staged, args.seed)
        rows.append(
            {
                "variant": variant,
                "truth": truth[variant],
                "cheap": cheap,
                "property": prop,
                "cheap_verdict": "HOLDS" if cheap["passed"] else "VIOLATED",
                "property_verdict": "HOLDS" if prop["passed"] else "VIOLATED",
            }
        )
        print(f"[native] {variant:18s} truth={truth[variant]:5s} cheap={'HOLDS' if cheap['passed'] else 'VIOLATED':8s} property={'HOLDS' if prop['passed'] else 'VIOLATED'}", flush=True)

    summary = {
        "pack": "authlib-jwt-dupkey-canonicalization",
        "seed": args.seed,
        "variants": rows,
        "cheap": {
            "valid_false_reject": sum(1 for r in rows if r["truth"] == "valid" and r["cheap_verdict"] == "VIOLATED"),
            "bad_kills": sum(1 for r in rows if r["truth"] == "bad" and r["cheap_verdict"] == "VIOLATED"),
            "bad_total": sum(1 for r in rows if r["truth"] == "bad"),
        },
        "property": {
            "valid_false_reject": sum(1 for r in rows if r["truth"] == "valid" and r["property_verdict"] == "VIOLATED"),
            "bad_kills": sum(1 for r in rows if r["truth"] == "bad" and r["property_verdict"] == "VIOLATED"),
            "bad_total": sum(1 for r in rows if r["truth"] == "bad"),
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[native] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
