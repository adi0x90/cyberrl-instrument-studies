"""Verify every claim in results/REPORT.md against raw evidence.

Deliberately *independent* of `analysis/recompute.py`: this file re-derives the
numbers with its own loops, so a bug in the report builder cannot silently
validate itself.  Only the Python standard library is used, and nothing here
touches the network, Docker, or the model.

    python3 verify.py          # from the experiment directory
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
RESULTS = ROOT / "results"

HOLDS, VIOLATED, UNKNOWN = "HOLDS", "VIOLATED", "UNKNOWN"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def median(samples: list[float]) -> float:
    return statistics.median(samples)


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def main() -> int:
    claims = jload(RESULTS / "claims.json")["claims"]
    native_values: dict = {}
    verdicts = jsonl(EVIDENCE / "a1" / "a1_verdicts.jsonl")
    native_path = EVIDENCE / "a1" / "a1_native.json"
    native = jload(native_path) if native_path.is_file() else {}
    loo = jload(EVIDENCE / "a1" / "a1_loo.json") if (EVIDENCE / "a1" / "a1_loo.json").is_file() else {}
    latency = jload(EVIDENCE / "a1" / "a1_latency.json") if (EVIDENCE / "a1" / "a1_latency.json").is_file() else {}
    a2 = jload(EVIDENCE / "a2" / "a2_tests.json") if (EVIDENCE / "a2" / "a2_tests.json").is_file() else {}
    a3 = jload(EVIDENCE / "a3" / "a3_reset.json") if (EVIDENCE / "a3" / "a3_reset.json").is_file() else {}

    def arm_rate(arm: str) -> float:
        bad = [row for row in verdicts if row["truth"] == "bad"]
        return sum(1 for row in bad if row[arm] == VIOLATED) / len(bad) if bad else 0.0

    def ep(selection) -> dict:
        chosen = [row for row in verdicts if selection(row)]
        return {"num": sum(1 for row in chosen if row["property"] == VIOLATED), "den": len(chosen)}

    problems: list[str] = []

    def computed(claim_id: str):
        if claim_id == "a1.rows":
            return len(verdicts)
        if claim_id == "a1.seeds":
            return len({row["seed"] for row in verdicts})
        if claim_id in {"a1.weak.kill_rate", "a1.static.kill_rate", "a1.property.kill_rate"}:
            return round(arm_rate(claim_id.split(".")[1]), 6)
        if claim_id == "a1.property.endpoint.canonical_lookup.rate":
            item = ep(lambda row: row["candidate"].startswith("adversarial["))
            return round(item["num"] / item["den"], 6) if item["den"] else 0.0
        if claim_id == "a1.property.endpoint.lookup_mutants.rate":
            item = ep(lambda row: row["candidate"].startswith("lookup_mutant["))
            return round(item["num"] / item["den"], 6) if item["den"] else 0.0
        if claim_id == "a1.property.endpoint.e2_bad_corpus.rate":
            item = ep(lambda row: row["kind"] == "e2_negative")
            return round(item["num"] / item["den"], 6) if item["den"] else 0.0
        if claim_id == "a1.property.endpoint.e2_positive_controls.rate":
            item = ep(lambda row: row["kind"] == "e2_positive")
            return round(item["num"] / item["den"], 6) if item["den"] else 0.0
        if claim_id == "a1.property_only_kills.count":
            identities = {
                (row["task_id"], row["candidate"])
                for row in verdicts
                if row["truth"] == "bad" and row["static"] != VIOLATED and row["property"] == VIOLATED
            }
            return len(identities)
        if claim_id == "a1.property.survivors.count":
            identities = {
                (row["task_id"], row["candidate"])
                for row in verdicts
                if row["truth"] == "bad" and row["property"] != VIOLATED
            }
            return len(identities)
        if claim_id.startswith("a1.loo.") and claim_id.endswith(".violated"):
            key = claim_id.split(".")[2]
            return loo[key]["violated"]
        if claim_id == "a1.latency.delta_p50_mean":
            return round(statistics.fmean(row["delta_ms"]["p50"] for row in latency["tasks"]), 4)
        if claim_id == "a1.native.cheap.bad_kills":
            return sum(1 for row in native["variants"] if row["truth"] == "bad" and row["cheap_verdict"] == VIOLATED)
        if claim_id == "a1.native.property.bad_kills":
            return sum(1 for row in native["variants"] if row["truth"] == "bad" and row["property_verdict"] == VIOLATED)
        if claim_id == "a1.native.property.valid_false_reject":
            return sum(1 for row in native["variants"] if row["truth"] == "valid" and row["property_verdict"] == VIOLATED)
        if claim_id == "a2.clean_packs":
            return len(a2.get("clean", []))
        if claim_id == "a2.clean_packs_passing":
            return sum(1 for row in a2.get("clean", []) if row.get("integrity_pass") is True)
        if claim_id == "a2.d1":
            defect = a2.get("defects", {}).get("D1", {})
            return defect.get("exit") == 2 and defect.get("C2_status") == "FAIL"
        if claim_id == "a2.d2":
            return a2.get("defects", {}).get("D2", {}).get("C4_status") == "FAIL"
        if claim_id == "a2.d3":
            defect = a2.get("defects", {}).get("D3", {})
            return defect.get("verify_exit") == 2 and defect.get("cites_drift_trigger") is True
        if claim_id.startswith("a3.gate."):
            pack = claim_id[len("a3.gate."):]
            for entry in a3.get("targets", []):
                if entry["pack"] == pack:
                    return entry["gate"]["state_equivalence"]
            return None
        if claim_id.startswith("native.") and claim_id in native_values:
            return native_values[claim_id]
        if claim_id == "a3.not_run.count":
            return len(a3.get("not_run", []))
        if claim_id == "a3.qemu.supported":
            return a3.get("qemu", {}).get("supported")
        if claim_id == "a3.criu.supported":
            return a3.get("criu", {}).get("supported")
        if claim_id.startswith("a3.") and claim_id.endswith(".p50"):
            _, pack, strategy, _ = claim_id.split(".")
            for entry in a3.get("targets", []):
                if entry["pack"] == pack:
                    for row in entry.get("timings", []):
                        if row["strategy"] == strategy:
                            samples = row.get("samples_s") or []
                            return round(percentile(samples, 0.50), 4) if samples else row.get("p50_s")
        return "UNRESOLVED"

    # ---- native-pack arm (ATT-2975): independent recomputation ----
    native_path = RESULTS / "B1_native.json"
    if native_path.is_file():
        native_doc = jload(native_path)
        graded = []
        for name in ("graded_authlib.jsonl", "graded_fastmcp.jsonl"):
            path = EVIDENCE / "bon" / name
            if path.is_file():
                graded.extend(jsonl(path))
        graded.sort(key=lambda row: (row["pack"], row["model"], row["sample"]))
        for n in (1, 4, 8, 16, 32):
            groups = {}
            for row in graded:
                groups.setdefault((row["pack"], row["model"]), []).append(row)
            cells = {"t1p_t2p": 0, "t1p_t2m": 0, "no_t1_pass": 0, "untested": 0}
            for items in groups.values():
                pool = items[:n]
                chosen = next((row for row in pool if row["t1"] == "HOLDS"), None)
                if chosen is not None and chosen.get("t2") == "VIOLATED":
                    cells["t1p_t2m"] += 1
                elif chosen is not None:
                    cells["t1p_t2p"] += 1
                elif all(row["t1"] == "untested" for row in pool):
                    cells["untested"] += 1
                else:
                    cells["no_t1_pass"] += 1
            t1_pass = cells["t1p_t2p"] + cells["t1p_t2m"]
            far = (cells["t1p_t2m"] / t1_pass) if t1_pass else None
            claims.append({"id": f"native.N{n}.t1_pass", "value": t1_pass,
                           "reported_from": "results/B1_native.json"})
            claims.append({"id": f"native.N{n}.cell.t1p_t2m", "value": cells["t1p_t2m"],
                           "reported_from": "results/B1_native.json"})
            claims.append({"id": f"native.N{n}.far", "value": far, "reported_from": "results/B1_native.json"})
        native_values = {
            f"native.N{n}.t1_pass": native_doc["analysis"]["N"][str(n)]["t1_pass"] for n in (1, 4, 8, 16, 32)
        }
        native_values.update({
            f"native.N{n}.cell.t1p_t2m": native_doc["analysis"]["N"][str(n)]["cells"]["t1p_t2m"] for n in (1, 4, 8, 16, 32)
        })
        native_values.update({
            f"native.N{n}.far": native_doc["analysis"]["N"][str(n)]["far_given_t1_pass"] for n in (1, 4, 8, 16, 32)
        })

    print(f"{'claim':50s} {'reported':>14s} {'recomputed':>14s}  status")
    print("-" * 94)
    for claim in claims:
        recomputed = computed(claim["id"])
        reported = claim["value"]
        if recomputed == "UNRESOLVED":
            ok = False
        elif isinstance(recomputed, float) and isinstance(reported, (int, float)):
            ok = abs(round(recomputed, 6) - round(float(reported), 6)) < 1e-9
        else:
            ok = recomputed == reported
        if not ok:
            problems.append(claim["id"])
        print(f"{claim['id']:50s} {str(reported):>14s} {str(recomputed):>14s}  {'PASS' if ok else 'FAIL'}")

    print("-" * 94)
    print(f"{len(claims) - len(problems)}/{len(claims)} claims recomputed identically")
    if problems:
        print(f"DIVERGED: {problems}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
