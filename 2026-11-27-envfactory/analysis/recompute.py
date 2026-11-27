"""Pure-stdlib recomputation of every headline number.

Both ``analysis/build_report.py`` and ``verify.py`` call these functions.  Keeping
one implementation means the report and its verifier cannot drift; the verifier
recomputes from raw evidence and compares, rather than trusting a summary file.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

HOLDS = "HOLDS"
VIOLATED = "VIOLATED"
UNKNOWN = "UNKNOWN"


def load_jsonl(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_matrix(rows: Iterable[dict], arm: str) -> dict[str, Any]:
    matrix: dict[str, dict[str, int]] = {verdict: {"bad": 0, "valid": 0} for verdict in (HOLDS, VIOLATED, UNKNOWN)}
    for row in rows:
        matrix[row[arm]][row["truth"]] += 1
    bad = matrix[HOLDS]["bad"] + matrix[VIOLATED]["bad"] + matrix[UNKNOWN]["bad"]
    valid = matrix[HOLDS]["valid"] + matrix[VIOLATED]["valid"] + matrix[UNKNOWN]["valid"]
    total = bad + valid
    unknown = matrix[UNKNOWN]["bad"] + matrix[UNKNOWN]["valid"]
    return {
        "matrix": matrix,
        "kills": matrix[VIOLATED]["bad"],
        "bad_total": bad,
        "valid_total": valid,
        "kill_rate": matrix[VIOLATED]["bad"] / bad if bad else None,
        "false_reject_rate": matrix[VIOLATED]["valid"] / valid if valid else None,
        "decisive_coverage": 1.0 - unknown / total if total else None,
    }


def endpoint(rows: Iterable[dict], arm: str, predicate) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    caught = sum(1 for row in selected if row[arm] == VIOLATED)
    return {"num": caught, "den": len(selected), "rate": caught / len(selected) if selected else None}


def a1_endpoints(rows: list[dict], arm: str = "property") -> dict[str, Any]:
    return {
        "canonical_lookup": endpoint(rows, arm, lambda r: r["candidate"].startswith("adversarial[")),
        "lookup_mutants": endpoint(rows, arm, lambda r: r["candidate"].startswith("lookup_mutant[")),
        "e2_bad_corpus": endpoint(rows, arm, lambda r: r["kind"] == "e2_negative"),
        "e2_positive_controls": endpoint(rows, arm, lambda r: r["kind"] == "e2_positive"),
        "seeded_bad": endpoint(rows, arm, lambda r: r["kind"] == "seeded"),
        "references_and_alternates": endpoint(rows, arm, lambda r: r["kind"] in {"reference", "alternate"}),
    }


def a1_property_only_kills(rows: list[dict]) -> list[dict]:
    """Bad candidates that the static obligation grader admits (per identity)."""

    seen: dict[str, dict] = {}
    for row in rows:
        if row["truth"] != "bad":
            continue
        if row["static"] == VIOLATED or row["property"] != VIOLATED:
            continue
        key = f'{row["task_id"]}|{row["candidate"]}'
        seen.setdefault(
            key,
            {
                "task_id": row["task_id"],
                "candidate": row["candidate"],
                "kind": row["kind"],
                "static": row["static"],
                "property": row["property"],
                "reason": (row.get("property_reasons") or [""])[0][:180],
            },
        )
    return sorted(seen.values(), key=lambda item: (item["task_id"], item["candidate"]))


def pctl(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def a3_rows(payload: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"targets": {}, "gate": {}, "concurrency": {}, "not_run": payload.get("not_run", [])}
    for entry in payload.get("targets", []):
        pack = entry["pack"]
        if "gate" in entry:
            out["gate"][pack] = entry["gate"]["state_equivalence"]
        strategies = {}
        for row in entry.get("timings", []):
            samples = row.get("samples_s") or []
            strategies[row["strategy"]] = {
                "completed": row["completed"],
                "failures": row["failures"],
                "p50": pctl(samples, 0.50) if samples else row.get("p50_s"),
                "p95": pctl(samples, 0.95) if samples else row.get("p95_s"),
                "p99": pctl(samples, 0.99) if samples else row.get("p99_s"),
                "recomputed_from_samples": bool(samples),
            }
        out["targets"][pack] = strategies
        out["concurrency"][pack] = entry.get("concurrency", [])
    return out


def a2_rows(payload: dict) -> dict[str, Any]:
    return {
        "clean_packs": len(payload.get("clean", [])),
        "clean_packs_passing": sum(1 for row in payload.get("clean", []) if row.get("integrity_pass")),
        "d1": payload.get("defects", {}).get("D1", {}).get("caught"),
        "d2": payload.get("defects", {}).get("D2", {}).get("caught"),
        "d3": payload.get("defects", {}).get("D3", {}).get("caught"),
    }


def summarise(evidence: Path) -> dict[str, Any]:
    """Everything the report asserts, computed from raw evidence."""

    verdicts = load_jsonl(evidence / "a1" / "a1_verdicts.jsonl")
    summary: dict[str, Any] = {
        "a1": {
            "rows": len(verdicts),
            "seeds": sorted({row["seed"] for row in verdicts}),
            "arms": {arm: arm_matrix(verdicts, arm) for arm in ("weak", "static", "property")},
            "endpoints": {arm: a1_endpoints(verdicts, arm) for arm in ("weak", "static", "property")},
            "property_only_kills": a1_property_only_kills(verdicts),
        }
    }
    native = evidence / "a1" / "a1_native.json"
    if native.is_file():
        doc = load_json(native)
        summary["a1"]["native"] = {
            "variants": doc["variants"],
            "cheap": doc["cheap"],
            "property": doc["property"],
        }
    latency = evidence / "a1" / "a1_latency.json"
    if latency.is_file():
        summary["a1"]["latency"] = load_json(latency)
    reset = evidence / "a3" / "a3_reset.json"
    if reset.is_file():
        summary["a3"] = a3_rows(load_json(reset))
        payload = load_json(reset)
        summary["a3"]["qemu"] = payload.get("qemu")
        summary["a3"]["criu"] = payload.get("criu")
        summary["a3"]["economics"] = payload.get("economics")
        summary["a3"]["reset_share_thresholds"] = payload.get("reset_share_thresholds", [])
    tests = evidence / "a2" / "a2_tests.json"
    if tests.is_file():
        summary["a2"] = a2_rows(load_json(tests))
    return summary


__all__ = [
    "HOLDS",
    "UNKNOWN",
    "VIOLATED",
    "a1_endpoints",
    "a1_property_only_kills",
    "a2_rows",
    "a3_rows",
    "arm_matrix",
    "endpoint",
    "load_json",
    "load_jsonl",
    "pctl",
    "summarise",
]
