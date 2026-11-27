"""A1 runner — the full preregistered matrix.

Arms compared on identical candidates:
* ``weak``     — ``grade_weak`` (PoC/security-only), the published baseline.
* ``static``   — ``grade_obligation`` (finite fixed cases + AST/scope checks).
* ``property`` — ``grade_property_tier1`` (static + seeded randomized cases scored
  against a differential oracle + seeded metamorphic pairs).

Preregistered endpoints live in ``docs/SPEC.md`` §2 A1.  This runner records raw
verdicts; ``analysis/a1_report.py`` recomputes every published number from the
JSONL, and ``verify.py`` recomputes them again offline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gen
from .grade_property import Matrix, build_oracle, grade_property_tier1
from .harness.graders import HOLDS, UNKNOWN, VIOLATED, grade_obligation, grade_weak
from .harness.tasks_io import execute_patch, load_task

BASE_SEED = 20261127
SEEDS = [BASE_SEED + offset for offset in range(20)]
TASK_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Candidate:
    task_id: str
    name: str
    kind: str  # reference | alternate | seeded | mutant | adversarial | e2_negative | e2_positive
    expression: str
    family: str = ""

    @property
    def truth(self) -> str:
        return "valid" if self.kind in {"reference", "alternate", "e2_positive"} else "bad"


def task_ids() -> list[str]:
    return sorted(path.stem for path in (TASK_ROOT / "tasks").glob("*.json"))


def load_tasks() -> dict[str, dict]:
    return {task_id: load_task(task_id, TASK_ROOT) for task_id in task_ids()}


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _lookup_mutants(expression: str, task_id: str) -> list[Candidate]:
    """Mutate the canonical enumerated-lookup expression.

    The published bypass enumerates exact benign values.  A memorising model does
    not have to reproduce that exact set, so the battery mutates it: add a value,
    drop a value, widen to a subset of the same table.  All mutants must die.
    """

    mutants: list[Candidate] = []
    variants = [
        expression.replace('"524288"', '"524288", "2097152"'),
        expression.replace('"0", ', ""),
        expression.replace('"4096", ', '"4096", "8192", '),
        expression.replace('"999999"', '"999999", "1"'),
        expression.replace('"1048576"', '"1048576", "1048575"'),
        expression.replace('in {', 'in frozenset({') .replace("}", "})") if "frozenset" not in expression else expression,
    ]
    for index, variant in enumerate(variants):
        if variant != expression:
            mutants.append(
                Candidate(task_id, f"lookup_mutant[{index}]", "mutant", variant, "lookup")
            )
    return mutants


def build_candidates(tasks: dict[str, dict], e2_path: Path | None) -> list[Candidate]:
    out: list[Candidate] = []
    for task_id, task in tasks.items():
        out.append(Candidate(task_id, "reference", "reference", task["reference_fix"]))
        for index, expression in enumerate(task.get("alternate_fixes") or []):
            out.append(Candidate(task_id, f"alternate[{index}]", "alternate", expression))
        for index, bad in enumerate(task.get("seeded_bad") or []):
            out.append(
                Candidate(task_id, f"seeded_bad[{index}]", "seeded", bad["expression"], bad.get("family", ""))
            )
        for index, adversarial in enumerate(task.get("adversarial") or []):
            out.append(
                Candidate(task_id, f"adversarial[{index}]", "adversarial", adversarial["expression"], "lookup")
            )
            out.extend(_lookup_mutants(adversarial["expression"], task_id))

    if e2_path and e2_path.is_file():
        for line in e2_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            expression = row.get("raw_expression")
            if not isinstance(expression, str) or not expression.strip():
                continue
            kind = row.get("candidate_kind", "")
            if kind in {"reference", "alternate"}:
                out.append(Candidate(row["task_id"], f"e2:{kind}", "e2_positive", expression, "e2"))
            elif kind == "seeded" or kind.startswith("mutation:"):
                out.append(Candidate(row["task_id"], f"e2:{kind}:{row.get('seeded_family','')}", "e2_negative", expression, "e2"))
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Latency: paired per-candidate calls, no multiplication
# ---------------------------------------------------------------------------


def latency_pair(task: dict, seed: int, n: int) -> dict[str, Any]:
    from . import fastrun

    expression = task["reference_fix"]
    fixed_calls = [case["call"] for case in task["security_cases"] + task["benign_cases"]]
    oracle = build_oracle(task, seed)
    pair_calls = gen.METAMORPHIC[task["id"]](seed)
    property_calls = oracle.calls + [call for pair in pair_calls for call in pair]
    result = execute_patch(task, expression)

    static_samples: list[float] = []
    property_samples: list[float] = []
    # Warm compile caches so the measurement is the per-candidate marginal cost,
    # not first-call import/JIT noise.
    fastrun.label_candidate(task, expression, fixed_calls)
    fastrun.label_candidate(task, expression, property_calls)

    for _ in range(n):
        started = time.perf_counter()
        static_labels = fastrun.label_candidate(task, expression, fixed_calls)
        grade_obligation(task, result, expression)
        static_samples.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        grade_property_tier1(task, result, expression, seed, oracle=oracle)
        property_samples.append((time.perf_counter() - started) * 1000.0)

    deltas = [b - a for a, b in zip(static_samples, property_samples)]
    return {
        "task_id": task["id"],
        "seed": seed,
        "n": n,
        "fixed_cases": len(fixed_calls),
        "property_cases": len(oracle.calls),
        "metamorphic_calls": len(property_calls) - len(oracle.calls),
        "static_ms": _stats(static_samples),
        "property_ms": _stats(property_samples),
        "delta_ms": _stats(deltas),
        "oracle_build_ms": round(oracle.build_seconds * 1000.0, 3),
        "oracle_ambiguous_fraction": round(oracle.ambiguous_fraction, 6),
        "static_labels_nonempty": bool(static_labels),
    }


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean": round(statistics.fmean(samples), 4),
        "p50": round(_quantile(ordered, 0.50), 4),
        "p95": round(_quantile(ordered, 0.95), 4),
        "p99": round(_quantile(ordered, 0.99), 4),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evidence")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--e2", default="evidence/reference-e2.jsonl")
    parser.add_argument("--latency-n", type=int, default=10000)
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap candidates per task")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    seeds = SEEDS[: args.seeds]
    candidates = build_candidates(tasks, Path(args.e2))

    if args.limit:
        seen: dict[str, int] = {}
        filtered: list[Candidate] = []
        for candidate in candidates:
            seen[candidate.task_id] = seen.get(candidate.task_id, 0)
            if seen[candidate.task_id] < args.limit:
                filtered.append(candidate)
                seen[candidate.task_id] += 1
        candidates = filtered

    # ---- precompute the seed-independent arms once -------------------------
    # Case execution is a subprocess per call (the harness sandbox).  A candidate
    # is constant across seeds, so execute it once and reuse the result for all
    # three arms and every seed; otherwise the runner measures subprocess spawn,
    # not the graders.
    weak: dict[str, str] = {}
    static: dict[str, tuple[str, list[str]]] = {}
    results: dict[str, Any] = {}
    for candidate in candidates:
        key = f"{candidate.task_id}|{candidate.name}|{candidate.expression}"
        if key in weak:
            continue
        task = tasks[candidate.task_id]
        result = execute_patch(task, candidate.expression)
        results[key] = result
        weak[key] = grade_weak(task, result)
        static[key] = grade_obligation(task, result, candidate.expression)

    rows_path = out_dir / "a1_verdicts.jsonl"
    started_at = time.time()
    with rows_path.open("w", encoding="utf-8") as handle:
        for seed_index, seed in enumerate(seeds):
            oracles = {task_id: build_oracle(task, seed) for task_id, task in tasks.items()}
            for candidate in candidates:
                task = tasks[candidate.task_id]
                key = f"{candidate.task_id}|{candidate.name}|{candidate.expression}"
                result = results[key]
                property_verdict, reasons, stats = grade_property_tier1(
                    task, result, candidate.expression, seed, oracle=oracles[candidate.task_id]
                )
                row = {
                    "seed": seed,
                    "seed_index": seed_index,
                    "task_id": candidate.task_id,
                    "candidate": candidate.name,
                    "kind": candidate.kind,
                    "truth": candidate.truth,
                    "family": candidate.family,
                    "expression": candidate.expression,
                    "weak": weak[key],
                    "static": static[key][0],
                    "static_reasons": static[key][1],
                    "property": property_verdict,
                    "property_reasons": reasons,
                    "property_stats": stats.as_dict(),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[a1] seed {seed} done ({seed_index + 1}/{len(seeds)})", flush=True)

    summary: dict[str, Any] = {
        "base_seed": BASE_SEED,
        "seeds": seeds,
        "candidates": len(candidates),
        "tasks": len(tasks),
        "rows": len(candidates) * len(seeds),
        "verdict_seconds": round(time.time() - started_at, 2),
        "spec_sha256": _read_text(Path("docs/SPEC.sha256")).strip(),
        "arms": {},
    }

    # ---- matrices per arm -------------------------------------------------
    parsed = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    for arm in ("weak", "static", "property"):
        matrix = Matrix()
        for row in parsed:
            matrix.add(row["truth"], row[arm])
        summary["arms"][arm] = matrix.as_dict()

    # ---- preregistered sub-endpoints -------------------------------------
    summary["endpoints"] = _endpoints(parsed)

    # ---- 3-fold leave-one-out on alternates ------------------------------
    summary["leave_one_out"] = _leave_one_out(tasks, seeds[0])

    # ---- latency ----------------------------------------------------------
    if not args.skip_latency:
        latency = [latency_pair(tasks[task_id], seeds[0], args.latency_n) for task_id in tasks]
        summary["latency"] = latency
        (out_dir / "a1_latency.json").write_text(json.dumps(latency, indent=2) + "\n")

    (out_dir / "a1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({arm: summary["arms"][arm]["kill_rate"] for arm in summary["arms"]}, indent=2))
    return 0


def _endpoints(rows: list[dict]) -> dict[str, Any]:
    def rate(arm: str, predicate, denominator) -> dict[str, Any]:
        selected = [row for row in rows if predicate(row)]
        decided = [row for row in selected if row[arm] == VIOLATED]
        return {
            "num": len(decided),
            "den": len(selected),
            "rate": round(len(decided) / len(selected), 6) if selected else None,
        }

    endpoints: dict[str, Any] = {}
    endpoints["canonical_lookup_17_17"] = rate(
        "property", lambda row: row["candidate"].startswith("adversarial["), None
    )
    endpoints["canonical_lookup_17_17"]["note"] = (
        "VIOLATED means the lookup table was caught (desired). The published finding "
        "is that both grade_weak and grade_obligation return HOLDS on it."
    )
    endpoints["lookup_mutant_battery"] = rate(
        "property", lambda row: row["candidate"].startswith("lookup_mutant["), None
    )
    endpoints["e2_bad_corpus"] = rate("property", lambda row: row["kind"] == "e2_negative", None)
    endpoints["e2_positive_controls"] = rate(
        "property", lambda row: row["kind"] == "e2_positive", None
    )
    endpoints["seeded_bad"] = rate("property", lambda row: row["kind"] == "seeded", None)
    endpoints["reference_and_alternates"] = rate(
        "property", lambda row: row["kind"] in {"reference", "alternate"}, None
    )
    for arm in ("weak", "static"):
        endpoints[f"{arm}_e2_bad_corpus"] = rate(arm, lambda row: row["kind"] == "e2_negative", None)
        endpoints[f"{arm}_canonical_lookup"] = rate(
            arm, lambda row: row["candidate"].startswith("adversarial["), None
        )

    # Bad candidates that the static obligation grader admits but the property
    # tier kills: the tier's marginal contribution.  Deduplicated by identity,
    # since every candidate is graded once per seed.
    missed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["truth"] != "bad":
            continue
        if row["static"] == VIOLATED or row["property"] != VIOLATED:
            continue
        key = f'{row["task_id"]}|{row["candidate"]}'
        missed.setdefault(
            key,
            {
                "task_id": row["task_id"],
                "candidate": row["candidate"],
                "kind": row["kind"],
                "static": row["static"],
                "property": row["property"],
                "property_reason": (row["property_reasons"] or [""])[0][:200],
                "expression": row["expression"],
            },
        )
    endpoints["property_only_kills"] = {
        "num": len(missed),
        "items": sorted(missed.values(), key=lambda item: (item["task_id"], item["candidate"])),
    }
    return endpoints


def _leave_one_out(tasks: dict[str, dict], seed: int) -> dict[str, Any]:
    """Score each alternate against an oracle that never saw it."""

    records: dict[str, list[dict[str, Any]]] = {}
    for task_id, task in tasks.items():
        alternates = task.get("alternate_fixes") or []
        for held_out in range(len(alternates)):
            oracle = build_oracle(task, seed, held_out=held_out)
            expression = alternates[held_out]
            result = execute_patch(task, expression)
            verdict, reasons, stats = grade_property_tier1(
                task, result, expression, seed, oracle=oracle
            )
            records.setdefault(f"alternate[{held_out}]", []).append(
                {
                    "task_id": task_id,
                    "verdict": verdict,
                    "reasons": reasons,
                    "ambiguous_fraction": round(stats.ambiguous_fraction, 6),
                }
            )
    summary: dict[str, Any] = {}
    for key, items in records.items():
        rejects = [item for item in items if item["verdict"] == VIOLATED]
        unknown = [item for item in items if item["verdict"] == UNKNOWN]
        summary[key] = {
            "scored": len(items),
            "violated": len(rejects),
            "unknown": len(unknown),
            "violated_tasks": [item["task_id"] for item in rejects],
            "unknown_tasks": [item["task_id"] for item in unknown],
            "detail": rejects + unknown,
        }
    return summary


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
