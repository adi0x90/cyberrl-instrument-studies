#!/usr/bin/env python3
"""Recompute every claim in the accompanying essay directly from archived JSONL.

Runs completely offline using only the Python standard library. Exits non-zero
if any reported count, rate, or p-value drifts from the archived records.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTIER = ROOT / "2026-11-20-frontier-reexperiment"
MASKING = ROOT / "2026-11-20-masking-grpo"

sys.path.insert(0, str(FRONTIER))
from analysis.summarize import mcnemar_exact, wilson_ci  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_e1() -> None:
    rows = load_jsonl(FRONTIER / "evidence" / "e1.jsonl")
    assert len(rows) == 216, f"expected 216 E1 records, got {len(rows)}"

    by_env: dict[str, list[dict]] = {}
    by_pair: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_env.setdefault(r["envelope"], []).append(r)
        by_pair.setdefault((r["model"], r["task_id"], r["decoding"]), {})[r["envelope"]] = r

    strict_adm = sum(1 for r in by_env["strict"] if r["admitted"])
    fence_adm = sum(1 for r in by_env["fence"] if r["admitted"])
    tool_adm = sum(1 for r in by_env["toolcall"] if r["admitted"])
    admitted = [r for r in rows if r["admitted"]]
    holds_given_adm = sum(1 for r in admitted if r["verdict"] == "HOLDS")

    strict_only = sum(
        1 for p in by_pair.values() if p["strict"]["admitted"] and not p["toolcall"]["admitted"]
    )
    tool_only = sum(
        1 for p in by_pair.values() if p["toolcall"]["admitted"] and not p["strict"]["admitted"]
    )
    p_val = mcnemar_exact(strict_only, tool_only)

    tc_reasons = Counter(r["admission_reason"] for r in by_env["toolcall"] if not r["admitted"])

    assert (strict_adm, fence_adm, tool_adm) == (64, 61, 47)
    assert (len(admitted), holds_given_adm) == (172, 172)
    assert (strict_only, tool_only) == (18, 1)
    assert tc_reasons["empty response"] == 24
    assert tc_reasons["unterminated fenced code block"] == 1
    assert abs(p_val - 7.62939453125e-05) < 1e-12

    print(
        f"[E1 OK] strict={strict_adm}/72 ({strict_adm/72:.1%}), "
        f"fence={fence_adm}/72 ({fence_adm/72:.1%}), "
        f"toolcall={tool_adm}/72 ({tool_adm/72:.1%}); "
        f"HOLDS|admitted={holds_given_adm}/{len(admitted)}; "
        f"McNemar {strict_only} vs {tool_only} (p={p_val:.2e}); "
        f"toolcall rejects={dict(tc_reasons)}"
    )


def check_e2() -> None:
    rows = load_jsonl(FRONTIER / "evidence" / "e2.jsonl")
    assert len(rows) == 279, f"expected 279 E2 records, got {len(rows)}"

    negatives = [
        r for r in rows if r["seeded_family"] in {"ast_wrapper", "feature_deletion", "guard_too_weak", "symptom_suppression"}
    ]
    pos_controls = [r for r in rows if r["candidate_kind"] in {"reference", "alternate"}]
    adversarial = [r for r in rows if r["candidate_kind"] == "adversarial"]

    weak_fa = [r for r in negatives if r["weak_verdict"] == "HOLDS" and r["verdict"] == "VIOLATED"]
    by_fam = Counter(r["seeded_family"] for r in weak_fa)
    pos_fr = [r for r in pos_controls if r["verdict"] != "HOLDS"]
    lo, hi = wilson_ci(len(weak_fa), len(negatives))

    weak_only = sum(1 for r in rows if r["weak_verdict"] == "HOLDS" and r["verdict"] != "HOLDS")
    obl_only = sum(1 for r in rows if r["verdict"] == "HOLDS" and r["weak_verdict"] != "HOLDS")
    p_val = mcnemar_exact(weak_only, obl_only)

    assert len(negatives) == 134
    assert len(weak_fa) == 30
    assert by_fam["feature_deletion"] == 12
    assert by_fam["symptom_suppression"] == 17
    assert by_fam["guard_too_weak"] == 1
    assert len(pos_controls) == 48 and len(pos_fr) == 0
    assert len(adversarial) == 1
    assert adversarial[0]["weak_verdict"] == "HOLDS" and adversarial[0]["verdict"] == "HOLDS"
    assert (weak_only, obl_only) == (30, 0)

    print(
        f"[E2 OK] weak false accepts={len(weak_fa)}/{len(negatives)} "
        f"({len(weak_fa)/len(negatives):.1%}, Wilson {lo:.1%}-{hi:.1%}); "
        f"by family={dict(by_fam)}; pos false rejects={len(pos_fr)}/{len(pos_controls)}; "
        f"McNemar {weak_only} vs {obl_only} (p={p_val:.2e}); "
        f"adversarial probe={adversarial[0]['task_id']} (both HOLDS)"
    )


def check_e3() -> None:
    rows = load_jsonl(FRONTIER / "evidence" / "e3.jsonl")
    assert len(rows) == 180, f"expected 180 E3 records, got {len(rows)}"
    counts = Counter(r["verdict"] for r in rows)
    assert counts["CONVERSE_GUARD"] == 0
    print(f"[E3 OK] n={len(rows)}, verdicts={dict(counts)}")


def check_masking_grpo(label: str, subdir: str, expected: dict[str, tuple[int, int, float, int]]) -> None:
    d = MASKING / subdir
    summary = []
    for arm, (exp_eff, exp_target, exp_rpeu, exp_holds) in expected.items():
        metrics = json.loads((d / f"metrics_{arm}_20261121.json").read_text())
        ev = json.loads((d / f"eval_{arm}_20261121.json").read_text())
        eff = metrics["counts"]["effective_updates"]
        target = metrics["budget"]["target_effective_updates"]
        rollouts = metrics["counts"]["rollouts_consumed"]
        rpeu = rollouts / eff
        holds = ev["counts"]["holds"]
        draws = ev["counts"]["draws"]
        assert (eff, target, holds, draws) == (exp_eff, exp_target, exp_holds, 64), (
            f"{label} {arm}: got {(eff, target, holds, draws)}"
        )
        assert abs(rpeu - exp_rpeu) < 0.02, f"{label} {arm}: rpeu {rpeu} != {exp_rpeu}"
        summary.append(f"{arm}: {eff}/{target} eff, {rpeu:.2f} rollouts/upd, held-out {holds}/64")
    print(f"[GRPO {label} OK] " + " | ".join(summary))


def main() -> None:
    check_e1()
    check_e2()
    check_e3()
    check_masking_grpo(
        "1.5B",
        "evidence-1.5b",
        {
            "control": (8, 8, 30.00, 3),
            "masked": (8, 8, 30.00, 3),
            "penalty": (8, 8, 8.00, 3),
        },
    )
    check_masking_grpo(
        "7B",
        "evidence-7b",
        {
            "control": (18, 30, 13.33, 25),
            "masked": (12, 30, 20.00, 26),
            "penalty": (28, 30, 8.57, 27),
        },
    )
    print("\nAll reported numbers verified from archived JSONL/JSON.")


if __name__ == "__main__":
    main()
