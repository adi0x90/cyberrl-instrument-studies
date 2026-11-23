#!/usr/bin/env python3
"""Recompute every endpoint table from results/*.jsonl and write results/REPORT.md.

Design rules:
  * No endpoint number is hardcoded. Every count, rate, CI and p-value is derived
    from the JSONL records named by --e1/--e2/--e3.
  * The statistics are implemented here in the standard library (Wilson score
    interval, exact McNemar on the discordant pairs via the binomial distribution,
    two-sided Fisher exact via hypergeometric enumeration, Cohen's kappa), so the
    report builds on a bare Python 3.10 with no third-party packages. scipy is not
    required; where scipy is installed the numbers agree (checked during build).
  * A missing or empty input file still produces the full report shape with
    "no records" rows, so the report never hides a missing experiment.

Usage:
    python analysis/summarize.py                      # results/e1,e2,e3.jsonl -> results/REPORT.md
    python analysis/summarize.py --mock               # results/mock/e1,e2,e3.jsonl -> results/REPORT.md
    python analysis/summarize.py --e1 ... --e2 ... --e3 ... --out path/REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# --- statistics -----------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (0.0, 0.0) for n == 0: there is no interval to report, and callers
    render that as "n/a" rather than inventing a number.
    """
    if n <= 0:
        return (0.0, 0.0)
    k = max(0, min(int(k), int(n)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs).

    b and c are the two discordant counts; concordant pairs do not enter.
    """
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Sums the hypergeometric probabilities of all tables no more likely than the
    observed one (the standard "sum of probabilities <= p_observed" definition).
    """
    a, b, c, d = (int(a), int(b), int(c), int(d))
    if min(a, b, c, d) < 0:
        raise ValueError("fisher_exact takes non-negative counts")
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c
    denom = math.comb(n, col1) if 0 <= col1 <= n else 0
    if denom == 0:
        return 1.0

    def prob(x: int) -> float:
        if x < 0 or x > row1 or (col1 - x) < 0 or (col1 - x) > (n - row1):
            return 0.0
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / denom

    observed = prob(a)
    low = max(0, col1 - (n - row1))
    high = min(row1, col1)
    total = 0.0
    for x in range(low, high + 1):
        p = prob(x)
        if p <= observed * (1.0 + 1e-9) + 1e-12:
            total += p
    return min(1.0, total)


def kappa(pairs) -> float:
    """Cohen's kappa for two raters over categorical labels.

    ``pairs`` is an iterable of (rater_a, rater_b) label tuples. Returns nan for
    an empty input. When chance agreement is degenerate (pe == 1, i.e. both raters
    used a single identical label) it returns 1.0 if they also agreed on every
    item, else 0.0 -- the documented convention of this report.
    """
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return float("nan")
    labels_a = Counter(a for a, _ in pairs)
    labels_b = Counter(b for _, b in pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum((labels_a[label] / n) * (labels_b[label] / n) for label in set(labels_a) | set(labels_b))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


# --- formatting helpers ---------------------------------------------------


def pct(k: int, n: int) -> str:
    if n <= 0:
        return "0/0 (n/a)"
    return f"{k}/{n} ({100.0 * k / n:.1f}%)"


def ci_text(k: int, n: int) -> str:
    if n <= 0:
        return "n/a"
    low, high = wilson_ci(k, n)
    return f"{100.0 * low:.1f}%–{100.0 * high:.1f}%"


def fmt_p(p: float) -> str:
    if p != p:  # nan
        return "n/a"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_num(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.3f}"


def cell(value, limit: int | None = None) -> str:
    text = "—" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    if limit is not None and len(text) > limit:
        text = text[: max(1, limit - 1)] + "…"
    return text


def code(value, limit: int | None = 90) -> str:
    if value is None:
        return "—"
    text = cell(value, limit)
    if "`" in text:
        return f"`` {text} ``"
    return f"`{text}`"


def md_table(headers: list[str], rows: list[list]) -> str:
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"md_table row/heading mismatch: {len(row)} vs {len(headers)}")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(lines)


def key_part(value) -> str:
    return "" if value is None else str(value)


# --- IO -------------------------------------------------------------------


def load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Read a JSONL file; unparsable lines are counted, never silently dropped."""
    records: list[dict] = []
    problems: list[str] = []
    if not path.exists():
        problems.append(f"{path}: not found")
        return records, problems
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if len(problems) < 5:
                    problems.append(f"{path}:{line_number}: unparsable JSON line")
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
    if bad and len(problems) >= 5:
        problems.append(f"{path}: {bad} unparsable line(s) total")
    return records, problems


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - provenance must never break the report
        return {}


def combined_sha(sha_map: dict) -> str:
    if not isinstance(sha_map, dict) or not sha_map:
        return "n/a"
    payload = "\n".join(f"{name}:{sha_map[name]}" for name in sorted(sha_map))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- pre-committed readings (docs/DESIGN.md §2) --------------------------------

E1_ENVELOPE_PAIR = ("strict", "toolcall")
E2_FALSE_ACCEPT_FLOOR = 0.25
E3_CONVERSE_FLOOR = 0.10
E3_UNKNOWN_CEILING = 0.20


def e1_reading(records: list[dict]) -> str:
    """Apply docs/DESIGN.md §2.1's direction rule; recompute every ingredient."""
    strict = e1_pairs_by_model_task_decoding(records, E1_ENVELOPE_PAIR[0])
    tool = e1_pairs_by_model_task_decoding(records, E1_ENVELOPE_PAIR[1])
    shared = set(strict) & set(tool)
    if not shared:
        return "not established: no matched strict/toolcall units."
    only_strict = sum(1 for key in shared if strict[key] and not tool[key])
    only_tool = sum(1 for key in shared if tool[key] and not strict[key])
    p_value = mcnemar_exact(only_strict, only_tool)
    models = sorted({key[0] for key in shared})
    favourable = 0
    for model in models:
        model_keys = [key for key in shared if key[0] == model]
        a = sum(1 for key in model_keys if strict[key] and not tool[key])
        b = sum(1 for key in model_keys if tool[key] and not strict[key])
        favourable += 1 if a > b else 0
    established = favourable >= 2 and p_value < 0.05 and only_strict > only_tool
    return (
        f"{'**established** under the pre-committed rule' if established else '**not established at this n**'}"
        f" — paired strict-vs-toolcall: {only_strict} strict-only vs {only_tool} toolcall-only "
        f"(exact McNemar p = {fmt_p(p_value)}), direction favourable in {favourable}/{len(models)} models. "
        "Rule (DESIGN §2.1): a working finding needs the direction in >= 2/3 models *and* pooled "
        "p < 0.05."
    )


def e2_reading(records: list[dict]) -> str:
    pooled = [record for record in records if str(record.get("candidate_kind")) in E2_POOLED_KINDS]
    total, false_accepts, _ = e2_rates(pooled)
    if total == 0:
        return "not established: no negative candidates were graded."
    rate = false_accepts / total
    low, _high = wilson_ci(false_accepts, total)
    established = rate >= E2_FALSE_ACCEPT_FLOOR and low > 0.0
    return (
        f"{'**established** under the pre-committed rule' if established else '**not established at this n**'}"
        f" — pooled weak false accept {pct(false_accepts, total)}, Wilson lower bound "
        f"{100.0 * low:.1f}%. Rule (DESIGN §2.2): a working finding needs a pooled rate >= "
        f"{100.0 * E2_FALSE_ACCEPT_FLOOR:.0f}% with a Wilson lower bound above 0."
    )


def e3_reading(records: list[dict]) -> str:
    if not records:
        return "not established: no E3 records."
    unknowns = sum(1 for record in records if str(record.get("verdict")) == "UNKNOWN")
    if unknowns / len(records) > E3_UNKNOWN_CEILING:
        return (
            f"**VOID** — {pct(unknowns, len(records))} of units could not be classified "
            f"(UNKNOWN). Above the {100.0 * E3_UNKNOWN_CEILING:.0f}% ceiling the rate would be "
            "measuring admission, not polarity (DESIGN §6.7). Do not read E3-A as a rate."
        )
    converse = sum(1 for record in records if str(record.get("verdict")) == "CONVERSE")
    rate = converse / len(records)
    left = e3_variant_pairs(records, "explicit")
    right = e3_variant_pairs(records, "neutral")
    shared = set(left) & set(right)
    only_explicit = sum(1 for key in shared if left[key] == "CONVERSE" and right[key] != "CONVERSE")
    only_neutral = sum(1 for key in shared if right[key] == "CONVERSE" and left[key] != "CONVERSE")
    p_value = mcnemar_exact(only_explicit, only_neutral)
    present = rate >= E3_CONVERSE_FLOOR
    mitigation = p_value < 0.05 and only_neutral > only_explicit
    return (
        f"polarity failure {'**present**' if present else '**not established**'} at a pooled "
        f"converse rate of {pct(converse, len(records))} (rule: >= "
        f"{100.0 * E3_CONVERSE_FLOOR:.0f}%, DESIGN §2.3); stating the fact "
        f"{'**reduces**' if mitigation else 'does **not** measurably reduce'} it "
        f"(paired: {only_explicit} converse-only-explicit vs {only_neutral} converse-only-neutral, "
        f"exact McNemar p = {fmt_p(p_value)})."
    )


# --- E1 -------------------------------------------------------------------

E1_ENVELOPE_ORDER = ("strict", "fence", "toolcall")


def envelope_sort_key(envelope: str) -> tuple[int, str]:
    try:
        return (E1_ENVELOPE_ORDER.index(envelope), envelope)
    except ValueError:
        return (len(E1_ENVELOPE_ORDER), envelope)


def outcome_joint(record: dict) -> bool:
    return str(record.get("verdict", "")).upper() == "HOLDS"


def e1_pairs_by_model_task_decoding(records: list[dict], envelope: str) -> dict[tuple, bool]:
    pairs: dict[tuple, bool] = {}
    for record in records:
        if str(record.get("envelope")) != envelope:
            continue
        key = (
            str(record.get("model")),
            str(record.get("task_id")),
            key_part(record.get("decoding")),
        )
        pairs[key] = outcome_joint(record)
    return pairs


def section_e1(records: list[dict]) -> list[str]:
    out: list[str] = []
    out.append("## 1. E1 — output envelope x model")
    out.append("")
    out.append(
        "Hypothesis: the envelope the harness asks for is a real interface cost, not a "
        "formatting detail — models that can state a correct expression will still lose "
        "admissibility (and therefore joint success) when the envelope gets harder "
        "(strict JSON → fenced JSON → unified diff)."
    )
    out.append("")
    out.append(
        "Joint success means `harness.graders.grade_obligation(...)` returned `HOLDS`. "
        "A unit that was never admitted counts as a failure in the denominator, because "
        "an unadmitted patch does not exist."
    )
    out.append("")

    if not records:
        out.append("_No E1 records were found; every table below is empty by construction._")
        out.append("")

    models = sorted({str(r.get("model")) for r in records if r.get("model")})
    envelopes = sorted(
        {str(r.get("envelope")) for r in records if r.get("envelope")},
        key=envelope_sort_key,
    )

    out.append("### E1.a Admission and joint success per (model, envelope)")
    out.append("")
    rows = []
    for model in models:
        for envelope in envelopes:
            group = [
                record
                for record in records
                if str(record.get("model")) == model and str(record.get("envelope")) == envelope
            ]
            if not group:
                continue
            admitted = sum(1 for record in group if record.get("admitted"))
            joint = sum(1 for record in group if outcome_joint(record))
            rows.append(
                [
                    model,
                    envelope,
                    len(group),
                    pct(admitted, len(group)),
                    ci_text(admitted, len(group)),
                    pct(joint, len(group)),
                    ci_text(joint, len(group)),
                ]
            )
    if rows:
        out.append(
            md_table(
                [
                    "model",
                    "envelope",
                    "units",
                    "admitted",
                    "admission rate (Wilson 95%)",
                    "joint success",
                    "joint rate (Wilson 95%)",
                ],
                rows,
            )
        )
    else:
        out.append("_no records_")
    out.append("")

    out.append("**Reading (pre-committed, DESIGN §2.1):** " + e1_reading(records))
    out.append("")
    out.append("### E1.b Pooled over models, and pooled over envelopes")
    out.append("")
    by_envelope_rows = []
    for envelope in envelopes:
        group = [record for record in records if str(record.get("envelope")) == envelope]
        if not group:
            continue
        admitted = sum(1 for record in group if record.get("admitted"))
        joint = sum(1 for record in group if outcome_joint(record))
        by_envelope_rows.append(
            [
                envelope,
                len(group),
                pct(admitted, len(group)),
                ci_text(admitted, len(group)),
                pct(joint, len(group)),
                ci_text(joint, len(group)),
            ]
        )
    by_model_rows = []
    for model in models:
        group = [record for record in records if str(record.get("model")) == model]
        if not group:
            continue
        admitted = sum(1 for record in group if record.get("admitted"))
        joint = sum(1 for record in group if outcome_joint(record))
        by_model_rows.append(
            [
                model,
                len(group),
                pct(admitted, len(group)),
                ci_text(admitted, len(group)),
                pct(joint, len(group)),
                ci_text(joint, len(group)),
            ]
        )
    if by_envelope_rows:
        out.append("By envelope (all models pooled):")
        out.append("")
        out.append(
            md_table(
                ["envelope", "units", "admitted", "admission rate (95% CI)", "joint", "joint rate (95% CI)"],
                by_envelope_rows,
            )
        )
        out.append("")
    if by_model_rows:
        out.append("By model (all envelopes pooled):")
        out.append("")
        out.append(
            md_table(
                ["model", "units", "admitted", "admission rate (95% CI)", "joint", "joint rate (95% CI)"],
                by_model_rows,
            )
        )
        out.append("")
    if not by_envelope_rows:
        out.append("_no records_")
        out.append("")

    conditional_rows = []
    for model in models:
        for envelope in envelopes:
            group = [
                record
                for record in records
                if str(record.get("model")) == model and str(record.get("envelope")) == envelope
            ]
            admitted_group = [record for record in group if record.get("admitted")]
            if not admitted_group:
                continue
            joint = sum(1 for record in admitted_group if outcome_joint(record))
            conditional_rows.append(
                [
                    model,
                    envelope,
                    len(admitted_group),
                    pct(joint, len(admitted_group)),
                    ci_text(joint, len(admitted_group)),
                ]
            )
    if conditional_rows:
        out.append("")
        out.append(
            "Conditional on admission — of the patches that were admitted, how many keep the "
            "obligation? This separates 'could not produce an admissable answer' from 'produced "
            "an admissable answer that does not hold':"
        )
        out.append("")
        out.append(
            md_table(
                ["model", "envelope", "admitted units", "HOLDS given admitted", "rate (95% CI)"],
                conditional_rows,
            )
        )
        out.append("")

    out.append("### E1.c Paired McNemar across envelopes")
    out.append("")
    out.append(
        "Pairs are matched on (model, task, decoding), which is strictly stronger than the "
        "spec's (task, decoding) requirement: the same model answers the same task under two "
        "envelopes, so the only thing that differs is the envelope. The outcome per side is "
        "joint success. Only discordant pairs enter the exact binomial test."
    )
    out.append("")
    mcnemar_rows = []
    for index_a, envelope_a in enumerate(envelopes):
        for envelope_b in envelopes[index_a + 1:]:
            left = e1_pairs_by_model_task_decoding(records, envelope_a)
            right = e1_pairs_by_model_task_decoding(records, envelope_b)
            shared = sorted(set(left) & set(right), key=lambda item: tuple(str(x) for x in item))
            if shared:
                only_a = sum(1 for key in shared if left[key] and not right[key])
                only_b = sum(1 for key in shared if right[key] and not left[key])
                mcnemar_rows.append(
                    [
                        "all",
                        envelope_a,
                        envelope_b,
                        len(shared),
                        only_a,
                        only_b,
                        fmt_p(mcnemar_exact(only_a, only_b)),
                    ]
                )
            for model in models:
                model_keys = [key for key in shared if key[0] == model]
                if not model_keys:
                    continue
                only_a = sum(1 for key in model_keys if left[key] and not right[key])
                only_b = sum(1 for key in model_keys if right[key] and not left[key])
                mcnemar_rows.append(
                    [
                        model,
                        envelope_a,
                        envelope_b,
                        len(model_keys),
                        only_a,
                        only_b,
                        fmt_p(mcnemar_exact(only_a, only_b)),
                    ]
                )
    if mcnemar_rows:
        out.append(
            md_table(
                ["model", "envelope A", "envelope B", "pairs", "A only", "B only", "exact p"],
                mcnemar_rows,
            )
        )
    else:
        out.append("_no envelope pairs available_")
    out.append("")

    out.append("### E1.d Harness monotonicity check (strict ⊆ fence)")
    out.append("")
    instrument = [
        record
        for record in records
        if isinstance(record.get("consumer_admissions"), dict)
    ]
    strict_only = [
        record
        for record in instrument
        if record["consumer_admissions"].get("strict")
        and not record["consumer_admissions"].get("fence")
    ]
    fence_only = [
        record
        for record in instrument
        if record["consumer_admissions"].get("fence")
        and not record["consumer_admissions"].get("strict")
    ]
    out.append(
        "Admission is specified to be monotone: anything `admit_strict` admits, `admit_fence` "
        "must admit. Each recorded reply is replayed through all three consumers, so this is a "
        "check of the parsers on **identical text** — not a comparison of two different model "
        "answers. Over "
        f"{len(instrument)} replies there are **{len(strict_only)}** strict-admitted / "
        f"fence-rejected replies (must be 0) and {len(fence_only)} fence-admitted / "
        "strict-rejected replies (expected: the fence envelope is the superset)."
    )
    out.append("")
    for record in strict_only[:10]:
        out.append(
            f"- {cell(record.get('model'))} / {cell(record.get('task_id'))} / "
            f"{cell(record.get('envelope'))} envelope, reply: {code(record.get('raw_reply'), 70)}"
        )
    if strict_only:
        out.append("")
        out.append(
            "> A non-zero count here is a harness defect, not a model finding: fix "
            "`harness.envelopes` before reading E1-A/B/C."
        )
    out.append("")
    out.append("")
    return out


# --- E2 -------------------------------------------------------------------

E2_POOLED_KINDS = ("seeded", "mutation:swap_op", "mutation:invert_membership", "mutation:weaken_guard")


def e2_false_accept(record: dict) -> bool:
    return str(record.get("weak_verdict")) == "HOLDS" and str(record.get("verdict")) == "VIOLATED"


def e2_rates(records: list[dict]) -> tuple[int, int, int]:
    total = len(records)
    false_accepts = sum(1 for record in records if e2_false_accept(record))
    weak_holds = sum(1 for record in records if str(record.get("weak_verdict")) == "HOLDS")
    return total, false_accepts, weak_holds


def section_e2(records: list[dict]) -> list[str]:
    out: list[str] = []
    out.append("## 2. E2 — grader adequacy: weak vs obligation")
    out.append("")
    out.append(
        "Hypothesis: the cheap check (does the patch satisfy the test cases? `grade_weak`) and "
        "the expensive check (does the patch keep the obligation? `grade_obligation`) disagree "
        "in a structured, family-dependent way. A weak false accept is `grade_weak == HOLDS` "
        "while `grade_obligation == VIOLATED`; that patch passes the cases and is still wrong."
    )
    out.append("")

    if not records:
        out.append("_No E2 records were found; every table below is empty by construction._")
        out.append("")

    pooled = [record for record in records if str(record.get("candidate_kind")) in E2_POOLED_KINDS]
    controls = [
        record
        for record in records
        if str(record.get("candidate_kind")) in ("reference", "alternate")
    ]

    out.append("### E2.a Weak-grader false accepts by seeded-bad family")
    out.append("")
    total, false_accepts, weak_holds = e2_rates(pooled)
    out.append(
        f"Pooled over the negative candidates (seeded entries + the three parseable mutations "
        f"per family): **{pct(false_accepts, total)}** of candidates are weak-accepted and "
        f"obligation-rejected (Wilson 95% CI {ci_text(false_accepts, total)}). "
        f"The weak grader returned HOLDS on {pct(weak_holds, total)} of them."
    )
    out.append("")
    families = sorted({str(record.get("seeded_family")) for record in pooled})
    rows = []
    for family in families:
        group = [record for record in pooled if str(record.get("seeded_family")) == family]
        group_total, group_false, group_weak = e2_rates(group)
        rows.append(
            [
                family,
                group_total,
                pct(group_weak, group_total),
                pct(group_false, group_total),
                ci_text(group_false, group_total),
            ]
        )
    if rows:
        out.append(
            md_table(
                ["seeded-bad family", "candidates", "weak HOLDS", "weak false accept", "false-accept rate (95% CI)"],
                rows,
            )
        )
    else:
        out.append("_no negative candidates_")
    out.append("")

    out.append("**Reading (pre-committed, DESIGN §2.2):** " + e2_reading(records))
    out.append("")
    out.append("### E2.b Obligation false rejects on reference and alternate fixes")
    out.append("")
    out.append(
        "Positive controls: an obligation grader that rejects a genuinely correct fix is as "
        "damaging as one that accepts a broken fix. Reference and alternate fixes are correct "
        "by construction, so any `VIOLATED` here is a false reject."
    )
    out.append("")
    if controls:
        control_total = len(controls)
        rejected = sum(1 for record in controls if str(record.get("verdict")) == "VIOLATED")
        held = sum(1 for record in controls if str(record.get("verdict")) == "HOLDS")
        weak_rejected = sum(1 for record in controls if str(record.get("weak_verdict")) == "VIOLATED")
        out.append(
            f"**{pct(rejected, control_total)}** false rejects (Wilson 95% CI "
            f"{ci_text(rejected, control_total)}); {held} HOLDS, {control_total - held - rejected} "
            f"UNKNOWN. The weak grader rejected {pct(weak_rejected, control_total)} of the same "
            f"controls."
        )
        out.append("")
        rows = []
        for kind in ("reference", "alternate"):
            group = [record for record in controls if str(record.get("candidate_kind")) == kind]
            if not group:
                continue
            group_rejected = sum(1 for record in group if str(record.get("verdict")) == "VIOLATED")
            rows.append([kind, len(group), pct(group_rejected, len(group)), ci_text(group_rejected, len(group))])
        if rows:
            out.append(md_table(["control kind", "candidates", "false rejects", "rate (95% CI)"], rows))
            out.append("")
        rejected_examples = [record for record in controls if str(record.get("verdict")) == "VIOLATED"]
        for record in rejected_examples[:10]:
            out.append(
                f"- {cell(record.get('task_id'))} / {cell(record.get('candidate_kind'))}: "
                f"{code(record.get('raw_expression'), 70)} — "
                f"{cell('; '.join(str(item) for item in (record.get('reasons') or [])), 140)}"
            )
        if rejected_examples:
            out.append("")
    else:
        out.append("_no reference / alternate control candidates were graded_")
        out.append("")

    out.append("### E2.c Paired McNemar (weak vs obligation) and Cohen's kappa")
    out.append("")
    paired = [
        record
        for record in records
        if str(record.get("weak_verdict")) in ("HOLDS", "VIOLATED")
        and str(record.get("verdict")) in ("HOLDS", "VIOLATED")
    ]
    weak_only = sum(
        1 for record in paired
        if str(record.get("weak_verdict")) == "HOLDS" and str(record.get("verdict")) == "VIOLATED"
    )
    obligation_only = sum(
        1 for record in paired
        if str(record.get("weak_verdict")) == "VIOLATED" and str(record.get("verdict")) == "HOLDS"
    )
    kappa_value = kappa(
        [
            (str(record.get("weak_verdict")) == "HOLDS", str(record.get("verdict")) == "HOLDS")
            for record in paired
        ]
    )
    out.append(
        f"Over {len(paired)} candidates graded by both graders: weak-only HOLDS = {weak_only}, "
        f"obligation-only HOLDS = {obligation_only}, exact McNemar p = "
        f"**{fmt_p(mcnemar_exact(weak_only, obligation_only))}**. Cohen's kappa between the two "
        f"binary verdicts is **{fmt_num(kappa_value)}**."
    )
    out.append("")

    out.append("### E2.d Task-family breakdown (negative candidates)")
    out.append("")
    task_families = sorted({str(record.get("family")) for record in pooled})
    rows = []
    for family in task_families:
        group = [record for record in pooled if str(record.get("family")) == family]
        group_total, group_false, group_weak = e2_rates(group)
        rows.append(
            [family, group_total, pct(group_weak, group_total), pct(group_false, group_total), ci_text(group_false, group_total)]
        )
    if rows:
        out.append(
            md_table(
                ["task family", "candidates", "weak HOLDS", "weak false accept", "rate (95% CI)"],
                rows,
            )
        )
    else:
        out.append("_no negative candidates_")
    out.append("")

    out.append("### E2.e Hand-written adversarial candidates (reported separately)")
    out.append("")
    adversarial = [
        record for record in records if str(record.get("candidate_kind")) == "adversarial"
    ]
    out.append(
        "These are the per-family hand-written candidates whose intent is *satisfy the "
        "obligation checks and leave the root cause intact*. They are excluded from the pooled "
        "false-accept rate: they are diagnostic probes for the grader's blind spots, not a "
        "sample of model behaviour."
    )
    out.append("")
    if adversarial:
        rows = []
        for record in adversarial:
            rows.append(
                [
                    record.get("task_id"),
                    code(record.get("raw_expression"), 60),
                    record.get("weak_verdict"),
                    record.get("verdict"),
                    cell("; ".join(str(item) for item in (record.get("reasons") or [])), 120),
                ]
            )
        out.append(md_table(["task", "expression", "weak", "obligation", "obligation reasons"], rows))
    else:
        out.append("_no adversarial candidates present in the corpus_")
    out.append("")

    out.append("### E2.f Admitted E1 patches (reported separately)")
    out.append("")
    e1_patches = [record for record in records if str(record.get("candidate_kind")) == "e1_patch"]
    if e1_patches:
        patch_total = len(e1_patches)
        patch_false = sum(1 for record in e1_patches if e2_false_accept(record))
        out.append(
            f"{patch_total} admitted E1 patches were imported; **{pct(patch_false, patch_total)}** "
            f"are weak-accepted and obligation-rejected (Wilson 95% CI "
            f"{ci_text(patch_false, patch_total)})."
        )
        out.append("")
        rows = []
        for record in e1_patches:
            rows.append(
                [
                    record.get("task_id"),
                    record.get("model"),
                    code(record.get("raw_expression"), 55),
                    record.get("weak_verdict"),
                    record.get("verdict"),
                ]
            )
        out.append(md_table(["task", "model", "expression", "weak", "obligation"], rows))
    else:
        out.append("_no admitted E1 patches were available when E2 ran_")
    out.append("")
    return out


# --- E3 -------------------------------------------------------------------


def e3_variant_pairs(records: list[dict], variant: str) -> dict[tuple, str]:
    pairs: dict[tuple, str] = {}
    for record in records:
        if str(record.get("variant")) != variant:
            continue
        key = (
            str(record.get("task_id")),
            str(record.get("model")),
            key_part(record.get("decoding")),
        )
        pairs[key] = str(record.get("verdict"))
    return pairs


def section_e3(records: list[dict]) -> list[str]:
    out: list[str] = []
    out.append("## 3. E3 — polarity / converse guard")
    out.append("")
    out.append(
        "Hypothesis: some failures are not formatting failures but polarity failures — the "
        "expression is admissible, plausible, and implements the *converse* of the property "
        "(`!=` where `==` is required). Stating the polarity fact in the prompt should reduce "
        "the converse-guard rate; if it does not, the failure is not an instruction-following gap."
    )
    out.append("")
    out.append(
        "Classification is `harness.polarity.converse_guard(expression, task['polarity'])`. The "
        "polarity *markers* are never shown to the model; only the natural-language fact is, and "
        "only in the `explicit` variant."
    )
    out.append("")

    if not records:
        out.append("_No E3 records were found; every table below is empty by construction._")
        out.append("")

    models = sorted({str(record.get("model")) for record in records if record.get("model")})
    variants = sorted({str(record.get("variant")) for record in records if record.get("variant")})

    out.append("### E3.a Converse-guard rate per (model, variant)")
    out.append("")
    rows = []
    for model in models:
        for variant in variants:
            group = [
                record
                for record in records
                if str(record.get("model")) == model and str(record.get("variant")) == variant
            ]
            if not group:
                continue
            converse = sum(1 for record in group if str(record.get("verdict")) == "CONVERSE")
            correct = sum(1 for record in group if str(record.get("verdict")) == "CORRECT")
            unknown = len(group) - converse - correct
            rows.append(
                [
                    model,
                    variant,
                    len(group),
                    pct(converse, len(group)),
                    ci_text(converse, len(group)),
                    correct,
                    unknown,
                ]
            )
    if rows:
        out.append(
            md_table(
                [
                    "model",
                    "variant",
                    "units",
                    "converse guard",
                    "converse rate (Wilson 95%)",
                    "CORRECT",
                    "UNKNOWN",
                ],
                rows,
            )
        )
    else:
        out.append("_no records_")
    out.append("")

    out.append("**Reading (pre-committed, DESIGN §2.3 and §6.7):** " + e3_reading(records))
    out.append("")
    out.append("### E3.b Fisher exact (explicit vs neutral, 2x2)")
    out.append("")
    out.append(
        "Unpaired view: 2x2 table of converse/not against variant, tested two-sided. Pooled rows "
        "treat every (task, decoding) as its own observation, so they are anti-conservative when "
        "the same model is reused; the per-model rows are the safer read."
    )
    out.append("")
    explicit = [record for record in records if str(record.get("variant")) == "explicit"]
    neutral = [record for record in records if str(record.get("variant")) == "neutral"]

    def converse_counts(group: list[dict]) -> tuple[int, int]:
        converse = sum(1 for record in group if str(record.get("verdict")) == "CONVERSE")
        return converse, len(group) - converse

    fisher_rows = []
    if explicit or neutral:
        for model in ["all"] + models:
            left = explicit if model == "all" else [record for record in explicit if str(record.get("model")) == model]
            right = neutral if model == "all" else [record for record in neutral if str(record.get("model")) == model]
            if not left and not right:
                continue
            a, b = converse_counts(left)
            c, d = converse_counts(right)
            fisher_rows.append(
                [
                    model,
                    pct(a, a + b),
                    pct(c, c + d),
                    f"{a}/{b} vs {c}/{d}",
                    fmt_p(fisher_exact(a, b, c, d)),
                ]
            )
    if fisher_rows:
        out.append(
            md_table(
                [
                    "model",
                    "explicit converse",
                    "neutral converse",
                    "2x2 (conv/other vs conv/other)",
                    "Fisher exact p",
                ],
                fisher_rows,
            )
        )
    else:
        out.append("_no records_")
    out.append("")

    out.append("### E3.c McNemar exact (paired explicit vs neutral)")
    out.append("")
    out.append(
        "Paired view, matched on (task, model, decoding): the same task at the same decoding, "
        "with and without the polarity fact. Discordant pairs only."
    )
    out.append("")
    mcnemar_rows = []
    left_pairs = e3_variant_pairs(records, "explicit")
    right_pairs = e3_variant_pairs(records, "neutral")
    shared = sorted(set(left_pairs) & set(right_pairs), key=lambda item: tuple(str(x) for x in item))
    if shared:
        for model in ["all"] + models:
            keys = shared if model == "all" else [key for key in shared if key[1] == model]
            if not keys:
                continue
            only_explicit = sum(
                1 for key in keys if left_pairs[key] == "CONVERSE" and right_pairs[key] != "CONVERSE"
            )
            only_neutral = sum(
                1 for key in keys if right_pairs[key] == "CONVERSE" and left_pairs[key] != "CONVERSE"
            )
            mcnemar_rows.append(
                [
                    model,
                    len(keys),
                    only_explicit,
                    only_neutral,
                    fmt_p(mcnemar_exact(only_explicit, only_neutral)),
                ]
            )
    if mcnemar_rows:
        out.append(
            md_table(
                [
                    "model",
                    "pairs",
                    "converse only with explicit",
                    "converse only with neutral",
                    "exact p",
                ],
                mcnemar_rows,
            )
        )
    else:
        out.append("_no matched pairs_")
    out.append("")

    out.append("### E3.d Every returned expression (audit log)")
    out.append("")
    if records:
        rows = []
        for record in sorted(
            records,
            key=lambda item: (
                str(item.get("task_id")),
                str(item.get("model")),
                str(item.get("variant")),
                key_part(item.get("decoding")),
            ),
        ):
            rows.append(
                [
                    record.get("task_id"),
                    record.get("model"),
                    record.get("variant"),
                    key_part(record.get("decoding")),
                    record.get("verdict"),
                    code(record.get("raw_expression"), 70),
                ]
            )
        out.append(
            md_table(
                ["task", "model", "variant", "decoding", "classification", "expression"],
                rows,
            )
        )
    else:
        out.append("_no expressions logged_")
    out.append("")
    return out


# --- framing sections -----------------------------------------------------


def section_limits() -> list[str]:
    return [
        "## What this does not show",
        "",
        "- **It does not rank the three models.** Twelve tasks, three decodings and one vendor "
        "route cannot support a capability ranking, and the task corpus is written by the same "
        "people who wrote the graders.",
        "- **Mock artifacts are not evidence.** `--mock` replies are deterministic canned strings "
        "keyed by (model, prompt hash). They exercise the plumbing; they say nothing about any "
        "model. Every mock report must be regenerated before it is read.",
        "- **It does not measure the obligation grader's correctness.** E2 reports agreement "
        "between two checks and the false rejects on hand-written controls. The obligation rules "
        "are themselves a program with bugs; a patch graded HOLDS is 'not caught by this rule "
        "set', not 'correct'.",
        "- **The adversarial candidates are hand-written probes, not a sample.** The count of "
        "adversarial cases that slip past the obligation grader is an existence demonstration; "
        "it is not a rate and has no confidence interval.",
        "- **Mutations are syntactic, not semantic.** `swap_op`, `invert_membership` and "
        "`weaken_guard` explore a neighbourhood of the seeded entry that a textual rewrite can "
        "reach. A family whose real-world failure mode needs a restructured expression is "
        "under-sampled, so the per-family false-accept rates are not comparable across families.",
        "- **Envelope effects are confounded with prompt length and instruction count.** The "
        "three envelopes differ in the output contract and in how much text they add. E1 cannot "
        "separate 'format cost' from 'extra instruction cost'.",
        "- **Polarity classification is lexical.** `converse_guard` decides from markers, so a "
        "correct expression written in an unusual form can be labelled CONVERSE and vice versa. "
        "E3.d logs every expression so the labels can be audited by hand; the rate is only as "
        "good as the classifier.",
        "- **Temperature and seed are recorded, not guaranteed.** Vendors may ignore seeds or "
        "route to different backends, so a re-run is a new sample, not a replay. No variance "
        "decomposition is possible from two or three decodings.",
        "- **Only one serving path was measured.** All calls go through one aggregator with "
        "whatever quantisation and routing it applies. The results describe that endpoint, not "
        "an underlying model artifact.",
        "- **No causal claim about training.** A converse guard or a symptom-suppression patch is "
        "an observable output pattern; nothing here identifies why the model produced it.",
    ]


def section_provenance(
    inputs: list[tuple[str, Path, int]],
    manifests: list[tuple[str, Path]],
) -> list[str]:
    out = ["## Provenance", ""]
    out.append(
        "Manifests are written by each runner next to its JSONL. The report never edits them; "
        "it prints the fields that make a run identifiable."
    )
    out.append("")
    rows = []
    for experiment, path, count in inputs:
        rows.append([experiment, code(str(path), 70), count, path.exists()])
    out.append(md_table(["experiment", "input", "records", "exists"], rows))
    out.append("")

    manifest_rows = []
    for experiment, path in manifests:
        data = load_manifest(path)
        if not data:
            manifest_rows.append([experiment, code(str(path), 60), "_manifest missing or unreadable_", "", "", "", "", "", "", ""])
            continue
        manifest_rows.append(
            [
                experiment,
                code(str(path), 60),
                data.get("git_sha", "unknown"),
                data.get("utc", "unknown"),
                data.get("mock"),
                data.get("seed"),
                ", ".join(str(model) for model in (data.get("models") or [])),
                len(data.get("task_sha256") or {}),
                combined_sha(data.get("task_sha256") or {}),
                code(" ".join(str(part) for part in (data.get("argv") or [])), 70),
            ]
        )
    out.append(
        md_table(
            [
                "experiment",
                "manifest",
                "git sha",
                "utc",
                "mock",
                "seed",
                "models",
                "task files",
                "combined task sha (16)",
                "argv",
            ],
            manifest_rows,
        )
    )
    out.append("")
    out.append(
        "Recompute any table above with "
        "`python analysis/summarize.py --e1 results/e1.jsonl --e2 results/e2.jsonl "
        "--e3 results/e3.jsonl --out results/REPORT.md`."
    )
    out.append("")
    return out


# --- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the frontier re-experiment report from JSONL records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="read mock artifacts from results/mock/ instead of results/ (no effect on --out)",
    )
    parser.add_argument("--e1", default=None, help="E1 JSONL path")
    parser.add_argument("--e2", default=None, help="E2 JSONL path")
    parser.add_argument("--e3", default=None, help="E3 JSONL path")
    parser.add_argument("--out", default=str(RESULTS_DIR / "REPORT.md"), help="report output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = RESULTS_DIR / "mock" if args.mock else RESULTS_DIR
    inputs = [
        ("E1", Path(args.e1) if args.e1 else base / "e1.jsonl"),
        ("E2", Path(args.e2) if args.e2 else base / "e2.jsonl"),
        ("E3", Path(args.e3) if args.e3 else base / "e3.jsonl"),
    ]

    records_by_experiment: dict[str, list[dict]] = {}
    problems: list[str] = []
    for experiment, path in inputs:
        records, warnings = load_jsonl(path)
        records_by_experiment[experiment] = records
        problems.extend(warnings)

    lines: list[str] = []
    lines.append("# Frontier re-experiment — recomputed report")
    lines.append("")
    lines.append(
        "Generated by `analysis/summarize.py` from the JSONL records below. Every count, rate, "
        "confidence interval and p-value in this document is recomputed from those records; no "
        "endpoint number is hardcoded."
    )
    lines.append("")
    lines.append(
        "Statistics: Wilson score interval (95%, z = 1.96), exact McNemar on discordant pairs "
        "(two-sided binomial, p = 0.5), two-sided Fisher exact by hypergeometric enumeration, "
        "and Cohen's kappa for the weak/obligation double labels. Implemented in the standard "
        "library; scipy is not required."
    )
    lines.append("")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"Generated at {generated}. Mock artifacts: {'yes' if args.mock else 'no'}.")
    lines.append("")
    if problems:
        lines.append("> Input problems:")
        for problem in problems:
            lines.append(f"> - {problem}")
        lines.append("")

    lines.extend(section_e1(records_by_experiment["E1"]))
    lines.extend(section_e2(records_by_experiment["E2"]))
    lines.extend(section_e3(records_by_experiment["E3"]))
    lines.extend(section_limits())
    lines.append("")
    manifests = [
        (experiment, path.with_name(path.stem + ".manifest.json"))
        for experiment, path in inputs
    ]
    lines.extend(
        section_provenance(
            [(experiment, path, len(records_by_experiment[experiment])) for experiment, path in inputs],
            manifests,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(
        f"[report] wrote {out} from "
        + ", ".join(f"{experiment}={len(records_by_experiment[experiment])} records" for experiment, _ in inputs),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
