#!/usr/bin/env python3
"""Recompute every endpoint of the masking experiment and write ``results/REPORT.md``.

Arms: ``control`` and ``masked`` (``docs/SPEC.md`` v1) plus ``penalty``
("Amendment v2").  Any arm whose artefacts are absent is reported as **not run**
in every table; it is never silently omitted and never rendered as a zero.

Inputs (all optional individually; a missing file is reported, never hidden).
A *seed run* is the flat ``*_<arm>.json[l]`` set or a ``seed<number>/``
subdirectory (or a ``*_<arm>_<number>.json`` suffix); the seed is read from the
metrics file's ``seed`` field:

* ``rollouts_<arm>.jsonl`` -- one record per attempted completion.  **Every**
  arm-level count in the report is recomputed here from these records, not read
  from the metrics file: groups are rebuilt by ``group_index``, rewards and
  admission flags are re-read, and the arm treatment is re-applied through
  :func:`grpo.group_advantage`.  The metrics file is then cross-checked against
  the recomputation and any disagreement is printed as an integrity problem.
* ``metrics_<arm>.json`` -- the run manifest (model, revision, GPU, git SHA, stop
  reason, budget) and the per-group log the training loop wrote.
* ``eval_<arm>.json`` -- the held-out draws (4 tasks x ``--decodings``; Amendment
  v2 fixes 16, so 64 draws).  The co-primary endpoints are recomputed from the
  per-draw records, not from the stored aggregates.

Rules this file follows:

* **No endpoint number is hardcoded.**  Everything is derived from the inputs,
  including the denominators, the intervals and the decision-rule verdict.  The
  only constants are the pre-committed thresholds from ``docs/SPEC.md``.
* **Statistics are standard library.**  Wilson score interval and the exact
  paired sign test (two-sided binomial on discordant pairs).  No scipy.
* **A missing input still produces the full report shape**, with explicit
  "not run" rows, so an absent arm or a missing eval cannot look like a null
  result or a zero.
* **The protocol hash is checked.**  The SPEC's SHA-256 is recorded by the
  runners; if ``docs/SPEC.md`` no longer hashes to the same value, the report
  says so at the top -- that is what a post-hoc protocol edit would look like.

Usage:
    python analysis/summarize.py                       # results/ -> results/REPORT.md
    python analysis/summarize.py --dry-run             # results/dryrun/ -> .../REPORT.md
    python analysis/summarize.py --dir results --out results/REPORT.md
    python analysis/summarize.py --dir tmp/two-arms --out tmp/two-arms/REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grpo  # noqa: E402  (stdlib-only maths; safe to import here)

DEFAULT_RESULTS = ROOT / "results"
DRYRUN_SUBDIR = "dryrun"

# --- pre-committed constants (docs/SPEC.md, "Decision rule") -------------------

#: "Masked better only if held-out joint success exceeds control by >= 12
#: percentage points with non-overlapping Wilson intervals."
OUTCOME_THRESHOLD = 0.12
Z_95 = 1.96

#: The budget the protocol fixes, used only to state a shortfall when a metrics
#: file is missing.  Never used as an observed value.
TARGET_EFFECTIVE_UPDATES = 24
MAX_GROUPS = 96

ARMS: tuple[str, ...] = tuple(grpo.ARMS)

#: The pre-registered pair comparisons, as ``(treatment, reference)``.  A
#: comparison's difference is ``treatment - reference``.
PAIRS: tuple[tuple[str, str], ...] = (
    ("masked", "control"),
    ("penalty", "control"),
    ("penalty", "masked"),
)

#: Rendered wherever an arm (or a pair) was never run.
NOT_RUN = "not run"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""

    if n <= 0:
        return (0.0, 0.0)
    successes = max(0, min(int(successes), int(n)))
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_sign_test_p(b: int, c: int) -> float:
    """Two-sided exact binomial (sign) test on the discordant pair counts.

    ``b`` and ``c`` are the two discordant counts; concordant pairs carry no
    information about the direction.  Returns 1.0 when there are no discordant
    pairs.
    """

    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0/0 (n/a)"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def ci_text(successes: int, n: int) -> str:
    if n <= 0:
        return "n/a"
    low, high = wilson_ci(successes, n)
    return f"{100.0 * low:.1f}%–{100.0 * high:.1f}%"


def fmt_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def fmt_num(value: Optional[float], places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def fmt_p(value: float) -> str:
    if value != value:  # nan
        return "n/a"
    return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"


def cell(value, limit: Optional[int] = None) -> str:
    text = "—" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    if limit is not None and len(text) > limit:
        text = text[: max(1, limit - 1)] + "…"
    return text


def rel_path(path: Path, base: Path) -> str:
    """A path relative to ``base`` when possible, so the report is not wall-to-wall
    absolute paths."""

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def code(value, limit: int = 90) -> str:
    if value is None:
        return "—"
    text = cell(value, limit)
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def md_table(headers: list[str], rows: list[list]) -> str:
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"md_table row/heading mismatch: {len(row)} vs {len(headers)}")
    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Read a JSONL file; unparsable lines are counted, never silently dropped."""

    records: list[dict] = []
    problems: list[str] = []
    if not path.exists():
        return records, [f"{path}: not found"]
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


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Per-arm recomputation
# ---------------------------------------------------------------------------


@dataclass
class GroupView:
    """One group rebuilt from the rollout records and re-treated by the arm."""

    group_index: int
    task_id: str
    n: int
    rewards: list[float]
    valid: list[bool]
    stats: grpo.GroupStats
    effective: bool

    @property
    def skipped(self) -> bool:
        return self.stats.arm == "masked" and self.stats.n_valid < 2


@dataclass
class ArmAnalysis:
    arm: str
    rollouts_path: Path
    metrics_path: Path
    eval_path: Path
    rollouts: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    eval_payload: dict = field(default_factory=dict)
    groups: list[GroupView] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    discrepancies: list[str] = field(default_factory=list)
    seed: Optional[int] = None

    # -- rollouts ---------------------------------------------------------

    @property
    def present(self) -> bool:
        return bool(self.rollouts)

    @property
    def ran(self) -> bool:
        """Did this arm produce *any* of its three artefacts?

        An arm with no files at all is "not run" and must be reported as such;
        an arm whose files exist but are empty is "ran with no records".
        """

        return any(
            path.exists()
            for path in (self.rollouts_path, self.metrics_path, self.eval_path)
        )

    @property
    def records(self) -> int:
        return len(self.rollouts)

    @property
    def admitted(self) -> int:
        return sum(1 for record in self.rollouts if record.get("admitted"))

    @property
    def holds(self) -> int:
        return sum(1 for record in self.rollouts if str(record.get("verdict")) == "HOLDS")

    @property
    def groups_sampled(self) -> int:
        return len(self.groups)

    @property
    def effective_updates(self) -> int:
        return sum(1 for group in self.groups if group.effective)

    @property
    def groups_skipped(self) -> int:
        return sum(1 for group in self.groups if group.skipped)

    @property
    def groups_zero_variance(self) -> int:
        return sum(1 for group in self.groups if not group.effective)

    @property
    def rollouts_consumed(self) -> int:
        return sum(group.n for group in self.groups)

    @property
    def effective_group_rate(self) -> Optional[float]:
        return (self.effective_updates / self.groups_sampled) if self.groups_sampled else None

    @property
    def mean_group_variance(self) -> Optional[float]:
        if not self.groups:
            return None
        return sum(group.stats.variance for group in self.groups) / len(self.groups)

    @property
    def mean_abs_advantage_admitted(self) -> Optional[float]:
        total = 0.0
        count = 0
        for group in self.groups:
            if group.stats.mean_abs_advantage_admitted is None or not group.stats.n_valid:
                continue
            total += group.stats.mean_abs_advantage_admitted * group.stats.n_valid
            count += group.stats.n_valid
        return (total / count) if count else None

    @property
    def rollouts_per_effective_update(self) -> Optional[float]:
        if not self.effective_updates:
            return None
        return self.rollouts_consumed / self.effective_updates

    @property
    def admission_rate(self) -> Optional[float]:
        return (self.admitted / self.records) if self.records else None

    @property
    def train_joint_success(self) -> Optional[float]:
        return (self.holds / self.records) if self.records else None

    @property
    def train_joint_success_given_admitted(self) -> Optional[float]:
        return (self.holds / self.admitted) if self.admitted else None

    @property
    def verdict_counts(self) -> Counter:
        return Counter(str(record.get("verdict")) for record in self.rollouts)

    # -- held-out ---------------------------------------------------------

    @property
    def eval_records(self) -> list[dict]:
        records = self.eval_payload.get("records")
        return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []

    @property
    def eval_draws(self) -> int:
        return len(self.eval_records)

    @property
    def eval_admitted(self) -> int:
        return sum(1 for record in self.eval_records if record.get("admitted"))

    @property
    def eval_holds(self) -> int:
        return sum(1 for record in self.eval_records if str(record.get("verdict")) == "HOLDS")

    @property
    def eval_present(self) -> bool:
        return bool(self.eval_records)

    def held_out_joint(self) -> tuple[Optional[float], tuple[float, float]]:
        if not self.eval_draws:
            return None, (0.0, 0.0)
        rate = self.eval_holds / self.eval_draws
        return rate, wilson_ci(self.eval_holds, self.eval_draws)

    def held_out_admission(self) -> tuple[Optional[float], tuple[float, float]]:
        if not self.eval_draws:
            return None, (0.0, 0.0)
        rate = self.eval_admitted / self.eval_draws
        return rate, wilson_ci(self.eval_admitted, self.eval_draws)

    def eval_per_task(self) -> list[dict]:
        buckets: dict[str, list[dict]] = {}
        for record in self.eval_records:
            buckets.setdefault(str(record.get("task_id")), []).append(record)
        rows = []
        for task_id in sorted(buckets):
            records = buckets[task_id]
            admitted = sum(1 for record in records if record.get("admitted"))
            holds = sum(1 for record in records if str(record.get("verdict")) == "HOLDS")
            rows.append(
                {
                    "task_id": task_id,
                    "draws": len(records),
                    "admitted": admitted,
                    "holds": holds,
                    "admission_rate": (admitted / len(records)) if records else None,
                    "joint_success": (holds / len(records)) if records else None,
                }
            )
        return rows

    # -- budget -----------------------------------------------------------

    @property
    def target_effective_updates(self) -> int:
        budget = self.metrics.get("budget")
        if isinstance(budget, dict) and isinstance(budget.get("target_effective_updates"), int):
            return budget["target_effective_updates"]
        return TARGET_EFFECTIVE_UPDATES

    @property
    def max_groups(self) -> int:
        budget = self.metrics.get("budget")
        if isinstance(budget, dict) and isinstance(budget.get("max_groups"), int):
            return budget["max_groups"]
        return MAX_GROUPS

    @property
    def shortfall(self) -> int:
        return max(0, self.target_effective_updates - self.effective_updates)

    @property
    def stop_reason(self) -> str:
        return str(self.metrics.get("stop_reason") or "unknown")

    @property
    def status(self) -> str:
        return str(self.metrics.get("status") or ("missing-metrics" if not self.metrics else "unknown"))


def _build_groups(arm: str, rollouts: list[dict], problems: list[str]) -> list[GroupView]:
    """Rebuild the groups from the per-completion records and re-treat them."""

    buckets: dict[int, list[dict]] = {}
    for record in rollouts:
        raw_index = record.get("group_index")
        if not isinstance(raw_index, int):
            problems.append(f"record without an integer group_index: {record.get('group_id')!r}")
            continue
        buckets.setdefault(raw_index, []).append(record)

    groups: list[GroupView] = []
    for group_index in sorted(buckets):
        records = sorted(buckets[group_index], key=lambda item: int(item.get("sample_index") or 0))
        task_ids = {str(record.get("task_id")) for record in records}
        if len(task_ids) > 1:
            problems.append(f"group {group_index} mixes tasks {sorted(task_ids)}")
        rewards = [float(record.get("reward") or 0.0) for record in records]
        valid = [bool(record.get("admitted")) for record in records]
        stats = grpo.group_stats(rewards, valid, arm)
        groups.append(
            GroupView(
                group_index=group_index,
                task_id=next(iter(task_ids)) if task_ids else "unknown",
                n=len(records),
                rewards=rewards,
                valid=valid,
                stats=stats,
                effective=stats.effective,
            )
        )
    return groups


def _cross_check(analysis: ArmAnalysis) -> None:
    """Compare the metrics file's own counts with the recomputation."""

    metrics = analysis.metrics
    if not metrics:
        return
    if metrics.get("arm") not in (None, analysis.arm):
        analysis.discrepancies.append(
            f"metrics file says arm={metrics.get('arm')!r}, this file is {analysis.arm!r}"
        )

    counts = metrics.get("counts") or {}
    if isinstance(counts, dict):
        errored = counts.get("groups_errored") or 0
        pairs = [
            ("groups_sampled", analysis.groups_sampled),
            ("effective_updates", analysis.effective_updates),
            ("groups_skipped", analysis.groups_skipped),
            ("groups_zero_variance", analysis.groups_zero_variance),
            ("rollouts_consumed", analysis.rollouts_consumed),
            ("records_written", analysis.records),
            ("admitted", analysis.admitted),
            ("holds", analysis.holds),
        ]
        if errored:
            # A group that raised before producing completions is logged in the
            # metrics file and cannot appear in the rollouts.  Those three counts
            # are expected to differ by exactly `errored`, so they are not
            # reported as disagreements.
            analysis.problems.append(
                f"{errored} group(s) raised before producing completions; group/rollout counts "
                "are compared against the non-errored remainder by the reader, not flagged below"
            )
            pairs = [pair for pair in pairs if pair[0] not in
                     ("groups_sampled", "rollouts_consumed", "records_written", "admitted", "holds")]
        for key, recomputed in pairs:
            logged = counts.get(key)
            if isinstance(logged, int) and logged != recomputed:
                analysis.discrepancies.append(
                    f"counts.{key}: metrics says {logged}, rollouts recompute to {recomputed}"
                )

    logged_groups = metrics.get("groups")
    if isinstance(logged_groups, list) and logged_groups:
        by_index = {
            int(item["group_index"]): item
            for item in logged_groups
            if isinstance(item, dict) and isinstance(item.get("group_index"), int)
        }
        for group in analysis.groups:
            logged = by_index.get(group.group_index)
            if logged is None:
                analysis.discrepancies.append(
                    f"group {group.group_index}: present in the rollouts, absent from metrics"
                )
                continue
            if bool(logged.get("effective")) != group.effective:
                analysis.discrepancies.append(
                    f"group {group.group_index}: effective {logged.get('effective')!r} in metrics, "
                    f"recomputed {group.effective!r}"
                )

    for record in analysis.rollouts:
        if record.get("arm") not in (None, analysis.arm):
            analysis.discrepancies.append(
                f"rollout arm {record.get('arm')!r} does not match {analysis.arm!r}"
            )
            break


def analyse_arm(
    arm: str,
    base: Path,
    overrides: dict[str, Optional[str]],
    seed: Optional[int] = None,
) -> ArmAnalysis:
    rollouts_path = Path(overrides["rollouts"] or base / f"rollouts_{arm}.jsonl")
    metrics_path = Path(overrides["metrics"] or base / f"metrics_{arm}.json")
    eval_path = Path(overrides["eval"] or base / f"eval_{arm}.json")

    analysis = ArmAnalysis(
        arm=arm,
        rollouts_path=rollouts_path,
        metrics_path=metrics_path,
        eval_path=eval_path,
        seed=seed,
    )

    rollouts, problems = load_jsonl(rollouts_path)
    analysis.rollouts = rollouts
    analysis.problems.extend(problems)
    analysis.metrics = load_json(metrics_path)
    analysis.eval_payload = load_json(eval_path)
    if not metrics_path.exists():
        analysis.problems.append(f"{metrics_path}: not found")
    if not eval_path.exists():
        analysis.problems.append(f"{eval_path}: not found")

    analysis.groups = _build_groups(arm, rollouts, analysis.problems)
    _cross_check(analysis)

    sizes = {group.n for group in analysis.groups}
    if len(sizes) > 1:
        analysis.problems.append(f"groups have unequal sizes: {sorted(sizes)}")

    return analysis


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """One pre-registered comparison, ``arm_a - arm_b``.

    The v1 reading is the ``(masked, control)`` instance of it: the ``masked_*``
    aliases are kept so a v1 caller keeps working unchanged.
    """

    verdict: str
    arm_a: str
    arm_b: str
    rate_a: Optional[float]
    rate_b: Optional[float]
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    diff: Optional[float]
    threshold_met: bool
    non_overlapping: bool
    direction: str

    @property
    def label(self) -> str:
        return f"{self.arm_a} vs {self.arm_b}"

    @property
    def readable(self) -> str:
        if self.verdict == "not-established":
            return "NOT ESTABLISHED (no held-out data)"
        if self.verdict == "no-measurable-difference":
            return "NO MEASURABLE DIFFERENCE"
        if self.verdict.endswith("-better"):
            winner = self.verdict[: -len("-better")]
            return f"{winner.upper()} BETTER"
        return self.verdict.upper()

    def rate_for(self, arm: str) -> Optional[float]:
        if arm == self.arm_a:
            return self.rate_a
        if arm == self.arm_b:
            return self.rate_b
        return None

    def ci_for(self, arm: str) -> tuple[float, float]:
        if arm == self.arm_a:
            return self.ci_a
        if arm == self.arm_b:
            return self.ci_b
        return (0.0, 0.0)

    # -- v1-compatible aliases (masked is the treatment, control the reference)

    @property
    def masked_rate(self) -> Optional[float]:
        return self.rate_for("masked")

    @property
    def control_rate(self) -> Optional[float]:
        return self.rate_for("control")

    @property
    def masked_ci(self) -> tuple[float, float]:
        return self.ci_for("masked")

    @property
    def control_ci(self) -> tuple[float, float]:
        return self.ci_for("control")


def decide_pair(arm_a: ArmAnalysis, arm_b: ArmAnalysis) -> Decision:
    """Apply the pre-committed rule to one pair's held-out joint success.

    ``arm_a`` is the treatment and ``arm_b`` the reference: the difference is
    ``a - b``.  The arm named ``<x>-better`` is the higher one, and only when the
    gap is at least ``OUTCOME_THRESHOLD`` **and** the two Wilson intervals do not
    overlap; otherwise the pair is ``no-measurable-difference``.  With no held-out
    data on either side the comparison is ``not-established`` (a missing
    measurement, not a null result).
    """

    rate_a, ci_a = arm_a.held_out_joint()
    rate_b, ci_b = arm_b.held_out_joint()
    common = {
        "arm_a": arm_a.arm,
        "arm_b": arm_b.arm,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "ci_a": ci_a,
        "ci_b": ci_b,
    }

    if rate_a is None or rate_b is None:
        return Decision(
            verdict="not-established",
            diff=None,
            threshold_met=False,
            non_overlapping=False,
            direction="unknown",
            **common,
        )

    diff = rate_a - rate_b
    a_above = ci_a[0] > ci_b[1]
    b_above = ci_b[0] > ci_a[1]
    non_overlapping = bool(a_above or b_above)
    threshold_met = abs(diff) >= OUTCOME_THRESHOLD

    if diff > 0 and threshold_met and a_above:
        verdict = f"{arm_a.arm}-better"
    elif diff < 0 and threshold_met and b_above:
        verdict = f"{arm_b.arm}-better"
    else:
        verdict = "no-measurable-difference"

    direction = arm_a.arm if diff > 0 else (arm_b.arm if diff < 0 else "tie")
    return Decision(
        verdict=verdict,
        diff=diff,
        threshold_met=threshold_met,
        non_overlapping=non_overlapping,
        direction=direction,
        **common,
    )


def decide(masked: ArmAnalysis, control: ArmAnalysis) -> Decision:
    """v1 spelling: the masked-vs-control comparison, as a :func:`decide_pair`."""

    return decide_pair(masked, control)


def paired_groups(first: ArmAnalysis, second: ArmAnalysis) -> list[tuple[GroupView, GroupView]]:
    """Groups sampled by both arms for the same prompt (same task, same index)."""

    by_index = {group.group_index: group for group in second.groups}
    pairs: list[tuple[GroupView, GroupView]] = []
    for group in first.groups:
        other = by_index.get(group.group_index)
        if other is not None and other.task_id == group.task_id:
            pairs.append((group, other))
    return pairs


# ---------------------------------------------------------------------------
# Seed runs
# ---------------------------------------------------------------------------

#: A directory holding one seed's artefacts: ``seed20261121/``, ``seed_20261121/``
#: or a bare ``20261121/``.
SEED_DIR_RE = re.compile(r"^(?:seed[-_]?)?(\d{4,})$", re.IGNORECASE)


@dataclass
class SeedRun:
    """One seed's artefact set: where its files live and what its seed is."""

    label: str
    seed: Optional[int]
    base: Path
    paths: dict[str, dict[str, Path]]


def _arm_paths(base: Path, arm: str, suffix: str = "") -> dict[str, Path]:
    return {
        "rollouts": base / f"rollouts_{arm}{suffix}.jsonl",
        "metrics": base / f"metrics_{arm}{suffix}.json",
        "eval": base / f"eval_{arm}{suffix}.json",
    }


def _run_present(paths: dict[str, dict[str, Path]]) -> bool:
    return any(path.exists() for arm in ARMS for path in paths[arm].values())


def _seed_of(paths: dict[str, dict[str, Path]]) -> Optional[int]:
    """The run's seed: from any arm's metrics file, else None."""

    for arm in ARMS:
        seed = load_json(paths[arm]["metrics"]).get("seed")
        if isinstance(seed, int):
            return int(seed)
    return None


def _explicit_paths(base: Path, overrides: dict[str, dict[str, Optional[str]]]) -> dict[str, dict[str, Path]]:
    return {
        arm: {
            key: Path(overrides[arm][key])
            if overrides[arm][key]
            else _arm_paths(base, arm)[key]
            for key in ("rollouts", "metrics", "eval")
        }
        for arm in ARMS
    }


def discover_seed_runs(
    base: Path,
    overrides: dict[str, dict[str, Optional[str]]],
) -> list[SeedRun]:
    """Find every seed run under ``base``, in a stable order.

    1. The flat ``rollouts_<arm>.jsonl`` / ``metrics_<arm>.json`` /
       ``eval_<arm>.json`` set is one run (the v1 layout).
    2. Every subdirectory named ``seed<number>`` (also ``seed_<number>``,
       ``seed-<number>`` or a bare number) with arm artefacts is another run.
    3. Flat files suffixed with a number -- ``metrics<...>_<arm>_<number>.json``
       -- are another run per number.

    A directory with no artefacts at all still yields one (empty) flat run, so
    the report keeps its shape and can say "not run" instead of crashing.  An
    explicit CLI override collapses discovery to that one run.
    """

    if any(value for arm in overrides.values() for value in arm.values()):
        paths = _explicit_paths(base, overrides)
        return [SeedRun(label=str(base), seed=_seed_of(paths), base=base, paths=paths)]

    runs: list[SeedRun] = []

    flat = {arm: _arm_paths(base, arm) for arm in ARMS}
    if _run_present(flat):
        runs.append(SeedRun(label=base.name or str(base), seed=_seed_of(flat), base=base, paths=flat))

    if base.is_dir():
        for child in sorted(path for path in base.iterdir() if path.is_dir()):
            match = SEED_DIR_RE.match(child.name)
            if not match:
                continue
            paths = {arm: _arm_paths(child, arm) for arm in ARMS}
            if not _run_present(paths):
                continue
            seed = _seed_of(paths)
            if seed is None:
                seed = int(match.group(1))
            runs.append(SeedRun(label=rel_path(child, base), seed=seed, base=child, paths=paths))

        suffixed: set[str] = set()
        for arm in ARMS:
            for pattern in (
                f"rollouts_{arm}_[0-9]*.jsonl",
                f"metrics_{arm}_[0-9]*.json",
                f"eval_{arm}_[0-9]*.json",
            ):
                for path in base.glob(pattern):
                    digits = path.stem.rsplit("_", 1)[-1]
                    if digits.isdigit():
                        suffixed.add(digits)
        for digits in sorted(suffixed):
            paths = {arm: _arm_paths(base, arm, f"_{digits}") for arm in ARMS}
            if not _run_present(paths):
                continue
            seed = _seed_of(paths)
            runs.append(
                SeedRun(
                    label=f"seed {digits}",
                    seed=seed if seed is not None else int(digits),
                    base=base,
                    paths=paths,
                )
            )

    if not runs:
        runs.append(SeedRun(label=base.name or str(base), seed=None, base=base, paths=flat))
    return runs


def choose_primary(runs: list[SeedRun], analyses_by_run: list[dict[str, ArmAnalysis]]) -> int:
    """Index of the run the main sections describe.

    The run with the most arms present wins (so a v2 directory with all three
    arms outranks a v1 directory with two), then the highest seed, then the
    earliest discovered run.  §7 always reports every run and the pooled row, so
    this only picks which run the per-group tables describe.
    """

    def score(index: int) -> tuple[int, int, int]:
        present = sum(1 for analysis in analyses_by_run[index].values() if analysis.ran)
        seed = runs[index].seed if runs[index].seed is not None else -1
        return (present, seed, -index)

    return max(range(len(runs)), key=score)


def pool_arm(analyses: list[ArmAnalysis]) -> dict:
    """Pool one arm across seed runs **by summing draws**, never by averaging rates.

    Only runs where the arm actually ran contribute.  The pooled interval is a
    Wilson interval on the pooled draws; it is not a seed-level variance estimate.
    """

    usable = [analysis for analysis in analyses if analysis.ran]
    draws = sum(analysis.eval_draws for analysis in usable)
    holds = sum(analysis.eval_holds for analysis in usable)
    admitted = sum(analysis.eval_admitted for analysis in usable)
    groups = sum(analysis.groups_sampled for analysis in usable)
    effective = sum(analysis.effective_updates for analysis in usable)
    rollouts = sum(analysis.rollouts_consumed for analysis in usable)
    return {
        "seeds": [analysis.seed for analysis in usable],
        "runs": len(usable),
        "draws": draws,
        "holds": holds,
        "admitted": admitted,
        "joint_success": (holds / draws) if draws else None,
        "joint_ci": wilson_ci(holds, draws) if draws else (0.0, 0.0),
        "admission_rate": (admitted / draws) if draws else None,
        "admission_ci": wilson_ci(admitted, draws) if draws else (0.0, 0.0),
        "groups": groups,
        "effective_updates": effective,
        "rollouts": rollouts,
        "rollouts_per_effective_update": (rollouts / effective) if effective else None,
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def status_text(analysis: ArmAnalysis) -> str:
    if not analysis.ran:
        return NOT_RUN
    if not (analysis.records or analysis.eval_present or analysis.metrics):
        return "empty"
    return "ran"


def not_run_row(arm: str, columns: int) -> list:
    """An explicit row for an arm that produced no artefacts.

    Every measure cell is an em dash and the first data cell says "not run": a
    missing arm is never rendered as ``0/0`` or as a zero.
    """

    return [arm, NOT_RUN] + ["—"] * (columns - 2)


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]} vs {pair[1]}"


def pair_verdict(decision: Decision, analyses: dict[str, ArmAnalysis]) -> str:
    if not (analyses[decision.arm_a].ran and analyses[decision.arm_b].ran):
        return "NOT RUN"
    return decision.readable


def cost_ratio(treatment: ArmAnalysis, reference: ArmAnalysis) -> Optional[float]:
    """Rollouts per effective update, treatment / reference."""

    if not (treatment.rollouts_per_effective_update and reference.rollouts_per_effective_update):
        return None
    return treatment.rollouts_per_effective_update / reference.rollouts_per_effective_update


def joint_cell(holds: int, draws: int, ci: tuple[float, float]) -> str:
    if draws <= 0:
        return "n/a"
    return f"{pct(holds, draws)} [{100.0 * ci[0]:.1f}–{100.0 * ci[1]:.1f}%]"


def section_header(
    base: Path,
    analyses: dict[str, ArmAnalysis],
    spec_status: str,
    primary: SeedRun,
    dry: bool,
) -> list[str]:
    lines = ["# Invalid-action masking in GRPO — recomputed report", ""]
    lines.append(
        "Generated by `analysis/summarize.py` from the rollout, metrics and eval artefacts listed "
        "below. Every count, rate, interval and verdict in this document is recomputed from those "
        "files; no endpoint number is entered by hand. Statistics: Wilson score interval (95%, "
        f"z = {Z_95}) and the exact two-sided sign test on discordant pairs, implemented in the "
        "standard library."
    )
    lines.append("")
    lines.append(
        "Three arms are preregistered (`docs/SPEC.md`): **control** (invalid reward 0.0, all-G "
        "baseline), **masked** (invalid removed, admitted-only baseline) and **penalty** "
        "(Amendment v2: invalid reward −1.0, all-G baseline). An arm whose artefacts are absent is "
        "reported as **not run** everywhere; it is never omitted and never rendered as a zero."
    )
    lines.append("")
    if dry:
        lines.append(
            "> **DRY RUN — NOT EVIDENCE.** These artefacts were produced by `--dry-run`, which draws "
            "canned completions from the task files. They exercise the plumbing; they say nothing "
            "about any model. Regenerate the report from a GPU run before reading a number."
        )
        lines.append("")
    if spec_status == "mismatch":
        lines.append(
            "> **PROTOCOL HASH MISMATCH.** `docs/SPEC.md` no longer hashes to the value recorded by "
            "the runners. Either the protocol was edited after the run (a deviation that must be "
            "reported) or the artefacts came from a different protocol. Read the numbers with that "
            "in mind."
        )
        lines.append("")
    primary_seed = f" (seed {primary.seed})" if primary.seed is not None else ""
    lines.append(
        f"Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. "
        f"Artefact directory: `{base}`. Primary run: `{primary.label}`{primary_seed}; §7 reports "
        "every seed run and the pooled estimate."
    )
    lines.append("")
    return lines


def section_headline(
    analyses: dict[str, ArmAnalysis],
    decisions: dict[tuple[str, str], Decision],
    primary: SeedRun,
) -> list[str]:
    """The SPEC's headline table: one row per arm, including arms that did not run."""

    lines = ["## Headline", ""]
    columns = 10
    rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            rows.append(not_run_row(arm, columns))
            continue
        joint_rate, joint_ci = analysis.held_out_joint()
        admission_rate, admission_ci = analysis.held_out_admission()
        rows.append(
            [
                arm,
                status_text(analysis),
                f"{analysis.effective_updates}/{analysis.target_effective_updates}",
                fmt_num(analysis.mean_group_variance, 4),
                fmt_num(analysis.mean_abs_advantage_admitted, 4),
                pct(analysis.admitted, analysis.records),
                pct(analysis.holds, analysis.records),
                joint_cell(analysis.eval_holds, analysis.eval_draws, joint_ci),
                f"{pct(analysis.eval_admitted, analysis.eval_draws)}"
                + (
                    f" [{100.0 * admission_ci[0]:.1f}–{100.0 * admission_ci[1]:.1f}%]"
                    if analysis.eval_draws
                    else ""
                ),
                fmt_num(analysis.rollouts_per_effective_update, 2),
            ]
        )
    lines.append(
        md_table(
            [
                "arm",
                "status",
                "effective updates / target",
                "mean group variance",
                "mean |adv| admitted",
                "train admission",
                "train joint success",
                "held-out joint success [Wilson 95%]",
                "held-out admission [Wilson 95%]",
                "rollouts / effective update",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"Primary run `{primary.label}`"
        + (f" (seed {primary.seed})" if primary.seed is not None else "")
        + ". **not run** means no artefacts were found for that arm; for an arm that ran, a "
        "`0/0 (n/a)` cell means it produced no records, which is a different statement."
    )
    lines.append("")
    lines.append(
        "**Verdicts under the pre-committed rule, one per preregistered pair:** "
        + " · ".join(
            f"{pair_label(pair)} — {pair_verdict(decisions[pair], analyses)}" for pair in PAIRS
        )
        + "."
    )
    lines.append("")
    return lines


def section_inputs(analyses: dict[str, ArmAnalysis], base: Path, problems: list[str]) -> list[str]:
    lines = ["## 1. Inputs", ""]
    columns = 8
    rows = []
    for arm, analysis in analyses.items():
        if not analysis.ran:
            rows.append(
                [
                    arm,
                    NOT_RUN,
                    code(rel_path(analysis.rollouts_path, base), 60),
                    "—",
                    "—",
                    "no",
                    "no",
                    "—",
                ]
            )
            continue
        rows.append(
            [
                arm,
                status_text(analysis),
                code(rel_path(analysis.rollouts_path, base), 60),
                analysis.records,
                analysis.groups_sampled,
                "yes" if analysis.metrics else "no",
                "yes" if analysis.eval_present else "no",
                analysis.eval_draws,
            ]
        )
    lines.append(
        md_table(
            ["arm", "status", "rollouts", "records", "groups", "metrics", "eval", "eval draws"], rows
        )
    )
    lines.append("")
    if problems:
        lines.append("> Input problems:")
        for problem in problems:
            lines.append(f"> - {problem}")
        lines.append("")
    return lines


def section_co_primary(
    analyses: dict[str, ArmAnalysis],
    decisions: dict[tuple[str, str], Decision],
) -> list[str]:
    lines = ["## 2. Co-primary endpoints — held-out", ""]
    lines.append(
        "The held-out tasks are the lexicographically last four of the corpus; the adapter is the "
        "final update (no checkpoint selection). Amendment v2 fixes the evaluation at 16 decodings "
        "per task (64 draws per adapter); the draw column below is each arm's actual count. Joint "
        "success counts `HOLDS` from `grade_obligation`; a draw the strict envelope rejected counts "
        "as a failure in the joint denominator, because an unadmitted patch does not exist. "
        "Admission is reported separately so a format win cannot masquerade as a repair win."
    )
    lines.append("")
    columns = 7
    rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            rows.append(not_run_row(arm, columns))
            continue
        joint_rate, joint_ci = analysis.held_out_joint()
        admission_rate, admission_ci = analysis.held_out_admission()
        rows.append(
            [
                arm,
                analysis.eval_draws,
                pct(analysis.eval_admitted, analysis.eval_draws),
                f"{100.0 * admission_ci[0]:.1f}%–{100.0 * admission_ci[1]:.1f}%"
                if analysis.eval_draws
                else "n/a",
                pct(analysis.eval_holds, analysis.eval_draws),
                f"{100.0 * joint_ci[0]:.1f}%–{100.0 * joint_ci[1]:.1f}%"
                if analysis.eval_draws
                else "n/a",
                fmt_rate(
                    (analysis.eval_holds / analysis.eval_admitted) if analysis.eval_admitted else None
                ),
            ]
        )
    lines.append(
        md_table(
            [
                "arm",
                "draws",
                "admitted",
                "admission (Wilson 95%)",
                "HOLDS",
                "joint success (Wilson 95%)",
                "HOLDS | admitted",
            ],
            rows,
        )
    )
    lines.append("")

    for pair in PAIRS:
        treatment, reference = pair
        decision = decisions[pair]
        if not (analyses[treatment].ran and analyses[reference].ran):
            missing = [
                arm for arm in pair if not analyses[arm].ran
            ]
            lines.append(
                f"**{pair_label(pair)}: not run.** No artefacts were found for "
                f"{', '.join(missing)}, so this comparison has no denominator. It is not a null "
                "result."
            )
        elif decision.diff is None:
            lines.append(
                f"**{pair_label(pair)}: not established.** At least one arm has no held-out "
                "evaluation, so the co-primary endpoints have no denominator and the decision rule "
                "cannot fire."
            )
        else:
            lines.append(
                f"**{pair_label(pair)}:** {treatment} − {reference} joint success "
                f"**{100.0 * decision.diff:+.1f} percentage points**. {treatment} interval "
                f"{100.0 * decision.ci_a[0]:.1f}%–{100.0 * decision.ci_a[1]:.1f}%, {reference} "
                f"interval {100.0 * decision.ci_b[0]:.1f}%–{100.0 * decision.ci_b[1]:.1f}%: "
                f"{'non-overlapping' if decision.non_overlapping else 'overlapping'}."
            )
        lines.append("")

    lines.append("### 2.1 Per-task breakdown")
    lines.append("")
    rows = []
    for arm in ARMS:
        if not analyses[arm].ran:
            continue
        for row in analyses[arm].eval_per_task():
            rows.append(
                [
                    arm,
                    row["task_id"],
                    row["draws"],
                    pct(row["admitted"], row["draws"]),
                    pct(row["holds"], row["draws"]),
                ]
            )
    lines.append(
        md_table(["arm", "held-out task", "draws", "admitted", "joint success"], rows)
        if rows
        else "_no eval records_"
    )
    lines.append("")
    return lines


def section_decision(
    decisions: dict[tuple[str, str], Decision],
    analyses: dict[str, ArmAnalysis],
) -> list[str]:
    lines = ["## 3. Decision rule (pre-committed, applied mechanically)", ""]
    lines.append(
        "The rule from `docs/SPEC.md`, applied to **every** preregistered pair: the arm with the "
        f"higher held-out joint success is better only if it exceeds the other by ≥ "
        f"{100.0 * OUTCOME_THRESHOLD:.0f} percentage points **and** the two Wilson 95% intervals do "
        "not overlap; the comparison in the other direction uses the same rule. Otherwise the pair "
        "is reported as no measurable difference, with both intervals and the rollout-cost ratio. A "
        "pair with a missing arm is NOT ESTABLISHED (a missing measurement, not a null result) and "
        "is never scored as zero."
    )
    lines.append("")

    rows = []
    for pair in PAIRS:
        decision = decisions[pair]
        treatment, reference = pair
        treatment_analysis = analyses[treatment]
        reference_analysis = analyses[reference]
        if not (treatment_analysis.ran and reference_analysis.ran):
            rows.append([decision.label, NOT_RUN, NOT_RUN, "—", "—", "—", "NOT RUN", "—"])
            continue
        ratio = cost_ratio(treatment_analysis, reference_analysis)
        if decision.diff is None:
            rows.append(
                [
                    decision.label,
                    "no eval",
                    "no eval",
                    "—",
                    "—",
                    "—",
                    "NOT ESTABLISHED",
                    f"{ratio:.2f}×" if ratio is not None else "n/a",
                ]
            )
            continue
        rows.append(
            [
                decision.label,
                joint_cell(treatment_analysis.eval_holds, treatment_analysis.eval_draws, decision.ci_a),
                joint_cell(reference_analysis.eval_holds, reference_analysis.eval_draws, decision.ci_b),
                f"{100.0 * decision.diff:+.1f} pp",
                "yes" if decision.threshold_met else "no",
                "yes" if decision.non_overlapping else "no",
                decision.readable,
                f"{ratio:.2f}×" if ratio is not None else "n/a",
            ]
        )
    lines.append(
        md_table(
            [
                "comparison (a − b)",
                "a joint success [Wilson 95%]",
                "b joint success [Wilson 95%]",
                "difference",
                "difference ≥ 12 pp",
                "intervals non-overlapping",
                "verdict",
                "cost ratio (a/b)",
            ],
            rows,
        )
    )
    lines.append("")

    missing_pairs = [pair for pair in PAIRS if not (analyses[pair[0]].ran and analyses[pair[1]].ran)]
    if missing_pairs:
        lines.append(
            "Pairs with a missing arm: "
            + ", ".join(pair_label(pair) for pair in missing_pairs)
            + " — each is NOT ESTABLISHED (no artefacts), and is not a null result."
        )
        lines.append("")

    for pair in PAIRS:
        treatment, reference = pair
        decision = decisions[pair]
        if not (analyses[treatment].ran and analyses[reference].ran) or decision.diff is None:
            continue
        checks = [
            [
                "difference ≥ threshold",
                f"{100.0 * abs(decision.diff):.1f} pp vs {100.0 * OUTCOME_THRESHOLD:.0f} pp",
                "yes" if decision.threshold_met else "no",
            ],
            [
                "intervals non-overlapping",
                f"{treatment} [{100.0 * decision.ci_a[0]:.1f}, {100.0 * decision.ci_a[1]:.1f}] vs "
                f"{reference} [{100.0 * decision.ci_b[0]:.1f}, {100.0 * decision.ci_b[1]:.1f}]",
                "yes" if decision.non_overlapping else "no",
            ],
            [
                "direction",
                f"{decision.direction} higher" if decision.direction != "tie" else "tie",
                "—",
            ],
        ]
        lines.append(f"**{decision.label}**")
        lines.append("")
        lines.append(md_table(["pre-committed condition", "observed", "met"], checks))
        lines.append("")
        if decision.verdict.endswith("-better"):
            lines.append(
                f"**Verdict: {decision.readable}.** For this pair the pre-committed reading is "
                f"**supported**: {decision.direction} exceeds the other arm by at least the "
                "pre-committed margin with non-overlapping intervals."
            )
        else:
            lines.append(
                f"**Verdict: {decision.readable}.** For this pair the pre-committed reading is "
                "**not established** at this n: the difference does not clear both conditions. This "
                "is a statement about the resolution of the held-out draws, not evidence that the "
                "treatments are equivalent."
            )
        lines.append("")

    lines.append("### 3.1 Cost, which the verdict does not weigh")
    lines.append("")
    columns = 9
    cost_rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            cost_rows.append(not_run_row(arm, columns))
            continue
        cost_rows.append(
            [
                arm,
                analysis.effective_updates,
                f"{analysis.effective_updates}/{analysis.target_effective_updates}",
                analysis.shortfall,
                analysis.groups_sampled,
                f"{analysis.groups_sampled}/{analysis.max_groups}",
                analysis.groups_skipped,
                analysis.rollouts_consumed,
                fmt_num(analysis.rollouts_per_effective_update, 2),
            ]
        )
    lines.append(
        md_table(
            [
                "arm",
                "effective updates",
                "of target",
                "shortfall",
                "groups sampled",
                "of cap",
                "skipped",
                "rollouts",
                "rollouts / effective update",
            ],
            cost_rows,
        )
    )
    lines.append("")

    for pair in PAIRS:
        treatment, reference = pair
        if not (analyses[treatment].ran and analyses[reference].ran):
            lines.append(
                f"Rollout-cost ratio ({treatment} / {reference}): not computable — at least one "
                "arm was not run."
            )
            continue
        ratio = cost_ratio(analyses[treatment], analyses[reference])
        if ratio is None:
            lines.append(
                f"Rollout-cost ratio ({treatment} / {reference}): cannot be formed — at least one "
                "arm recorded no effective update, so its rollouts-per-effective-update denominator "
                "is zero."
            )
        else:
            lines.append(
                f"**Rollout-cost ratio ({treatment} / {reference}): {ratio:.2f}×.** Budget parity is "
                "on effective gradient steps, so the arms are not equalised on rollouts; this ratio "
                "is the price of the treatment at that parity, and a cost difference with no outcome "
                "difference is itself a reportable result."
            )
        lines.append("")

    shortfalls = [
        (arm, analyses[arm].shortfall)
        for arm in ARMS
        if analyses[arm].ran and analyses[arm].shortfall
    ]
    if shortfalls:
        target = next(
            analyses[arm].target_effective_updates for arm in ARMS if analyses[arm].ran
        )
        lines.append(
            f"Shortfall against the {target}-update budget: "
            + ", ".join(f"{arm} {shortfall}" for arm, shortfall in shortfalls)
            + ". An arm that stopped at the group cap has less than the preregistered dose and must "
            "not be read as a completed arm."
        )
        lines.append("")
    return lines


def section_secondary(analyses: dict[str, ArmAnalysis]) -> list[str]:
    lines = ["## 4. Secondary endpoints — mechanism and cost", ""]
    lines.append(
        "The load-bearing mechanistic quantity is the **effective update**: a group whose arm "
        "treatment yields a nonzero-variance baseline, i.e. a group that actually produced a "
        "gradient. `--arm` decides the baseline population: control and penalty normalise over all "
        "G samples, masked over the admitted ones; the reward vector differs between control "
        "(invalid 0.0) and penalty (invalid −1.0). The effective-group rate is a descriptive "
        "fraction of the groups this run sampled; the Wilson interval is shown but is only "
        "indicative, because successive groups are not i.i.d. draws from a fixed population -- the "
        "policy changes after every effective update."
    )
    lines.append("")
    columns = 12
    rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            rows.append(not_run_row(arm, columns))
            continue
        rows.append(
            [
                arm,
                status_text(analysis),
                f"{analysis.effective_updates}/{analysis.groups_sampled}",
                fmt_rate(analysis.effective_group_rate),
                ci_text(analysis.effective_updates, analysis.groups_sampled),
                pct(analysis.groups_skipped, analysis.groups_sampled),
                pct(analysis.groups_zero_variance, analysis.groups_sampled),
                fmt_num(analysis.mean_group_variance, 4),
                fmt_num(analysis.mean_abs_advantage_admitted, 4),
                pct(analysis.admitted, analysis.records),
                pct(analysis.holds, analysis.records),
                pct(analysis.holds, analysis.admitted),
            ]
        )
    lines.append(
        md_table(
            [
                "arm",
                "status",
                "effective / groups",
                "effective-group rate",
                "Wilson 95%",
                "skipped",
                "zero-variance",
                "mean group variance",
                "mean |adv| admitted",
                "train admission",
                "train joint success",
                "train HOLDS | admitted",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Reading the columns: *skipped* is the masked arm's fewer-than-two-admitted case (the "
        "protocol's skip; no other arm skips); *zero-variance* is every group that produced no "
        "gradient, skipped ones included; *mean |adv| admitted* is the average absolute advantage "
        "carried by admitted samples, which is where masking's trade-off shows up -- masking removes "
        "format pressure but also removes contrast, and penalty increases the spread by pushing "
        "invalid samples down. Train success counts every rollout in the denominator, so an "
        "unadmitted completion is a failure there."
    )
    lines.append("")
    return lines


def section_paired(analyses: dict[str, ArmAnalysis]) -> list[str]:
    lines = ["## 5. Paired per-group variance (same prompts in each arm pair)", ""]
    lines.append(
        "Every arm rotates the same 8 train tasks on the same group index, so group *k* asks the same "
        "prompt in every arm. The pairing is on prompts, not on completions: arms sample different "
        "completions, so sampling noise is not cancelled -- only the prompt is held fixed."
    )
    lines.append("")
    columns = 8
    rows = []
    any_pairs = False
    for pair in PAIRS:
        treatment, reference = pair
        if not (analyses[treatment].ran and analyses[reference].ran):
            rows.append([pair_label(pair), NOT_RUN] + ["—"] * (columns - 2))
            continue
        pairs = paired_groups(analyses[treatment], analyses[reference])
        if not pairs:
            rows.append([pair_label(pair), 0, "—", "—", "—", "n/a", "n/a", "n/a"])
            continue
        any_pairs = True
        higher_treatment = sum(
            1 for treatment_group, reference_group in pairs
            if treatment_group.stats.variance > reference_group.stats.variance
        )
        higher_reference = sum(
            1 for treatment_group, reference_group in pairs
            if reference_group.stats.variance > treatment_group.stats.variance
        )
        ties = len(pairs) - higher_treatment - higher_reference
        mean_treatment = sum(group.stats.variance for group, _ in pairs) / len(pairs)
        mean_reference = sum(group.stats.variance for _, group in pairs) / len(pairs)
        p_value = paired_sign_test_p(higher_reference, higher_treatment)
        rows.append(
            [
                pair_label(pair),
                len(pairs),
                higher_treatment,
                higher_reference,
                ties,
                fmt_num(mean_treatment, 4),
                fmt_num(mean_reference, 4),
                fmt_p(p_value),
            ]
        )
    lines.append(
        md_table(
            [
                "comparison",
                "paired groups",
                "a var > b var",
                "b var > a var",
                "ties",
                "mean variance (a)",
                "mean variance (b)",
                "sign-test p",
            ],
            rows,
        )
    )
    lines.append("")
    if not any_pairs:
        lines.append(
            "_No pair of arms sampled the same group index, so nothing can be paired; pairs whose "
            "arm was not run are marked above._"
        )
        lines.append("")
    return lines


def section_counts(analyses: dict[str, ArmAnalysis]) -> list[str]:
    lines = ["## 6. Counts, denominators and the audit trail", ""]
    lines.append(
        "Every rollout count is reported, including failures and skipped groups, so a rate can "
        "always be traced back to its denominator."
    )
    lines.append("")
    columns = 9
    rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            rows.append(not_run_row(arm, columns))
            continue
        counts = analysis.verdict_counts
        rows.append(
            [
                arm,
                analysis.groups_sampled,
                analysis.records,
                analysis.admitted,
                counts.get("HOLDS", 0),
                counts.get("VIOLATED", 0),
                counts.get("UNKNOWN", 0),
                analysis.groups_skipped,
                analysis.groups_zero_variance,
            ]
        )
    lines.append(
        md_table(
            ["arm", "groups", "rollouts", "admitted", "HOLDS", "VIOLATED", "UNKNOWN", "skipped",
             "zero-variance"],
            rows,
        )
    )
    lines.append("")

    lines.append("### 6.1 Stop reasons and budget")
    lines.append("")
    columns = 9
    rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            rows.append(not_run_row(arm, columns))
            continue
        rows.append(
            [
                arm,
                analysis.status,
                analysis.stop_reason,
                analysis.metrics.get("abort_reason") or "—",
                analysis.effective_updates,
                analysis.target_effective_updates,
                analysis.groups_sampled,
                analysis.max_groups,
                fmt_num(analysis.metrics.get("wall_time_s"), 1),
            ]
        )
    lines.append(
        md_table(
            ["arm", "status", "stop reason", "abort reason", "effective", "target", "groups", "cap",
             "wall time (s)"],
            rows,
        )
    )
    lines.append("")

    lines.append("### 6.2 Integrity checks (recomputation vs the metrics file)")
    lines.append("")
    discrepancies = [
        f"{arm}: {problem}"
        for arm in ARMS
        for problem in analyses[arm].discrepancies
    ]
    ran_problems = [
        f"{arm}: {problem}"
        for arm in ARMS
        if analyses[arm].ran
        for problem in analyses[arm].problems
    ]
    not_run = [arm for arm in ARMS if not analyses[arm].ran]
    reported = discrepancies + ran_problems
    if reported:
        lines.append(
            "The report recomputes groups, effective flags and counts from the rollout JSONL and "
            "compares them with what the training loop logged. The following disagree:"
        )
        lines.append("")
        for problem in reported:
            lines.append(f"- {cell(problem, 220)}")
        lines.append("")
        lines.append(
            "> A disagreement is a harness defect or a mixed run. Fix the cause before quoting any "
            "endpoint above."
        )
    elif not any(analyses[arm].ran for arm in ARMS):
        lines.append(
            "No arm produced artefacts, so there is nothing to cross-check. That is a missing run, "
            "not a clean one."
        )
    else:
        lines.append(
            "The arms' recomputed group counts, effective flags, admission counts and hold counts "
            "agree with the metrics files, and every input named in §1 that exists is consistent. No "
            "disagreement was found."
        )
    lines.append("")
    if not_run:
        lines.append(
            "Arms not run: " + ", ".join(not_run) + " — no artefacts were found, so they are reported "
            "as not run rather than as zero."
        )
        lines.append("")

    lines.append("### 6.3 Per-group log")
    lines.append("")
    rows = []
    for arm in ARMS:
        for group in analyses[arm].groups:
            rows.append(
                [
                    arm,
                    group.group_index,
                    group.task_id,
                    "yes" if group.effective else "no",
                    "yes" if group.skipped else "no",
                    fmt_num(group.stats.mean, 3),
                    fmt_num(group.stats.std, 4),
                    group.stats.n_valid,
                    fmt_num(group.stats.mean_abs_advantage_admitted, 4),
                ]
            )
    if rows:
        lines.append(
            md_table(
                ["arm", "group", "task", "effective", "skipped", "mean reward", "group std",
                 "admitted", "mean |adv| admitted"],
                rows,
            )
        )
    else:
        lines.append("_no groups_")
    lines.append("")
    return lines


def section_seeds(
    runs: list[SeedRun],
    analyses_by_run: list[dict[str, ArmAnalysis]],
) -> list[str]:
    lines = ["## 7. Seeds — per-seed rows and the pooled estimate", ""]
    lines.append(
        "Seed runs are detected from the metrics files: the flat `*_<arm>` set is one run, and each "
        "`seed<number>/` subdirectory (or `*_<arm>_<number>` suffix) is another. Per-seed rows are "
        "printed next to the pooled row. A pooled proportion **sums draws**; it never averages "
        "per-seed rates. Pooling does not create replication: seeds are few, and the counts are not "
        "equal across arms (the v1 seed covers control and masked only), so the pooled interval is a "
        "binomial interval on the pooled draws, not a seed-level variance estimate."
    )
    lines.append("")
    if len(runs) == 1:
        single = runs[0]
        lines.append(
            "Only one seed run was found (`"
            + single.label
            + "`"
            + (f", seed {single.seed}" if single.seed is not None else "")
            + "), so a pooled row is undefined; its rows are shown so that the absence of a second "
            "seed is visible rather than implied."
        )
        lines.append("")

    columns = 11
    rows = []
    for run, analyses in zip(runs, analyses_by_run):
        label = run.label + (f" (seed {run.seed})" if run.seed is not None else "")
        for arm in ARMS:
            analysis = analyses[arm]
            if not analysis.ran:
                rows.append([label, arm, NOT_RUN] + ["—"] * (columns - 3))
                continue
            joint_rate, joint_ci = analysis.held_out_joint()
            admission_rate, admission_ci = analysis.held_out_admission()
            rows.append(
                [
                    label,
                    arm,
                    status_text(analysis),
                    analysis.groups_sampled,
                    analysis.effective_updates,
                    analysis.rollouts_consumed,
                    fmt_num(analysis.rollouts_per_effective_update, 2),
                    analysis.eval_draws,
                    analysis.eval_holds,
                    joint_cell(analysis.eval_holds, analysis.eval_draws, joint_ci),
                    f"{pct(analysis.eval_admitted, analysis.eval_draws)}"
                    + (
                        f" [{100.0 * admission_ci[0]:.1f}–{100.0 * admission_ci[1]:.1f}%]"
                        if analysis.eval_draws
                        else ""
                    ),
                ]
            )

    if len(runs) > 1:
        for arm in ARMS:
            pooled = pool_arm([analyses[arm] for analyses in analyses_by_run])
            if not pooled["runs"]:
                rows.append([f"pooled ({len(runs)} seeds)", arm, NOT_RUN] + ["—"] * (columns - 3))
                continue
            rows.append(
                [
                    f"pooled ({pooled['runs']} seed{'s' if pooled['runs'] != 1 else ''})",
                    arm,
                    "pooled",
                    pooled["groups"],
                    pooled["effective_updates"],
                    pooled["rollouts"],
                    fmt_num(pooled["rollouts_per_effective_update"], 2),
                    pooled["draws"],
                    pooled["holds"],
                    joint_cell(pooled["holds"], pooled["draws"], pooled["joint_ci"]),
                    f"{pct(pooled['admitted'], pooled['draws'])}"
                    + (
                        f" [{100.0 * pooled['admission_ci'][0]:.1f}–"
                        f"{100.0 * pooled['admission_ci'][1]:.1f}%]"
                        if pooled["draws"]
                        else ""
                    ),
                ]
            )
        lines.append("")
        lines.append(
            "Pooled rows sum each arm's runs that exist; an arm with fewer seeds has fewer draws in "
            "its pooled row, which is why the draw counts are printed rather than assumed equal."
        )
        lines.append("")

    lines.append(
        md_table(
            [
                "seed run",
                "arm",
                "status",
                "groups",
                "effective updates",
                "rollouts",
                "rollouts / effective update",
                "held-out draws",
                "held-out HOLDS",
                "joint success [Wilson 95%]",
                "admission [Wilson 95%]",
            ],
            rows,
        )
    )
    lines.append("")
    return lines


def section_limits(analyses: dict[str, ArmAnalysis]) -> list[str]:
    dry = any(
        analysis.metrics.get("dry_run") or analysis.eval_payload.get("dry_run")
        for analysis in analyses.values()
    )
    lines = ["## What this does not show", ""]
    lines.append(
        "- **It does not establish equivalence when the verdict is 'no measurable difference'.** "
        "The rule can only fire on a large gap: with 64 draws per arm under Amendment v2 "
        "(32 under v1), two intervals overlap for any moderate difference, so the binding constraint "
        "is non-overlap, not the "
        f"{100.0 * OUTCOME_THRESHOLD:.0f}-point threshold. A null here is 'this design could not "
        "see it at this n', not 'the treatments are the same'."
    )
    lines.append(
        "- **It is not a capability claim.** Eight train tasks, four held-out tasks and a 1.5B "
        "policy measure whether a baseline treatment changes the gradient, not how good the model "
        "is. A null on held-out success does not refute the mechanism, and a win on held-out "
        "success would not generalise past this corpus."
    )
    lines.append(
        "- **Few seeds, and not equal across arms.** Amendment v2 adds one seed for all three arms "
        "on top of the v1 seed, which covers control and masked only. The pooled row sums draws and "
        "is reported alongside the per-seed rows; it is not a hierarchical model and it does not "
        "turn draws into independent replication. The paired per-group variance comparison is "
        "within-run and paired on prompts only."
    )
    lines.append(
        "- **The effective-update counts are not a Bernoulli sample.** The policy changes after "
        "every effective update, so successive groups are not i.i.d.; the Wilson interval on the "
        "effective-group rate is descriptive, not a confidence statement about a population."
    )
    lines.append(
        "- **Budget parity is on effective steps, not compute.** Every arm is asked for the same "
        "target number of effective updates under the same group cap (24/96 in the original "
        "protocol, 8/40 after the v1 deviation; the manifest records which), but the arms spend "
        "different numbers of rollouts to get there. The cost ratio is reported rather than "
        "equalised, so a cheap arm and an expensive arm are not compared at equal compute."
    )
    lines.append(
        "- **Admission is reported separately from HOLDS for a reason.** A model can learn the "
        "strict JSON envelope without learning the repair, and that would show up as a held-out "
        "admission win with no joint-success win. Neither endpoint alone identifies which happened."
    )
    lines.append(
        "- **The penalty arm changes one number, not the whole reward.** A rejected completion is "
        "worth −1.0 instead of 0.0; a rejected action and a wrong-but-valid repair are otherwise "
        "scored by the same rules. A difference between penalty and control is a difference in the "
        "valuation of format failure, not a different task or a different grader."
    )
    lines.append(
        "- **No PPO clipping, no KL term.** The importance ratio is 1 by construction (one step "
        "per group), so this isolates the baseline treatment; it is not a full GRPO recipe and its "
        "numbers should not be compared with one."
    )
    lines.append(
        "- **`HOLDS` means 'not caught by this grader', not 'correct'.** The obligation grader is a "
        "program with its own blind spots; it rejects security failures, benign-behaviour loss, "
        "no-ops, trivial wrappers and out-of-scope names, and nothing else."
    )
    lines.append(
        "- **The final update only is evaluated.** There is no checkpoint selection by design, so a "
        "transient improvement earlier in a run is invisible, and a late collapse is fully weighted."
    )
    lines.append(
        "- **Format learning and repair learning are not separated within a run.** The train-side "
        "admission and joint-success rates move together when the model first learns the envelope; "
        "the report cannot attribute a mid-run jump to either."
    )
    if dry:
        lines.append(
            "- **These particular artefacts are a dry run.** The completions are canned strings "
            "derived from the task files, so every number above describes the harness, not a model. "
            "The report proves the plumbing; it is not a result."
        )
    lines.append("")
    return lines


def section_provenance(analyses: dict[str, ArmAnalysis], base: Path, spec_status: str) -> list[str]:
    lines = ["## Provenance", ""]
    lines.append(
        "The manifest fields below come from the metrics/eval files written by the runners. They "
        "are what makes a run identifiable; the report never edits them."
    )
    lines.append("")
    columns = 12
    rows = []
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            rows.append([arm, NOT_RUN] + ["—"] * (columns - 2))
            continue
        metrics = analysis.metrics
        model = metrics.get("model") or analysis.eval_payload.get("model") or {}
        budget = metrics.get("budget") or {}
        decoding = metrics.get("decoding") or {}
        seed = analysis.seed if analysis.seed is not None else metrics.get("seed")
        rows.append(
            [
                arm,
                seed if seed is not None else "unknown",
                code(rel_path(analysis.rollouts_path, ROOT), 48),
                metrics.get("git_sha") or analysis.eval_payload.get("git_sha") or "unknown",
                metrics.get("utc_start") or "unknown",
                model.get("model_id") or "unknown",
                model.get("requested_revision") or "unknown",
                model.get("resolved_revision") or "unknown",
                model.get("gpu") or "—",
                f"{decoding.get('temperature')}/{decoding.get('top_p')}/{decoding.get('max_new_tokens')}"
                if decoding
                else "—",
                budget.get("target_effective_updates", "—"),
                budget.get("max_groups", "—"),
            ]
        )
    lines.append(
        md_table(
            ["arm", "seed", "rollouts", "git sha", "utc start", "model", "revision requested",
             "revision resolved", "gpu", "temp/top_p/max_tok", "target eff.", "group cap"],
            rows,
        )
    )
    lines.append("")
    if spec_status == "match":
        lines.append("`docs/SPEC.md` hashes to the value recorded by the runners: the protocol was "
                     "not edited after the run.")
    elif spec_status == "mismatch":
        lines.append("`docs/SPEC.md` does **not** hash to the value recorded by the runners: the "
                     "protocol changed after the run, or the artefacts predate it.")
    else:
        lines.append("The protocol hash could not be checked (no recorded hash in the artefacts).")
    lines.append("")
    lines.append(
        "Recompute this report with `python analysis/summarize.py "
        f"--dir {base} --out {base / 'REPORT.md'}`."
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the masking experiment report from its artefacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read results/dryrun/ instead of results/ (no effect on --out)",
    )
    parser.add_argument("--dir", default=None, help="artefact directory")
    parser.add_argument("--out", default=None, help="report path")
    for arm in ARMS:
        parser.add_argument(
            f"--rollouts-{arm}", default=None, help=f"override rollouts_{arm}.jsonl"
        )
    for arm in ARMS:
        parser.add_argument(
            f"--metrics-{arm}", default=None, help=f"override metrics_{arm}.json"
        )
    for arm in ARMS:
        parser.add_argument(f"--eval-{arm}", default=None, help=f"override eval_{arm}.json")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    base = Path(args.dir) if args.dir else (
        DEFAULT_RESULTS / DRYRUN_SUBDIR if args.dry_run else DEFAULT_RESULTS
    )
    out_path = Path(args.out) if args.out else base / "REPORT.md"

    overrides = {
        arm: {
            "rollouts": getattr(args, f"rollouts_{arm}"),
            "metrics": getattr(args, f"metrics_{arm}"),
            "eval": getattr(args, f"eval_{arm}"),
        }
        for arm in ARMS
    }

    runs = discover_seed_runs(base, overrides)
    analyses_by_run = [
        {
            arm: analyse_arm(arm, run.base, run.paths[arm], seed=run.seed)
            for arm in ARMS
        }
        for run in runs
    ]
    primary_index = choose_primary(runs, analyses_by_run)
    primary = runs[primary_index]
    analyses = analyses_by_run[primary_index]
    decisions = {
        pair: decide_pair(analyses[pair[0]], analyses[pair[1]]) for pair in PAIRS
    }

    recorded_hashes = {
        analysis.metrics.get("spec_sha256")
        for run_analyses in analyses_by_run
        for analysis in run_analyses.values()
        if analysis.metrics.get("spec_sha256")
    }
    try:
        current_hash = hashlib.sha256((ROOT / "docs" / "SPEC.md").read_bytes()).hexdigest()
    except OSError:
        current_hash = "unknown"

    if not recorded_hashes:
        spec_status = "unknown"
    elif current_hash in recorded_hashes:
        spec_status = "match"
    else:
        spec_status = "mismatch"

    problems: list[str] = []
    for run, run_analyses in zip(runs, analyses_by_run):
        prefix = "" if run is primary else f"[{run.label}] "
        for arm in ARMS:
            problems.extend(f"{prefix}{arm}: {problem}" for problem in run_analyses[arm].problems)

    dry = any(
        analysis.metrics.get("dry_run") or analysis.eval_payload.get("dry_run")
        for run_analyses in analyses_by_run
        for analysis in run_analyses.values()
    )

    lines: list[str] = []
    lines.extend(section_header(base, analyses, spec_status, primary, dry))
    lines.extend(section_headline(analyses, decisions, primary))
    lines.extend(section_inputs(analyses, base, problems))
    lines.extend(section_co_primary(analyses, decisions))
    lines.extend(section_decision(decisions, analyses))
    lines.extend(section_secondary(analyses))
    lines.extend(section_paired(analyses))
    lines.extend(section_counts(analyses))
    lines.extend(section_seeds(runs, analyses_by_run))
    lines.extend(section_limits(analyses))
    lines.extend(section_provenance(analyses, base, spec_status))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    verdicts = " · ".join(
        f"{pair_label(pair)}: {pair_verdict(decisions[pair], analyses)}" for pair in PAIRS
    )
    print(f"[report] wrote {out_path} — {verdicts}", file=sys.stderr)
    for arm in ARMS:
        analysis = analyses[arm]
        if not analysis.ran:
            print(f"[report] {arm}: not run", file=sys.stderr)
        else:
            joint_rate, joint_ci = analysis.held_out_joint()
            print(
                f"[report] {arm}: effective {analysis.effective_updates}"
                f"/{analysis.target_effective_updates}, held-out joint "
                f"{pct(analysis.eval_holds, analysis.eval_draws)}, "
                f"admission {pct(analysis.eval_admitted, analysis.eval_draws)}",
                file=sys.stderr,
            )
    if len(runs) > 1:
        print(f"[report] seed runs: {len(runs)} (pooled row in §7)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
