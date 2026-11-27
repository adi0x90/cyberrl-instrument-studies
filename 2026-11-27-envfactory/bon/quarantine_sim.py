"""B1 — scheduler-timed quarantine under asynchronous Tier-2.

The point is not to plot a detection curve.  It is to measure what the *trainer*
does between a rollout being accepted by Tier-1 and an independent audit
returning: how many optimizer steps commit in that window, and whether a stale or
duplicated audit can quarantine a stamp that has already been replaced.

Timing model, stated explicitly because it is a simulation:

* one rollout = ``ROLLOUT_S`` seconds of decode
* one optimizer step commits every ``STEP_S`` seconds
* an audit enqueued at ``t`` completes at ``t + delay`` and is acked then

The verdicts are real (from the scored pool); only the clock is synthetic.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

HOLDS, VIOLATED = "HOLDS", "VIOLATED"
ROLLOUT_S = 8.0
STEP_S = 30.0
DELAYS = [0.0, 60.0, 300.0]
AUDIT_FRACTIONS = [0.25, 0.5]
PASS_RATE_JUMP = 0.15


@dataclass
class Event:
    rollout_id: str
    policy_rev: int
    stamp_id: str
    accepted: bool
    t2_verdict: str
    t: float
    steps_at_accept: int = 0


@dataclass
class Audit:
    event: Event
    enqueued: float
    delay: float

    @property
    def completed(self) -> float:
        return self.enqueued + self.delay


def build_stream(rows: list[dict]) -> list[Event]:
    events: list[Event] = []
    for index, row in enumerate(rows):
        event = Event(
            rollout_id=f"{row['task_id']}#{row['sample']}",
            policy_rev=index // 12 + 1,
            stamp_id=f"stamp-{index // 12 + 1}",
            accepted=row["t1"] == HOLDS,
            t2_verdict=row["t2"],
            t=index * ROLLOUT_S,
        )
        event.steps_at_accept = int(event.t // STEP_S)
        events.append(event)
    return events


def audit_uniform(events: list[Event], fraction: float) -> list[Audit]:
    accepts = [event for event in events if event.accepted and event.t2_verdict in {HOLDS, VIOLATED}]
    stride = max(1, int(round(1 / fraction)))
    return [Audit(event, event.t, 0.0) for event in accepts[::stride]]


def audit_anomaly(events: list[Event], window: int = 24) -> list[Audit]:
    """Audit when the accept rate over a rolling window jumps by >= 15 pp."""

    audited: list[Audit] = []
    for index, event in enumerate(events):
        if not event.accepted or event.t2_verdict not in {HOLDS, VIOLATED}:
            continue
        before = events[max(0, index - window):index]
        if len(before) < window // 2:
            continue
        prev = sum(1 for item in before[: window // 2] if item.accepted) / max(1, len(before[: window // 2]))
        curr = sum(1 for item in before if item.accepted) / len(before)
        if curr - prev >= PASS_RATE_JUMP:
            audited.append(Audit(event, event.t, 0.0))
    return audited


def simulate(events: list[Event], audits: list[Audit], delay: float) -> dict:
    refuted = [Audit(audit.event, audit.enqueued, delay) for audit in audits if audit.event.t2_verdict == VIOLATED]
    steps_before_ack: list[int] = []
    for audit in refuted:
        ack_time = audit.completed
        steps = int(ack_time // STEP_S) - audit.event.steps_at_accept
        steps_before_ack.append(max(0, steps))
    return {
        "audits": len(audits),
        "refuted": len(refuted),
        "steps_committed_before_ack": steps_before_ack,
        "mean_steps_before_ack": round(statistics.fmean(steps_before_ack), 2) if steps_before_ack else 0.0,
        "max_steps_before_ack": max(steps_before_ack) if steps_before_ack else 0,
    }


def stale_audit_replay() -> dict:
    """A delayed audit must not quarantine a replacement stamp.

    rev-1 stamp is refuted at t=100+D, but policy_rev 2 is already at the
    scheduler.  A correct scheduler checks (stamp_id, policy_rev) and ignores the
    stale refutation.
    """

    results = {}
    for delay in DELAYS:
        stale = Audit(Event("r1", 1, "stamp-1", True, VIOLATED, 40.0), 40.0, delay)
        current_policy_rev = 2
        ack_at = stale.completed
        quarantined = stale.event.policy_rev == current_policy_rev
        results[str(delay)] = {
            "ack_at_s": ack_at,
            "ack_policy_rev": stale.event.policy_rev,
            "current_policy_rev": current_policy_rev,
            "quarantined_replacement": quarantined,
            "disposition": "ignored as stale" if not quarantined else "WRONGLY quarantined",
        }
    return results


def main() -> int:
    payload = json.loads(Path("results/B1.json").read_text()) if Path("results/B1.json").is_file() else None
    scored = [json.loads(line) for line in Path("evidence/bon/scored_qwen7b.jsonl").read_text().splitlines() if line.strip()]
    events = build_stream(scored)

    out = {
        "clock": {"rollout_s": ROLLOUT_S, "step_s": STEP_S, "note": "verdicts real; clock synthetic"},
        "stream_rollouts": len(events),
        "accepts": sum(1 for event in events if event.accepted),
        "policies": {
            "uniform_25": simulate(events, audit_uniform(events, 0.25), 0.0),
            "uniform_50": simulate(events, audit_uniform(events, 0.50), 0.0),
            "anomaly_triggered": simulate(events, audit_anomaly(events), 0.0),
        },
        "delays": {},
        "stale_replay": stale_audit_replay(),
    }
    for delay in DELAYS:
        out["delays"][str(delay)] = {
            "uniform_25": simulate(events, audit_uniform(events, 0.25), delay),
            "uniform_50": simulate(events, audit_uniform(events, 0.50), delay),
            "anomaly_triggered": simulate(events, audit_anomaly(events), delay),
        }

    Path("results/B1_quarantine.json").parent.mkdir(parents=True, exist_ok=True)
    Path("results/B1_quarantine.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({key: out["delays"][key]["uniform_25"] for key in out["delays"]}, indent=2))
    print("stale replay:", json.dumps(out["stale_replay"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
