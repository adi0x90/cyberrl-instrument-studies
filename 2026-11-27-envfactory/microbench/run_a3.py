"""A3 runner — gate, timings, concurrency, and the two unsupported cells.

Evidence contract: every row carries the target, the strategy, how many resets
were requested vs completed, and the failure count.  Targets that could not be
exercised are recorded as ``not_run`` with a reason, never silently omitted.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .reset_bench import (
    RESETS,
    TARGETS,
    concurrency_sweep,
    cost_per_committed_step,
    criu_redis_probe,
    qemu_kvm_probe,
    reset_share_threshold,
    state_equivalence,
    time_strategy,
)

STRATEGIES = ["s1_cold_run", "s2_warm_layers", "s3_overlay_teardown"]
WORKERS = [1, 4, 8, 16]
VPS_HOURLY_RATE = 0.952  # quoted for bench-sweep-32; [ESTIMATE — verify at booking]


def main() -> int:
    started = time.time()
    out = Path(os.environ.get("A3_OUT", "evidence/a3/a3_reset.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "tool": "microbench",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "resets_requested": RESETS,
        "targets": [],
        "not_run": [],
        "qemu": qemu_kvm_probe(),
        "criu": None,
        "economics": {},
        "limits": [
            "Reset latency is a property of image layers, service startup and state teardown; "
            "the injected pack source is not inside these images.",
            "KVM is absent on the benchmark host: hardware-accelerated VM snapshot restore "
            "was not attempted; software emulation is not a comparable reset path.",
        ],
    }

    for target in TARGETS:
        try:
            gate = state_equivalence(target)
        except Exception as exc:  # noqa: BLE001 - one broken target must not erase the others
            payload["targets"].append({"pack": target.pack, "image": target.image, "error": str(exc)[:500]})
            print(f"[a3] {target.pack}: gate error {exc}", flush=True)
            continue
        print(f"[a3] gate {target.pack}: equivalence={gate['state_equivalence']}", flush=True)
        entry = {"pack": target.pack, "image": target.image, "gate": gate, "timings": []}
        if not gate["state_equivalence"]:
            entry["skipped_timing"] = "state-equivalence gate failed"
            payload["targets"].append(entry)
            continue

        for strategy in STRATEGIES:
            row = time_strategy(target, strategy, RESETS)
            entry["timings"].append(row)
            print(
                f"[a3] {target.pack} {strategy}: n={row['completed']} "
                f"p50={row['p50_s']} p95={row['p95_s']}",
                flush=True,
            )
        entry["concurrency"] = concurrency_sweep(target, WORKERS)
        payload["targets"].append(entry)

    for pack, reason in [
        ("authlib-jwt-dupkey-canonicalization", "no long-running service; the pack is a library call path, so a service-state marker cannot be written"),
        ("fastmcp-canonical-authz-alias", "service image not built in this round; no digest-pinned container was qualified"),
        ("coremqtt-publish-varint-payloadlen", "library, not a service; no mutable service state to reset"),
    ]:
        payload["not_run"].append({"pack": pack, "reason": reason})

    if os.environ.get("A3_CRIU", "1") == "1":
        try:
            payload["criu"] = criu_redis_probe()
        except Exception as exc:  # noqa: BLE001 - a failed probe is evidence
            payload["criu"] = {"supported": False, "error": str(exc)[:400]}

    elapsed_hours = (time.time() - started) / 3600.0
    payload["economics"] = {
        "pool_pricing": cost_per_committed_step(
            gpu_hours=0.0,
            cpu_hours=round(elapsed_hours, 4),
            api_spend=0.0,
            committed_steps=0,
            gpu_rate=0.0,
            cpu_rate=VPS_HOURLY_RATE,
        ),
        "cpu_hours_this_run": round(elapsed_hours, 4),
        "cpu_rate_used": VPS_HOURLY_RATE,
        "note": "No GPU pool was engaged and no optimizer step was committed, so the unit price is undefined by construction.",
    }
    thresholds = []
    for entry in payload["targets"]:
        for row in entry.get("timings", []):
            if row.get("p50_s"):
                thresholds.append({"pack": entry["pack"], "strategy": row["strategy"],
                                   **reset_share_threshold(row["p50_s"], grade_s=0.05, decode_s=0.0)})
    payload["reset_share_thresholds"] = thresholds
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[a3] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
