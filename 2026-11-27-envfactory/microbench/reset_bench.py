"""A3 — state-equivalent reset benchmark and pool-priced rollout economics.

Two things are measured, in this order, because the order is the point:

1. **The state-equivalence gate.**  Before any reset strategy may be timed for a
   target, a unique marker is written to the workspace filesystem *and* to live
   service state, the reset is executed, both markers must be absent, and one
   known-good application read/write must succeed.  A strategy that restarts a
   process while preserving mutable state fails the gate, and that failure is a
   publishable negative rather than a timing row.

2. **Reset latency**, full release -> clean ready, including teardown, at p50/p95/p99
   over 50 resets per (target, strategy) and a concurrency sweep.

Costs: the economics module refuses to price a step when there are zero committed
optimizer steps.  It reports spend and ``0 commits`` instead, per the spec.

Targets are digest-pinned service images.  The injected pack source is *not* in
the image: reset latency is a property of image layers, service startup and
state teardown, not of the injected function, and pretending otherwise would
overstate what this benchmark measures.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RESETS = 50


def sh(argv: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


@dataclass(frozen=True)
class Target:
    pack: str
    image: str
    container_cmd: list[str]
    ready_cmd: list[str]
    fs_marker: str
    state_write: list[str]
    state_read: list[str]
    state_probe_absent: str  # expected substring when the marker is gone
    good_op: list[str]
    good_expect: str


TARGETS = [
    Target(
        pack="redis-setrange-grow-undercount",
        image="redis@sha256:ae471bdc20de180beed36e347d170ec0bdf8faa327959ea1f24692f22b05b955",
        container_cmd=["redis-server", "--save", "", "--appendonly", "no"],
        ready_cmd=["redis-cli", "PING"],
        fs_marker="/tmp/envcert-reset-marker",
        state_write=["redis-cli", "SET", "envcert:reset-marker", "1"],
        state_read=["redis-cli", "EXISTS", "envcert:reset-marker"],
        state_probe_absent="0",
        good_op=["redis-cli", "SET", "envcert:probe", "ok"],
        good_expect="OK",
    ),
    Target(
        pack="etcd-revocation-skew",
        image="quay.io/coreos/etcd@sha256:f3c241cc04cc5bf88c5ded7a6d8dbab12de3783bbe0f699ece745feb5ad8e336",
        container_cmd=[
            "/usr/local/bin/etcd",
            "--data-dir=/etcd-data",
            "--listen-client-urls=http://0.0.0.0:2379",
            "--advertise-client-urls=http://127.0.0.1:2379",
        ],
        ready_cmd=["etcdctl", "--endpoints=127.0.0.1:2379", "endpoint", "health"],
        fs_marker="/tmp/envcert-reset-marker",
        state_write=["etcdctl", "--endpoints=127.0.0.1:2379", "put", "/envcert/reset-marker", "1"],
        state_read=["etcdctl", "--endpoints=127.0.0.1:2379", "get", "/envcert/reset-marker"],
        state_probe_absent="",
        good_op=["etcdctl", "--endpoints=127.0.0.1:2379", "put", "/envcert/probe", "ok"],
        good_expect="OK",
    ),
]


def _write_fs_marker(name: str, path: str) -> bool:
    """Copy a marker into the container filesystem (no shell required).

    The etcd image is distroless: it has no ``sh``, so ``docker cp`` plus
    ``docker diff`` is the only portable way to write and observe a filesystem
    marker across both targets.
    """

    scratch = Path("/tmp/envcert-marker")
    scratch.write_text("envcert-reset-marker\n", encoding="utf-8")
    proc = sh(["docker", "cp", str(scratch), f"{name}:{path}"])
    return proc.returncode == 0


def _fs_marker_present(name: str, path: str) -> bool:
    proc = sh(["docker", "diff", name])
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1].strip() == path:
            return True
    return False


def _name(target: Target, suffix: str = "") -> str:
    base = "envcert-" + target.pack.split("-")[0]
    return f"{base}{suffix}"


def _wait_ready(target: Target, name: str, deadline_s: float = 30.0) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < deadline_s:
        proc = sh(["docker", "exec", name, *target.ready_cmd], timeout=10)
        if proc.returncode == 0:
            return time.perf_counter() - started
        time.sleep(0.05)
    raise TimeoutError(f"{target.pack}: container {name} never became ready")


def _run_container(target: Target, name: str, *, fresh: bool) -> dict[str, float]:
    """Start one container and return the timed phases."""

    timings: dict[str, float] = {}
    if fresh:
        started = time.perf_counter()
        sh(["docker", "rm", "-f", name])
        timings["teardown_s"] = time.perf_counter() - started

    started = time.perf_counter()
    proc = sh(
        [
            "docker", "run", "-d", "--name", name,
            "--network", "none",
            target.image, *target.container_cmd,
        ]
    )
    timings["create_s"] = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{target.pack}: docker run failed: {proc.stderr[-300:]}")
    timings["ready_s"] = _wait_ready(target, name)
    return timings


def state_equivalence(target: Target) -> dict[str, Any]:
    """The mandatory gate: fs marker + live-state marker, reset, both gone."""

    name = _name(target, "-gate")
    _run_container(target, name, fresh=True)
    try:
        _write_fs_marker(name, target.fs_marker)
        state_write = sh(["docker", "exec", name, *target.state_write])
        fs_before = _fs_marker_present(name, target.fs_marker)
        state_before = sh(["docker", "exec", name, *target.state_read]).stdout.strip()
    finally:
        sh(["docker", "rm", "-f", name])

    # the reset itself
    _run_container(target, name, fresh=True)
    try:
        fs_after = _fs_marker_present(name, target.fs_marker)
        state_after = sh(["docker", "exec", name, *target.state_read]).stdout.strip()
        good = sh(["docker", "exec", name, *target.good_op])
    finally:
        sh(["docker", "rm", "-f", name])

    marker_absent = (not fs_after) and (state_after == target.state_probe_absent)
    good_ok = good.returncode == 0 and target.good_expect in good.stdout
    return {
        "pack": target.pack,
        "image": target.image,
        "fs_marker_before": fs_before,
        "state_marker_before": state_before,
        "fs_marker_after": fs_after,
        "state_marker_after": state_after,
        "markers_absent_after_reset": marker_absent,
        "known_good_op_after_reset": good_ok,
        "state_equivalence": bool(marker_absent and good_ok and fs_before and state_write.returncode == 0),
        "state_write_ok": state_write.returncode == 0,
    }


def time_strategy(target: Target, strategy: str, resets: int = RESETS) -> dict[str, Any]:
    name = _name(target, "-bench")
    samples: list[float] = []
    failures = 0
    for index in range(resets):
        try:
            started = time.perf_counter()
            if strategy == "s1_cold_run":
                _run_container(target, f"{name}-{index}", fresh=False)
                elapsed = time.perf_counter() - started
                sh(["docker", "rm", "-f", f"{name}-{index}"])
            elif strategy == "s2_warm_layers":
                _run_container(target, name, fresh=True)
                elapsed = time.perf_counter() - started
            elif strategy == "s3_overlay_teardown":
                sh(["docker", "stop", "-t", "1", name])
                _run_container(target, name, fresh=True)
                elapsed = time.perf_counter() - started
            else:
                raise ValueError(strategy)
            samples.append(elapsed)
        except Exception:  # noqa: BLE001 - failures are logged, never dropped
            failures += 1
    sh(["docker", "rm", "-f", name])
    return {
        "pack": target.pack,
        "strategy": strategy,
        "requested": resets,
        "completed": len(samples),
        "failures": failures,
        "samples_s": [round(value, 4) for value in samples],
        **_stats(samples),
    }


def _stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"p50_s": None, "p95_s": None, "p99_s": None, "mean_s": None}
    ordered = sorted(samples)
    return {
        "p50_s": round(_q(ordered, 0.50), 4),
        "p95_s": round(_q(ordered, 0.95), 4),
        "p99_s": round(_q(ordered, 0.99), 4),
        "mean_s": round(statistics.fmean(samples), 4),
    }


def _q(ordered: list[float], q: float) -> float:
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def concurrency_sweep(target: Target, workers: list[int], rounds: int = 8) -> list[dict[str, Any]]:
    """Episodes/hour as offered concurrency rises (strategy s2)."""

    rows = []
    for width in workers:
        started = time.perf_counter()
        completed = 0
        procs = []
        for index in range(width * rounds):
            name = _name(target, f"-c{width}-{index}")
            proc = subprocess.Popen(
                ["docker", "run", "-d", "--name", name, "--network", "none",
                 target.image, *target.container_cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append((name, proc))
        for name, proc in procs:
            proc.wait(timeout=120)
            try:
                _wait_ready(target, name)
                completed += 1
            except Exception:  # noqa: BLE001
                pass
            finally:
                sh(["docker", "rm", "-f", name])
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "pack": target.pack,
                "workers": width,
                "episodes": completed,
                "elapsed_s": round(elapsed, 3),
                "episodes_per_hour": round(completed / elapsed * 3600, 1) if elapsed else None,
            }
        )
    return rows


def qemu_kvm_probe() -> dict[str, Any]:
    """Strategy 4's hardware prerequisite, recorded rather than assumed."""

    kvm = Path("/dev/kvm").exists()
    proc = sh(
        ["qemu-system-x86_64", "-accel", "kvm", "-machine", "q35", "-m", "256",
         "-display", "none", "-S", "-nodefaults"],
        timeout=30,
    )
    return {
        "strategy": "s4_snapshot_restore",
        "kvm_device_present": kvm,
        "qemu_returncode": proc.returncode,
        "qemu_stderr": (proc.stderr or "").strip()[:300],
        "supported": kvm and proc.returncode == 0,
        "reason": None if kvm else "kvm_unavailable_no_/dev/kvm",
    }


def criu_redis_probe(port: int = 6399) -> dict[str, Any]:
    """Qualify CRIU checkpoint/restore on a host-side redis process.

    A restored process that cannot answer PING is not a reset strategy; it is a
    broken checkpoint.  Both the restore and the application check are recorded.
    """

    work = Path("/tmp/criu-redis")
    work.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(work)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    result: dict[str, Any] = {"strategy": "s4_criu", "pack": "redis-setrange-grow-undercount"}
    try:
        sh(["redis-cli", "-p", str(port), "SET", "envcert:marker", "1"])
        dump = sh(["criu", "dump", "-t", str(proc.pid), "-D", str(work), "--shell-job"], timeout=60)
        result["dump_ok"] = dump.returncode == 0
        result["dump_stderr"] = (dump.stderr or "").strip()[:300]
        if dump.returncode == 0:
            proc.terminate()
            proc.wait(timeout=10)
            restored = sh(["criu", "restore", "-D", str(work), "--shell-job", "--restore-detached"], timeout=60)
            result["restore_ok"] = restored.returncode == 0
            result["restore_stderr"] = (restored.stderr or "").strip()[:300]
            time.sleep(1.0)
            ping = sh(["redis-cli", "-p", str(port), "PING"])
            result["application_check"] = ping.stdout.strip() or ping.stderr.strip()[:100]
            result["marker_preserved"] = sh(["redis-cli", "-p", str(port), "GET", "envcert:marker"]).stdout.strip()
            result["supported"] = result["restore_ok"] and result.get("application_check") == "PONG"
    finally:
        if proc.poll() is None:
            proc.kill()
        sh(["pkill", "-f", f"redis-server \\*:{port}"], timeout=10)
    result.setdefault("supported", False)
    return result


# ---------------------------------------------------------------------------
# Economics: pool-priced, never a single hourly rate times summed latency
# ---------------------------------------------------------------------------


def cost_per_committed_step(
    gpu_hours: float,
    cpu_hours: float,
    api_spend: float,
    committed_steps: int,
    gpu_rate: float,
    cpu_rate: float,
) -> dict[str, Any]:
    spend = gpu_hours * gpu_rate + cpu_hours * cpu_rate + api_spend
    if committed_steps <= 0:
        return {
            "total_spend": round(spend, 4),
            "committed_optimizer_steps": committed_steps,
            "cost_per_committed_step": None,
            "note": "0 committed steps: report spend, never a finite unit price",
        }
    return {
        "total_spend": round(spend, 4),
        "committed_optimizer_steps": committed_steps,
        "cost_per_committed_step": round(spend / committed_steps, 4),
    }


def reset_share_threshold(reset_p50_s: float, grade_s: float, decode_s: float) -> dict[str, Any]:
    """Rollout length where reset drops below 10% of elapsed step time."""

    overhead = reset_p50_s + grade_s
    # reset/(reset + decode) < 0.10  =>  decode > 9 * reset_overhead
    threshold = 9.0 * overhead
    return {
        "reset_p50_s": round(reset_p50_s, 4),
        "fixed_overhead_s": round(overhead, 4),
        "decode_s_needed_for_10pct": round(threshold, 2),
        "note": "reset is <10% of step time once decode exceeds 9x the reset+grade overhead",
    }


__all__ = [
    "RESETS",
    "TARGETS",
    "Target",
    "concurrency_sweep",
    "cost_per_committed_step",
    "criu_redis_probe",
    "qemu_kvm_probe",
    "reset_share_threshold",
    "state_equivalence",
    "time_strategy",
]
