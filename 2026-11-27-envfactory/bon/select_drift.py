"""B1 — paired selection drift on one shared pool.

The pool is generated once (n=32 per task/model) and rescored at
N ∈ {1,4,8,16,32}.  For every selection we record the Tier-1 verdict of the
selected candidate and the independent Tier-2 verdict of the *same* candidate,
then publish the four paired cells.

Primary metric:

    False-Accept Rate given Tier-1 pass = #(T1+ & T2-) / #(T1+)

with exact numerator/denominator at every N, plus a task-clustered bootstrap CI
(resampling tasks, not samples, because candidates within a task are paired).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

HOLDS, VIOLATED, UNKNOWN = "HOLDS", "VIOLATED", "UNKNOWN"
NS = [1, 4, 8, 16, 32]
REWARDS = {
    "poc": lambda row: 1.0 if row["weak"] == HOLDS else 0.0,
    "static": lambda row: 1.0 if row["t1"] == HOLDS else 0.0,
    "property": lambda row: 1.0 if (row["t1"] == HOLDS and row["t2"] == HOLDS) else 0.0,
}
BOOTSTRAP = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def select(samples: list[dict], n: int, reward) -> dict:
    pool = samples[:n]
    best = max(range(len(pool)), key=lambda index: (reward(pool[index]), -index))
    return pool[best]


def count_cells(rows: list[dict]) -> dict:
    cells = {"t1p_t2p": 0, "t1p_t2m": 0, "t1m_t2p": 0, "t1m_t2m": 0, "unknown": 0}
    for row in rows:
        t1_pass = row["t1"] == HOLDS
        t2_pass = row["t2"] == HOLDS
        if row["t1"] == UNKNOWN or row["t2"] == UNKNOWN:
            cells["unknown"] += 1
            continue
        if t1_pass and t2_pass:
            cells["t1p_t2p"] += 1
        elif t1_pass and not t2_pass:
            cells["t1p_t2m"] += 1
        elif not t1_pass and t2_pass:
            cells["t1m_t2p"] += 1
        else:
            cells["t1m_t2m"] += 1
    return cells


def analyse(rows: list[dict], rng: random.Random) -> dict:
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: row["sample"])

    out: dict = {"tasks": len(by_task), "N": {}}
    for n in NS:
        per_reward: dict = {}
        for name, reward in REWARDS.items():
            selected = [select(task_rows, n, reward) for task_rows in by_task.values()]
            cells = count_cells(selected)
            t1_pass = cells["t1p_t2p"] + cells["t1p_t2m"]
            far = cells["t1p_t2m"] / t1_pass if t1_pass else None

            # task-clustered bootstrap: resample tasks with replacement
            far_samples = []
            for _ in range(BOOTSTRAP):
                draw = [rng.choice(list(by_task)) for _ in range(len(by_task))]
                c = {"t1p_t2p": 0, "t1p_t2m": 0}
                for task_id in draw:
                    row = select(by_task[task_id], n, reward)
                    if row["t1"] == HOLDS:
                        if row["t2"] == HOLDS:
                            c["t1p_t2p"] += 1
                        elif row["t2"] == VIOLATED:
                            c["t1p_t2m"] += 1
                denom = c["t1p_t2p"] + c["t1p_t2m"]
                if denom:
                    far_samples.append(c["t1p_t2m"] / denom)
            lo = statistics.quantiles(far_samples, n=40)[0] if len(far_samples) > 40 else None
            hi = statistics.quantiles(far_samples, n=40)[-1] if len(far_samples) > 40 else None
            per_reward[name] = {
                "cells": cells,
                "t1_pass": t1_pass,
                "far_given_t1_pass": far,
                "far_ci": wilson(cells["t1p_t2m"], t1_pass) if t1_pass else None,
                "far_bootstrap95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            }
        out["N"][str(n)] = per_reward
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", nargs="+", required=True)
    parser.add_argument("--out", default="results/B1.json")
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    report: dict = {"metric": "FalseAcceptRateGivenTier1Pass", "files": args.scored, "models": {}}
    for path in args.scored:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        models = sorted({row["model"] for row in rows})
        for model in models:
            model_rows = [row for row in rows if row["model"] == model]
            admitted = sum(1 for row in model_rows if row["admitted"])
            report["models"][model] = {
                "pool": len(model_rows),
                "admitted": admitted,
                "p_admitted": round(admitted / len(model_rows), 4) if model_rows else None,
                "analysis": analyse(model_rows, rng),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for model, data in report["models"].items():
        print(f"\n{model}  P(admitted)={data['p_admitted']}")
        for n in NS:
            cells = data["analysis"]["N"][str(n)]["static"]["cells"]
            far = data["analysis"]["N"][str(n)]["static"]["far_given_t1_pass"]
            print(f"  N={n:2d} static: T1+&T2-={cells['t1p_t2m']:2d} T1+={data['analysis']['N'][str(n)]['static']['t1_pass']:2d} FAR={far}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
