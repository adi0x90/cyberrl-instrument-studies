"""ATT-2975 — selection drift on the native-pack pools.

The pool is per (pack, model): 32 sampled repairs.  The "reward" available to a
trainer on this surface is binary — Tier-1 passes or it does not — so Best-of-N
selection reduces to "take the first sample in the first N that Tier-1 accepts".
That is the honest structure of the signal; there is no partial credit to rank on.

Output per (pack, model, N): the four paired cells plus untested, and the primary
metric FAR = #(T1+ ∧ T2−) / #(T1+).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NS = [1, 4, 8, 16, 32]


def analyse(rows: list[dict]) -> dict:
    by_pack_model: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_pack_model.setdefault((row["pack"], row["model"]), []).append(row)
    for items in by_pack_model.values():
        items.sort(key=lambda row: row["sample"])

    out: dict = {"packs": {}, "N": {}}
    for (pack, model), items in sorted(by_pack_model.items()):
        out["packs"].setdefault(pack, {})[model] = {
            "samples": len(items),
            "untested": sum(1 for row in items if row["t1"] == "untested"),
            "t1_holds": sum(1 for row in items if row["t1"] == "HOLDS"),
            "t1_violated": sum(1 for row in items if row["t1"] == "VIOLATED"),
        }

    for n in NS:
        cells = {"t1p_t2p": 0, "t1p_t2m": 0, "no_t1_pass": 0, "untested": 0}
        per_pack: dict = {}
        for (pack, model), items in sorted(by_pack_model.items()):
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
            per_pack.setdefault(pack, {})[model] = {
                "selected": chosen["sample"] if chosen else None,
                "t1": chosen["t1"] if chosen else None,
                "t2": chosen.get("t2") if chosen else None,
                "refuted_by": chosen.get("refuted_by") if chosen else None,
            }
        t1_pass = cells["t1p_t2p"] + cells["t1p_t2m"]
        out["N"][str(n)] = {
            "cells": cells,
            "t1_pass": t1_pass,
            "far_given_t1_pass": (cells["t1p_t2m"] / t1_pass) if t1_pass else None,
            "per_pack": per_pack,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graded", nargs="+", required=True)
    parser.add_argument("--out", default="results/B1_native.json")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.graded:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())

    overall = analyse(rows)
    per_pack = {pack: analyse([r for r in rows if r["pack"] == pack])["N"] for pack in sorted({r["pack"] for r in rows})}
    report = {"metric": "FAR given Tier-1 pass", "N": NS, "analysis": overall, "per_pack": per_pack}
    fields = ["pack", "model", "sample", "t1", "t2", "refuted_by", "untested_reason"]
    report["rows"] = [{key: row.get(key) for key in fields} for row in rows]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    for n in NS:
        block = report["analysis"]["N"][str(n)]
        print(f"N={n:2d} cells={block['cells']} FAR={block['far_given_t1_pass']}")
    for pack, table in per_pack.items():
        print(f"-- {pack}")
        for n in NS:
            b = table[str(n)]
            c = b["cells"]
            print(f"   N={n:2d} T1+&T2+={c['t1p_t2p']} T1+&T2-={c['t1p_t2m']} FAR={b['far_given_t1_pass']}")
    for pack, models in report["analysis"]["packs"].items():
        for model, stat in models.items():
            print(f"{pack[:28]:28s} {model.split('/')[-1]:26s} untested={stat['untested']:2d} t1+= {stat['t1_holds']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
