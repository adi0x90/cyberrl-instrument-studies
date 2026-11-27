"""Generate repair candidates for one native pack, API-only (ATT-2975).

Resume is keyed on ``(pack, model, sample)`` so a killed run never restarts a pool,
and the spend cap is enforced before dispatch (never mid-pool truncation).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bon.models_ext import chat_capped
from bon.native_prompts import assert_prompt_clean, build_messages, observed_block

PRICE_IN_PER_M = 0.30
PRICE_OUT_PER_M = 1.00


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--repo", required=True, help="staged repository root")
    parser.add_argument("--file", required=True, help="path of the vulnerable file, repo-relative")
    parser.add_argument("--baseline", required=True, help="PoC output captured on the vulnerable tree")
    parser.add_argument("--product", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--cap-usd", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8000, help="reasoning models burn the budget before output")
    parser.add_argument("--reasoning-tokens", type=int, default=1024, help="reserve budget for content")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing", file=sys.stderr)
        return 2

    repo = Path(args.repo)
    file_text = (repo / args.file).read_text(encoding="utf-8")
    block = observed_block(Path(args.baseline).read_text(encoding="utf-8"))
    messages = build_messages(args.product, args.tag, args.file, file_text, block, args.language)
    assert_prompt_clean(messages[0]["content"] + messages[1]["content"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, int]] = set()
    spend = 0.0
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["pack"], row["model"], row["sample"]))
                spend += row["prompt_tokens"] / 1e6 * PRICE_IN_PER_M + row["completion_tokens"] / 1e6 * PRICE_OUT_PER_M

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    work = [(model, sample) for model in models for sample in range(args.n) if (args.pack, model, sample) not in done]
    print(f"[native-gen] {args.pack}: {len(work)} to fetch, spend so far ${spend:.3f}", flush=True)

    lock = threading.Lock()
    state = {"spend": spend, "written": 0, "stopped": False}

    def fetch(model: str, sample: int):
        with lock:
            if state["stopped"] or state["spend"] >= args.cap_usd:
                state["stopped"] = True
                return None
        reply = chat_capped(
            model,
            messages,
            temperature=1.0,
            seed=20261127 + sample,
            max_tokens=args.max_tokens,
            reasoning_max_tokens=args.reasoning_tokens,
        )
        cost = reply["prompt_tokens"] / 1e6 * PRICE_IN_PER_M + reply["completion_tokens"] / 1e6 * PRICE_OUT_PER_M
        with lock:
            state["spend"] += cost
            if state["spend"] > args.cap_usd:
                state["stopped"] = True
        return {
            "pack": args.pack,
            "model": model,
            "sample": sample,
            "temperature": 1.0,
            "seed": 20261127 + sample,
            "prompt_tokens": reply["prompt_tokens"],
            "completion_tokens": reply["completion_tokens"],
            "finish_reason": reply["finish_reason"],
            "raw_reply": reply["text"],
        }

    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, model, sample): (model, sample) for model, sample in work}
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            with lock:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                state["written"] += 1
                if state["written"] % 16 == 0:
                    print(f"[native-gen] {state['written']}/{len(work)} est ${state['spend']:.3f}", flush=True)

    print(f"[native-gen] {args.pack}: wrote {state['written']}, est spend ${state['spend']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
