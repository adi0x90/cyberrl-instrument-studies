"""B1 — shared Best-of-N pool, local vLLM path.

One pool of `n` completions per (task, model) at temperature 1.0 and a fixed seed,
under the published E1 prompt.  The pool is generated **once** and rescored at
N ∈ {1,4,8,16,32}; regenerating per N would make the comparison a comparison of
sampling noise, not of selection.

Writes one JSONL row per completion so the run is resumable and auditable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams  # noqa: E402

from bon.prompts import build_messages, prompt_sha


def load_tasks(tasks_dir: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(Path(tasks_dir).glob("*.json"))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default="propverifier/tasks")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["task_id"], row["model"]))

    started = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.85)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=0.95,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )

    with out_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            key = (task["id"], args.model)
            if key in done:
                print(f"[gen] skip {key}", flush=True)
                continue
            messages = build_messages(task)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate([prompt], sampling)
            for index, candidate in enumerate(outputs[0].outputs):
                handle.write(
                    json.dumps(
                        {
                            "task_id": task["id"],
                            "model": args.model,
                            "sample": index,
                            "seed": args.seed,
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "prompt_sha": prompt_sha(task),
                            "raw_reply": candidate.text,
                            "finish_reason": candidate.finish_reason,
                            "completion_tokens": len(candidate.token_ids),
                            "prompt_tokens": len(outputs[0].prompt_token_ids),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            print(f"[gen] {task['id']} x{len(outputs[0].outputs)} ({time.time() - started:.1f}s)", flush=True)

    print(f"[gen] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
