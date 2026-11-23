# 2026-11-20 masking-GRPO — invalid-action masking in group-relative policy gradient

Training harness for a **preregistered** three-arm experiment. The normative protocol is
[`docs/SPEC.md`](docs/SPEC.md); nothing here changes it. This directory contains the code
that runs it, the offline tests that check the maths, and the report generator that
recomputes every endpoint from the artefacts.

**Status: complete — archived evidence in `evidence-1.5b/` and `evidence-7b/`.** Both `Qwen2.5-Coder-1.5B-Instruct` and `Qwen2.5-Coder-7B-Instruct` runs finished across all three preregistered arms (`control`, `masked`, `penalty`) and two seeds (`20261120`, `20261121`). LoRA adapter checkpoints (~360 MB) are omitted from this lightweight archive; every rollout JSONL, metrics JSON, eval JSON, and recomputed `REPORT.md` is included.

The question: in a sparse-reward cyber-repair task where the strict action envelope rejects
a substantial fraction of rollouts, does **excluding invalid actions from the group
baseline** (masking) change held-out repair success, relative to scoring invalid actions as
reward 0 in the group statistics?

| Arm | Invalid action handling | Group statistics |
|---|---|---|
| **A — control** | reward 0.0, kept in the group | mean and population std over all G samples |
| **B — masked** | recorded, removed from the group | mean and std over admitted samples only; fewer than 2 admitted ⇒ group skipped and logged |
| **C — penalty** (Amendment v2) | reward **−1.0**, kept in the group | mean and std over all G samples |

All arms: same model, prompts, tasks, decoding, optimizer. Only two things differ: the
reward a rejected completion carries (`grpo.invalid_reward`: 0.0 for control/masked, −1.0
for penalty) and the baseline population (all G for control and penalty, admitted only for
masked). Amendment v2 also adds a second seed and fixes held-out evaluation at 16
decodings per task; unequal seed counts are reported, not hidden.

## What the maths implies before any data

For a **fixed** set of G completions, masked-effective implies control-effective: invalid
samples are reward 0 in control, so if control's eight rewards were all equal the admitted
subset would be equal too and masking would have no variance either. The contrapositive is
the useful one — masking can only *lose* effective groups (fewer than two admitted, or all
admitted samples identical), never create one. That is implemented and asserted in
`tests/test_grpo.py::test_masked_effective_implies_control_effective_for_binary_reward`.
That argument is specific to control's reward vector: penalty changes what a rejected
completion is worth, so it can create variance where control has none (a group of one HOLDS
and the rest rejected has spread under penalty, but not under control).

So the mechanism endpoint (`rollouts per effective update`) can only tie or favour control
under identical sampling. If masking helps anything, it must be through the **different
parameter trajectory** — no negative advantage spent on rejected completions — which is what
the co-primary held-out endpoints measure. This is why the protocol budgets *effective
steps* (so both arms get the same number of gradient steps) and reports rollouts as a cost
rather than a confound. Read the report with that asymmetry in mind; do not expect
`masked` to win the mechanism endpoint.

## Endpoints and decision rule (from the SPEC, applied in `analysis/summarize.py`)

Co-primary: **held-out joint success** (4 held-out tasks × `--decodings`; Amendment v2
fixes 16, so 64 draws per adapter, Wilson 95%) and **held-out admission rate** (Wilson
95%). Secondary: train joint success,
rollouts per effective update, mean group variance, mean |advantage| of admitted samples,
groups sampled / skipped / zero-variance.

The rule is applied to **every preregistered pair** — masked vs control, penalty vs
control, penalty vs masked:

* The higher arm is better only if held-out joint success exceeds the other by ≥ 12
  percentage points **and** the Wilson intervals do not overlap.
* The reverse comparison uses the same rule in the other direction.
* **No measurable difference** otherwise, reported with both intervals and the rollout-cost
  ratio. A cost difference with no outcome difference is itself reportable.
* A pair whose arm has no artefacts is reported as **not run**, never as a zero.

Budget parity is on **gradient steps**: every arm is asked for the same number of effective
updates (a group whose arm treatment yields a nonzero-variance baseline) under the same hard
group cap. The original protocol fixes 24/96; the v1 protocol deviation and Amendment v2 use
8 effective updates under a 40-group cap, which is what the dry run exercises. The arm stops
at whichever comes first; the shortfall is recorded, and the manifest records which budget
was in force.

## File map

```
2026-11-20-masking-grpo/
  README.md              # this page
  docs/SPEC.md           # the normative preregistration (do not edit after a run)
  train.py               # one arm: sample -> admit -> grade -> advantage -> one step
  evaluate.py            # held-out eval of a saved adapter (4 tasks x 8 decodings)
  rollout.py             # prompt, sampling, admission, reward, corpus split; reuses harness.*
  grpo.py                # pure maths: group_advantage, group_stats, group_loss, treated_mask
  model_utils.py         # torch/transformers/peft loading, LoRA, generate, update, adapter IO
  analysis/summarize.py  # recomputes every endpoint -> results/REPORT.md
  tests/test_grpo.py     # hand-computed advantage cases (stdlib only)
  tests/test_rollout.py  # prompt, split, admission/reward, dry-run CLIs, report
  Makefile  requirements.txt
  results/               # RUNTIME ONLY, gitignored by convention (never commit it)
    rollouts_<arm>.jsonl  metrics_<arm>.json  eval_<arm>.json  REPORT.md
    adapter_<arm>/        dryrun/            # dry-run artefacts live here
```

The corpus and graders are **not duplicated here**. `rollout.py` imports
`harness.envelopes.admit_strict`, `harness.graders.grade_obligation` and
`harness.tasks_io.*` from `../2026-11-20-frontier-reexperiment/`, resolved relative to this
file. The repository layout (both experiment directories as siblings) is therefore a
requirement, not a convenience — see the rsync commands below.

`results/` is disposable: it holds dry-run artefacts on a laptop and the real artefacts on
the pod. Nothing in it should be committed. If the repository root has no ignore rule for
it, add `results/` to the experiment's `.gitignore` before the run — or at minimum do not
`git add` it.

## Offline: tests and dry run (no GPU, no network, no API key)

```bash
cd .../experiments/2026-11-20-masking-grpo

make test        # pytest: hand-computed advantages, prompt contract, admission/reward,
                 # corpus split, dry-run CLIs, Wilson/sign-test, decision rule, report build
make dryrun      # all three arms + three held-out evals (16 decodings) + results/dryrun/REPORT.md

# the individual commands, if you prefer them explicitly:
python train.py --arm control --dry-run --effective-updates 2 --max-groups 6 \
  --out /tmp/r_control.jsonl --metrics /tmp/m_control.json
python train.py --arm masked  --dry-run --effective-updates 2 --max-groups 6 \
  --out /tmp/r_masked.jsonl  --metrics /tmp/m_masked.json
python train.py --arm penalty --dry-run --effective-updates 2 --max-groups 6 \
  --out /tmp/r_penalty.jsonl --metrics /tmp/m_penalty.json
python evaluate.py --arm control --dry-run --decodings 16 --out /tmp/eval_control.json
python analysis/summarize.py --dry-run          # -> results/dryrun/REPORT.md
```

`--dry-run` swaps the policy for a deterministic canned model built from the task files: no
torch, no adapter, no GPU. It exercises every branch (admission, HOLDS/VIOLATED/UNKNOWN,
zero-variance groups, skipped groups) and is byte-for-byte reproducible. **Dry-run numbers
are not evidence** — the report prints a banner saying so.

## GPU run: exact procedure

The pod is disposable. Everything that matters is written under `results/`, and the adapter
is what `evaluate.py` consumes. Run inside `tmux` so a dropped SSH session cannot kill an
arm.

### 0. Paths used below

| | |
|---|---|
| experiment dir on the pod | `/workspace/masking-grpo` |
| corpus of record on the pod | `/workspace/2026-11-20-frontier-reexperiment` (its sibling; required) |
| venv | `/workspace/venv` |
| model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| outputs | `/workspace/masking-grpo/results/` |

### 1. Stage the code on the pod

From the repository root on the workstation (both directories, as siblings — the masking
harness imports the frontier harness):

```bash
export VPS=root@<pod-ip>

rsync -av --delete -e "ssh -i ~/.ssh/<pod-key>" \
  ./2026-11-20-masking-grpo/ \
  "$VPS":/workspace/masking-grpo/

rsync -av --delete \
  --exclude '.venv' --exclude 'results' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'evidence' \
  -e "ssh -i ~/.ssh/<pod-key>" \
  ./2026-11-20-frontier-reexperiment/ \
  "$VPS":/workspace/2026-11-20-frontier-reexperiment/
```

`-e "ssh -i ~/.ssh/<pod-key>"` is what actually authenticates the copy. Without it rsync
falls back to the default identity and fails. `evidence/` is excluded so a remote copy can
never overwrite the archived frontier evidence.

### 2. Environment

```bash
ssh -i ~/.ssh/<pod-key> "$VPS"
python3 -m venv /workspace/venv
/workspace/venv/bin/python -m pip install -r /workspace/masking-grpo/requirements.txt

cd /workspace/masking-grpo
tmux new -s masking

# pin the revision. Pass the commit SHA; the resolved commit is recorded either way.
export MASKING_GRPO_MODEL_REVISION=<commit-sha>

make test        # offline gate: must be green before any GPU time is spent
```

If the revision is unknown, find it on the pod without training anything:

```bash
/workspace/venv/bin/python - <<'PY'
from transformers import AutoConfig
print(AutoConfig.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")._commit_hash)
PY
```

`--model-revision` (or `MASKING_GRPO_MODEL_REVISION`) is recorded in
`metrics_<arm>.json` as `model.requested_revision`; the loader also records the commit it
actually resolved as `model.resolved_revision`, plus `model.pinned`. A floating branch is
therefore visible in the manifest instead of silently passing as a pin.

### 3. Train all three arms

```bash
make train-a     # control -> results/rollouts_control.jsonl, metrics_control.json, adapter_control/
make train-b     # masked  -> results/rollouts_masked.jsonl,  metrics_masked.json,  adapter_masked/
make train-c     # penalty -> results/rollouts_penalty.jsonl, metrics_penalty.json, adapter_penalty/
```

`--hourly-rate <usd-per-hour>` (or `make train-a HOURLY_RATE=…`) records
`spend.estimated_spend_usd` against the SPEC's $3.00 cap. It is read from the command line
only; never write a rate or a key into a file in this repository.

Each arm stops at the target number of effective updates or the group cap, whichever comes
first. `status`,
`stop_reason`, `abort_reason` and the shortfall are written to the metrics file; the arm
also aborts on >10 consecutive update exceptions, two OOMs, or a wall-time overrun.

### 4. Evaluate and report

```bash
make eval ARM=control    # -> results/eval_control.json  (4 held-out tasks x 16 decodings)
make eval ARM=masked     # -> results/eval_masked.json
make eval ARM=penalty    # -> results/eval_penalty.json
make report              # -> results/REPORT.md (recomputed from the artefacts)
```

`make eval` uses the final-update adapter only. There is no checkpoint selection by design.

### 5. Bring the artefacts back, then destroy the pod

```bash
# on the workstation, before destroying anything:
rsync -av -e "ssh -i ~/.ssh/<pod-key>" \
  "$VPS":/workspace/masking-grpo/results/ \
  ./2026-11-20-masking-grpo/results/
```

Stop **and destroy** the instance so billing stops. The API key stays in the environment
(`$VULTR_API_KEY`), never in a file here:

```bash
# find the instance id
curl -s -H "Authorization: Bearer $VULTR_API_KEY" \
  https://api.vultr.com/v2/instances | python3 -c \
  'import json,sys; [print(i["id"], i["label"], i["status"]) for i in json.load(sys.stdin)["instances"]]'

# destroy it (this is what stops billing; a halted instance still bills)
curl -s -X DELETE -H "Authorization: Bearer $VULTR_API_KEY" \
  https://api.vultr.com/v2/instances/<INSTANCE_ID>
```

If the provider's console is being used instead: *halt* is not *delete*. Terminate the
instance, not just the process, and confirm it has disappeared from the instance list.

## Exact CLI

```
train.py     --arm control|masked|penalty   (required)
             --effective-updates 24      # stop after N effective gradient steps
             --max-groups 96             # hard cap on groups sampled
             --prompts-per-update 4      # groups per round; one optimizer step per group
             --g 8 --seed 20261120 --max-minutes 90
             --out results/rollouts_<arm>.jsonl
             --metrics results/metrics_<arm>.json
             --adapter-out results/adapter_<arm>
             --model Qwen/Qwen2.5-Coder-1.5B-Instruct --model-revision <sha>
             --dtype auto|bfloat16|float16|float32
             --local-files-only            # forbid the HF Hub; model must be cached
             --tasks-dir <corpus>  --hourly-rate <usd/h>  --dry-run

evaluate.py  --arm control|masked|penalty   (required)
             --adapter results/adapter_<arm>   (ignored by --dry-run)
             --out results/eval_<arm>.json --decodings 16 --seed 20261120
             --model … --model-revision <sha> --dtype … --local-files-only
             --max-minutes 30 --dry-run

analysis/summarize.py   --dir results --out results/REPORT.md  --dry-run
                        --rollouts-control|masked|penalty … --metrics-… … --eval-… … (overrides)

The summariser reads every arm that exists and prints `not run` for any that does not; it
also detects seed runs (the flat `<arm>` files, `seed<number>/` subdirectories, or
`*_<arm>_<number>` suffixes) and prints per-seed rows alongside a pooled row.
```

`--updates` from an earlier draft no longer exists: budget parity is on effective gradient
steps, so the flag is `--effective-updates` and the cap is `--max-groups`.

## What the run writes (and what the report reads)

`rollouts_<arm>.jsonl` — one record per attempted completion:
`task_id, group_id, group_index, sample_index, arm, completion, admitted, admission_reason,
expression, verdict, reasons, reward, prompt_tokens, completion_tokens, sum_logprob`.
`sum_logprob` is the policy score used in the loss (null on groups that took no step).

`metrics_<arm>.json` — the manifest (model id, requested/resolved revision, GPU, dtype,
git SHA, SPEC hash, argv, seed, task hashes, budget, spend) plus `status`, `stop_reason`,
`abort_reason`, `counts`, `endpoints` and one record per group (`effective`, `skipped`,
`zero_variance`, `mean_reward`, `admission_rate`, `group_std`, `group_variance`, `valid_count`,
`mean_abs_advantage_admitted`, `loss`, `optimizer_step`, `wall_time_s`, `gpu_mem_mb`,
`verdicts`, `error`).

`eval_<arm>.json` — every held-out draw plus the aggregate endpoints and per-task breakdown.

The report **recomputes** groups, effective flags, rates and intervals from the rollout
JSONL and then cross-checks the metrics file; any disagreement is printed as an integrity
problem (see §6.2 of the report). It also re-hashes `docs/SPEC.md` and raises a banner if the
protocol changed after the run.

## Conventions and guardrails

* **No secrets, no API calls.** Nothing in this directory reads a key, and the model is
  loaded from the local HF cache on the pod (`Qwen/Qwen2.5-Coder-1.5B-Instruct`). The only
  network use in the whole protocol is the pod's one-time model download, which the GPU
  provisioner owns. Pass `--local-files-only` (or `make train-a LOCAL_ONLY=1`) to forbid the
  Hub outright: the load then fails fast with a clear `ModelError` instead of reaching out,
  which is how the "no network" property is demonstrated rather than asserted. Verified
  here: with `--local-files-only` and no cached weights the run exits 2 before any rollout,
  with no Hub request.
* **Dry-run artefacts are not evidence.** They are canned strings derived from the task
  files; the report says so in a banner.
* **One arm per process.** `train.py` runs exactly one named arm in a process, so the arms
  cannot share optimiser state or sampling RNG by accident.
* **One optimizer step per group.** The SPEC says "batching by group" and "each batch of
  prompts produces one gradient step"; a batch *is* one group here, so the effective-update
  budget counts gradient steps 1:1 and `--prompts-per-update` is the number of groups sampled
  per round. This is the only reading under which "24 effective updates = 24 groups whose
  treatment had nonzero variance" is well defined. A zero-variance group takes no step at all
  (AdamW would otherwise still apply weight decay).
* **No PPO clipping, no KL term, no gradient clipping, weight_decay 0.** The importance ratio
  is 1 by construction (one step per group), which is what isolates the baseline treatment.
  `TorchPolicy.update` cross-checks its tensor loss against `grpo.group_loss` on the detached
  log-probabilities and raises on disagreement, so the GPU path cannot silently diverge from
  the tested pure-Python maths.
* **The tests never need a GPU.** `harness.tasks_io`'s per-case worker is forced to `fork`
  inside `tests/test_rollout.py`, because under pytest the parent's `__main__` is the pytest
  console script and the harness's default `spawn` would re-execute it.

## Known limits

* Dry-run completions are drawn from each task's reference fix, alternates, and seeded-bad
  entries, so dry runs verify harness plumbing rather than model capability.
* `results/` and `results7b/` are gitignored runtime directories; archived evidence lives in
  `evidence-1.5b/` and `evidence-7b/`.
