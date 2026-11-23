# RUNBOOK — running the 2026-11-20 frontier re-experiment on a VPS

The exact procedure used for the 20 November 2026 run, in the order it was executed.
The experiment is finished and its instance is destroyed; this file exists so the run can
be repeated, audited, or adapted with no guesswork about flags, paths or ordering.

Preregistration lives in [`SPEC.md`](SPEC.md) (interfaces) and [`DESIGN.md`](DESIGN.md)
(endpoints, falsifiers, decision rules). Read `DESIGN.md` before reading any number.

> **The OpenRouter API key must never be written into this repository.** It lives in the
> environment of the machine that spends the calls. Not in a file under this directory, not
> in `requirements.txt`, not in a Makefile variable, not in a commit, not in a shell history
> that is kept. The runners read `OPENROUTER_API_KEY` from the environment only, and they
> never log or write its value. If a key is ever pasted into a tracked file, rotate it — the
> git history is the leak.

---

## 0. Shape of the run

| | units | model calls | where |
|---|---|---|---|
| E1 | 216 = 12 tasks × 3 envelopes × 3 models × 2 decodings | 216 | three per-model shards, then merge |
| E2 | 279 candidates | 0 (local grading only) | after E1 is merged |
| E3 | 180 = 10 polarity-sensitive tasks × 2 variants × 3 models × 3 decodings | 180 | three per-model shards, then merge |

396 model calls total. No GPU. A `vc2-2c-4gb` Vultr instance in Bangalore was enough; the
remote box is disposable and `results/` is the only thing worth carrying back.

Because the three models were run as **separate shards**, a dropped session costs one
model's progress, not the whole arm: `--resume` skips units already in that shard's JSONL.

---

## 1. Stage the code on the instance

From this replication package root on the workstation, with your SSH key
`~/.ssh/<vps-key>` and the VPS address in `$VPS`:

```bash
export VPS=root@<instance-ip>

rsync -av --delete \
  --exclude '.venv' --exclude 'results' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'evidence' \
  -e "ssh -i ~/.ssh/<vps-key>" \
  ./2026-11-20-frontier-reexperiment/ \
  "$VPS":/opt/frontier/
```

Notes:

- `-e "ssh -i ~/.ssh/<vps-key>"` authenticates the transfer; without it
  `rsync` falls back to the default identity.
- `results/` and `__pycache__/` are excluded because they are local runtime state.
- `evidence/` is excluded on a **fresh** run: the instance produces shards, the archive is
  built on the workstation afterwards (§7). Never let a remote copy overwrite the archived
  evidence.

## 2. Install the runtime

```bash
ssh -i ~/.ssh/<vps-key> "$VPS"
cd /opt/frontier
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # httpx + pytest (+ numpy/scipy, unused by the report)
```

`analysis/summarize.py` implements Wilson, exact McNemar, Fisher exact and Cohen's κ in the
standard library, so a bare interpreter can regenerate the report; `numpy`/`scipy` are not
required for it. The runners need `httpx`.

## 3. Stage the key outside the repo

The key lives at `/root/.openrouter_key`, mode `600`, and is exported by sourcing a small
env file. Nothing under `/opt/frontier` ever contains the value.

```bash
# on the instance
mkdir -p /root/cyberrl && chmod 700 /root/cyberrl

# transfer from the workstation (run this on the workstation):
#   scp -i ~/.ssh/<vps-key> ~/.openrouter_key "$VPS":/root/.openrouter_key
# then, back on the instance:
chmod 600 /root/.openrouter_key

cat > /root/cyberrl/env.sh <<'EOF'
# Sourced, never committed. Exports the key for the runners.
export OPENROUTER_API_KEY="$(cat /root/.openrouter_key)"
EOF
chmod 600 /root/cyberrl/env.sh

source /root/cyberrl/env.sh
# check presence only -- never echo the value:
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "key present" || echo "key MISSING"
```

`make e1` / `make e3` only test that the variable is *set* (`${VAR+x}`) and fail fast
otherwise. Do not paste the key into a shell command line, which leaks it into `ps` and the
history file.

## 4. Run inside tmux

A dropped SSH session must not kill the run:

```bash
tmux new -s frontier
source /root/cyberrl/env.sh
cd /opt/frontier
mkdir -p results/logs            # the tee targets below live here
```

Cheap pre-flight — the plumbing, with no calls and no key:

```bash
make test                 # offline pytest; also the local gate before any run
.venv/bin/python run_e1.py --mock --limit 2 --out results/smoke_e1.jsonl
```

### 4.1 E1 — three per-model shards

```bash
.venv/bin/python run_e1.py --resume --models deepseek/deepseek-v4.1-flash-1017 \
  --seed 20261120 --out results/e1_ds41.jsonl 2>&1 | tee results/logs/e1_ds41.log
.venv/bin/python run_e1.py --resume --models z-ai/glm-5.3-flash \
  --seed 20261120 --out results/e1_glm.jsonl  2>&1 | tee results/logs/e1_glm.log
.venv/bin/python run_e1.py --resume --models tencent/hy4-flash \
  --seed 20261120 --out results/e1_hy4.jsonl  2>&1 | tee results/logs/e1_hy4.log
```

72 units per shard (12 tasks × 3 envelopes × 2 decodings), 216 in total. Each runner writes
`results/e1_<shard>.manifest.json` next to its JSONL and appends one record per attempted
unit, failures included.

### 4.2 E3 — three per-model shards

```bash
.venv/bin/python run_e3.py --resume --models deepseek/deepseek-v4.1-flash-1017 \
  --seed 20261120 --out results/e3_ds41.jsonl 2>&1 | tee results/logs/e3_ds41.log
.venv/bin/python run_e3.py --resume --models z-ai/glm-5.3-flash \
  --seed 20261120 --out results/e3_glm.jsonl  2>&1 | tee results/logs/e3_glm.log
.venv/bin/python run_e3.py --resume --models tencent/hy4-flash \
  --seed 20261120 --out results/e3_hy4.jsonl  2>&1 | tee results/logs/e3_hy4.log
```

60 units per shard (10 polarity-sensitive tasks × 2 variants × 3 decodings), 180 in total.
`run_e3.py` selects the polarity-sensitive tasks itself and records the two it skips in the
manifest (`tasks_skipped_not_polarity_sensitive`).

## 5. Merge the shards

```bash
.venv/bin/python scripts/merge_shards.py --exp e1      # results/e1_ds41, _glm, _hy4 -> results/e1.jsonl
.venv/bin/python scripts/merge_shards.py --exp e3      # -> results/e3.jsonl
```

The script concatenates the shards in sorted order, writes `results/<exp>.jsonl`, and
synthesises `results/<exp>.manifest.json` from the first shard's manifest: the three SPEC
model ids, the merged `out` path, an `argv` note naming the merge, `records_written`, and the
`shards` that went in. It is deterministic and reads no clock and no network.

The three shards are what the models actually produced; the merged file is those same lines
in a fixed order. **Only the merged files are authoritative** — the `results/e1_*.jsonl` and
`results/e3_*.jsonl` shards are retained as provenance, and the report never reads them.

## 6. E2, then the report

E2 needs no key: it builds its candidate corpus locally and grades it with both graders.

```bash
.venv/bin/python run_e2.py --e1 results/e1.jsonl --out results/e2.jsonl
.venv/bin/python scripts/reclassify_e3.py --evidence results/e3.jsonl   # idempotent; see §9
.venv/bin/python analysis/summarize.py \
  --e1 results/e1.jsonl --e2 results/e2.jsonl --e3 results/e3.jsonl \
  --out results/REPORT.md
```

`run_e2.py --e1` imports the **admitted E1 patches** (96 of the 216 units), so E2 must run
after E1 is merged. Every count, rate, CI and p-value in `REPORT.md` is recomputed from the
three JSONL files; no endpoint number is hardcoded.

## 7. Bring the result back and archive it

Build the archive on the workstation — never let a remote `results/` overwrite `evidence/`:

```bash
rsync -av -e "ssh -i ~/.ssh/<vps-key>" \
  "$VPS":/opt/frontier/results/ ./results-vps/

# archive: merged JSONL + every manifest + the runner logs
cp results-vps/e*.jsonl results-vps/e*.manifest.json evidence/
cp results-vps/logs/*.log evidence/logs/

# the report is regenerated against the archived paths, so its Provenance table
# names the files a reader actually has
.venv/bin/python analysis/summarize.py \
  --e1 evidence/e1.jsonl --e2 evidence/e2.jsonl --e3 evidence/e3.jsonl \
  --out evidence/REPORT.md
```

What ends up in `evidence/` and which file every number comes from is documented in
[`evidence/README.md`](../evidence/README.md).

## 8. Teardown

Destroy the instance with the Vultr v2 DELETE call, then make sure nothing local still holds
the key:

```bash
curl -s -X DELETE "https://api.vultr.com/v2/instances/$INSTANCE_ID" \
  -H "Authorization: Bearer $VULTR_API_KEY" -w '%{http_code}\n'
# expect 204

unset OPENROUTER_API_KEY
tmux kill-server          # on the instance, or let the destroy take it
```

The instance is throw-away; the archived `evidence/` is not. If the key was ever pasted into
a persistent shell history or a file, rotate it — destroying the box does not un-leak it.

---

## 9. Procedural notes from the 2026-11-20 run

Two things went wrong on the night and are recorded here so the next run does not repeat them.
Neither is repaired by editing data; both are disclosed in the archived report.

1. **A task file was edited between E3 shards.** `tasks/bounds_content_length_01.json`
   changed between the first two E3 shards and the third, so 18 E3 units and 18 E1 units
   pool two versions of that task. The earlier bytes were not retained. The per-shard
   manifests carry a differing `task_sha256`, which is how it is visible; `DESIGN.md` §6.4
   and §6.6 exist to prevent exactly this. **Rule: freeze `tasks/` before the first shard and
   do not touch it until every shard has finished.**
2. **The polarity classifier matched an operator inside a composite operator.** The first
   version matched the converse marker `>` inside `>=`, producing four false `CONVERSE`
   labels on `session.epoch >= state.epoch` expressions. The fix made operator matching
   boundary-aware (two regression tests in `tests/test_polarity.py` pin it), and the four
   records were re-labelled **offline from their saved `raw_expression`** — no new model
   calls, no new data — by `scripts/reclassify_e3.py`, which records the prior verdict in
   `verdict_reclassifier_before`; `evidence/REPORT.md` was regenerated afterwards. Running
   that script against the archived evidence today changes 0 records.

The merge step is also now a script (`scripts/merge_shards.py`) rather than the `cat` plus
hand-written manifest used on the night, so the merged JSONL is rebuildable by command from
the retained shards.
