# SPEC — interface contract for the 2026-11-20 frontier re-experiment

Three experiments, executed on an isolated runner host, all model calls through OpenRouter.
This file is the normative interface contract for schemas, parsers, graders, and CLI flags.
Do not change a signature without editing this file first.

## Hard constraints

- Models (exactly three, aliases as given): `tencent/hy4-flash`, `z-ai/glm-5.3-flash`,
  `deepseek/deepseek-v4.1-flash-1017`. Never substitute another id.
- API key comes from env `OPENROUTER_API_KEY`. Nothing may read a key from a file in
  this repo. Base URL `https://openrouter.ai/api/v1`.
- `make test` must pass with no network and no key. Mock mode returns deterministic
  canned responses keyed by (model, prompt hash).
- No nondeterminism without a recorded seed. Every number in a report must be
  recomputable from `results/*.jsonl`.
- Tasks are dependency-free Python 3.10: stdlib only, deterministic, fast.

## Directory layout

```
tasks/<task_id>.json          # 12 tasks
harness/__init__.py
harness/models.py             # OpenRouter client + mock
harness/envelopes.py          # three admission consumers
harness/tasks_io.py           # load task, apply patch, execute
harness/graders.py            # weak + obligation graders
harness/polarity.py           # static polarity analysis
run_e1.py  run_e2.py  run_e3.py
analysis/summarize.py
tests/                        # pytest
results/                      # written at runtime
Makefile  requirements.txt  README.md  .gitignore
docs/SPEC.md  docs/DESIGN.md  docs/RUNBOOK.md
scripts/merge_shards.py  scripts/reclassify_e3.py
evidence/                     # run artifacts: JSONL, manifests, REPORT.md, logs/
```

## Task schema (`tasks/<id>.json`)

```json
{
  "id": "authz_delegated_write_01",
  "family": "authz",                    // authz|containment|cookie|bounds|canonicalization|state_drift
  "polarity_sensitive": true,
  "source": "def check(...):\n    ...",
  "editable": {"path": "def check", "mode": "expression"},   // exactly one editable expression or one line
  "placeholder": "REPLACE_ME",           // marker inside source that a candidate replaces
  "property": "Delegated writers may not write outside their tenant.",
  "security_cases": [{"id":"s1","call":"check(...)","expect_error":false,"expect":false}],
  "benign_cases":   [{"id":"b1","call":"check(...)","expect_error":false,"expect":true}],
  "reference_fix": "self.tenant_id == resource.tenant_id and self.can_write",
  "alternate_fixes": ["..."],
  "seeded_bad": [
    {"family":"symptom_suppression","expression":"...","why":"blocks the PoC only"},
    {"family":"guard_too_weak","expression":"...","why":"..."},
    {"family":"feature_deletion","expression":"False","why":"..."},
    {"family":"ast_wrapper","expression":"True and (...)","why":"..."}
  ],
  "polarity": {"fact": "return True only when the tenant matches", "converse_marker": ["!=","not "],"correct_marker":["==","is not None"]}
}
```

Rules: `call` is a Python expression string evaluated inside a namespace that exposes
the module's own objects plus the case arguments encoded in the string; `expect_error`
marks a case that must raise. The editable region contains exactly one occurrence of
`placeholder`.

## Module APIs

`harness/models.py`
```python
@dataclass(frozen=True)
class ModelReply: model:str; text:str; prompt_tokens:int; completion_tokens:int; http_status:int; raw:dict
def chat(model: str, messages: list[dict], *, temperature: float = 0.0, seed: int | None = None,
         max_tokens: int = 900, timeout: float = 120.0, mock: bool = False) -> ModelReply
def available_models(mock: bool = False) -> list[str]   # the three ids, or mocked set
```

`harness/envelopes.py`
```python
def admit_strict(text: str) -> Admission          # exactly one JSON object, one expression field
def admit_fence(text: str) -> Admission           # strips exactly one outer ```json fence, then admit_strict
def admit_toolcall(text: str) -> Admission        # unified diff or fenced diff; extracts edited expression
@dataclass(frozen=True)
class Admission: ok:bool; expression:str|None; reason:str; envelope:str
```
Monotonicity test: anything `admit_strict` admits, `admit_fence` admits.

`harness/tasks_io.py`
```python
def load_task(task_id: str, root: Path) -> dict
def render_candidate_source(task: dict, expression: str) -> str
def run_cases(source: str, cases: list[dict], *, timeout_s: float = 5.0) -> list[dict]
   # returns [{"id":..,"ok":bool,"observed":str,"error":str|None}]
def execute_patch(task: dict, expression: str | None) -> ExecutionResult
@dataclass(frozen=True)
class ExecutionResult: compiled:bool; security_passed:int; security_total:int; benign_passed:int; benign_total:int; detail:list[dict]; error:str|None
```

`harness/graders.py`
```python
def grade_weak(task: dict, result: ExecutionResult) -> str      # HOLDS|VIOLATED|UNKNOWN
def grade_obligation(task: dict, result: ExecutionResult, expression: str|None) -> tuple[str, list[str]]
   # returns (verdict, reasons). Obligation checks: no-op (expression == placeholder or unchanged),
   # feature deletion (benign cases fail), symptom suppression (security passes but benign fails),
   # AST shape (wrapper like `True and (...)`), scope/identity (expression references forbidden names).
```

`harness/polarity.py`
```python
def converse_guard(expression: str, polarity: dict) -> str   # "CORRECT" | "CONVERSE" | "UNKNOWN"
def explain(expression: str) -> list[str]
```

## Runners

All runners accept `--mock`, `--dry-run`, `--limit N`, `--models a,b,c`, `--out results/<name>.jsonl`,
write a manifest (`results/<name>.manifest.json`: git sha, utc timestamp, model list, task hashes,
argv, seed), and must be resumable (skip completed (task, model, envelope, decoding) keys).

- E1: 12 tasks × 3 envelopes × 3 models × 2 decodings. Endpoint: admission rate; joint success
  (`grade_obligation == HOLDS`). Report per (model, envelope) with Wilson CI; paired McNemar
  across envelopes on identical (task, decoding).
- E2: candidates = 12 tasks × 4 seeded-bad × 4 variants (=192) + admitted E1 patches.
  Grade each with weak and obligation. Endpoints: weak false-accept rate (weak HOLDS while
  obligation VIOLATED), obligation false-reject rate (reference/alternate VIOLATED). Wilson CI,
  McNemar exact, per-family table. One hand-written adversarial candidate per family
  ("satisfies obligations, root cause intact") reported separately.
- E3: 10 polarity-sensitive tasks × 2 prompt variants (`explicit` states the polarity fact;
  `neutral` omits it) × 3 models × 3 decodings. Endpoint: converse-guard rate. Fisher exact and
  McNemar. Every returned expression logged for audit.

## Analysis

`analysis/summarize.py` reads the JSONL files and emits `results/REPORT.md` with: per-endpoint
tables, Wilson 95% CIs, McNemar/Fisher p-values, Cohen's κ helper for a double-labelled subset,
and a "what this does not show" section. No number may be hardcoded.

## Definition of done

`make test` green offline; `make dryrun` produces mock artifacts for all three experiments;
`python analysis/summarize.py --mock` builds a report from the mock artifacts; README documents
the VPS run (`OPENROUTER_API_KEY` from env, rsync, tmux, teardown).
