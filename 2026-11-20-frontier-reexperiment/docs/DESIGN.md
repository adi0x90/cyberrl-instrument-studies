# DESIGN — the 2026-11-20 frontier re-experiment

This file is the reasoning layer under the code. `docs/SPEC.md` says *what the interfaces are*;
this file says *what the experiment is for, what would refute it, and what it cannot show*.
Every claim here is bounded by the corpus: 12 hand-written tasks, three models, one serving
path, one day.

---

## 0. The question, in one paragraph

Frontier models are used as reward sources and patch authors for security-relevant code.
The failure mode that matters is not "cannot write Python" — it is **silently wrong or
silently unmeasurable**: an answer that the harness cannot admit (so the pipeline records a
failure it cannot interpret), a patch that passes the tests and drops the property (so the
cheap grader pays for a regression), or a guard that is well-formed, plausible, and
implements the *converse* of the policy. Three experiments measure those three failure
modes as separately as the harness allows:

| | experiment | failure mode probed | unit |
|---|---|---|---|
| E1 | envelope × model | *can the answer even be admitted?* | task × envelope × model × decoding |
| E2 | weak vs obligation grader | *does passing the tests mean keeping the property?* | candidate |
| E3 | polarity prompt variant | *is the guard upside down?* | task × variant × model × decoding |

They are deliberately not a capability benchmark. They are a **measurement-instrument
study**: E2 asks whether the grader can be trusted, E1 whether the interface can be
trusted, E3 whether the most basic semantic property — which way round the check goes —
survives.

---

## 1. The instrument

Each experiment leans on one piece of `harness/`, and that piece is itself under test.

- **Envelopes** (`harness.envelopes`) are admission consumers, not graders: `admit_strict`
  wants exactly one JSON object, `admit_fence` strips exactly one outer `json` fence first,
  `admit_toolcall` reads a unified diff. The specification promises **monotonicity**: what
  `admit_strict` admits, `admit_fence` must admit. E1 treats a violation of that promise as
  an instrument defect, and reports it separately (report section E1.d) so it is never read
  as model behaviour.
- **Graders** (`harness.graders`) are the two hypotheses of E2. `grade_weak` is the naive
  reward signal: did every security case pass? (Benign behaviour is deliberately ignored —
  that is what "weak" means.) `grade_obligation` adds the rules a reward signal needs:
  no-op, feature deletion, symptom suppression, AST wrapper, scope/identity.
- **Polarity** (`harness.polarity`) is a conservative marker-based classifier:
  `converse_guard(expression, polarity)` returns `CORRECT` / `CONVERSE` / `UNKNOWN`. It is
  lexical, therefore wrong sometimes; E3.d logs every expression so the labels can be
  audited instead of trusted.
- **Task execution** (`harness.tasks_io`) runs cases in a throw-away child process with a
  timeout. A unit whose `ExecutionResult` reports an infrastructure failure is graded
  `UNKNOWN`, not `VIOLATED` — infra errors are the instrument's problem, not the model's.

Nothing in the runners reimplements a harness behaviour. Where a *generation-time* decision
is needed that is not a harness behaviour — "does this mutation still parse?" — the runner
uses `ast.parse(..., mode="eval")` and says so (see §4).

---

## 2. Endpoints → hypothesis → falsifier

Decision conventions below are **pre-committed before looking at results**. They are
conventions for reading a small-n experiment, not universal thresholds: a stated convention
that the data misses is still reported.

### 2.1 E1 — output envelope

| # | endpoint | how it is computed | hypothesis it tests | falsifier — what would refute it |
|---|---|---|---|---|
| E1-A | admission rate per (model, envelope) | admitted / units, `admit_*` outcome; Wilson 95% CI | **H1a.** The envelope is a real interface cost: admission falls strict → fence → toolcall. | Admission does *not* fall (per-model Δ ≤ 0, or CIs overlap completely) for all three models. Then the "format compliance" story is unsupported on tasks of this size. |
| E1-B | joint success per (model, envelope) | `grade_obligation == HOLDS` / units; Wilson 95% CI; not-admitted units count as failures | **H1b.** The cost that matters is *end-to-end*: fewer usable patches, not just fewer parse successes. | Joint success is flat across envelopes for a model *and* its admission is flat. That model's pipeline is envelope-insensitive; the claim then holds only for other models, or nowhere. |
| E1-C | joint success **given** admission | HOLDS / admitted units | **H1c.** Format cost and reasoning cost are distinguishable: if conditioning removes the envelope gap, the loss was admission, not understanding. | The conditional rate *also* drops from strict to toolcall: harder envelopes change what the model writes, not only whether it parses. (Either result is informative; neither confirms H1b on its own.) |
| E1-D | paired McNemar across envelope pairs | matched on (model, task, decoding); outcome = joint success; exact two-sided binomial on discordant pairs | **H1d.** The envelope effect is a within-unit effect, not a between-task artefact. | Discordant pairs are symmetric (b ≈ c, exact p ≈ 1): envelope membership carries no information about success once the unit is fixed. |
| E1-E | instrument check: strict ⊆ fence | count units with strict admitted and fence not admitted | **H1e (instrument, not model).** The admission promise holds, so differences are attributable to the envelope, not to parser bugs. | One or more violations. The correct response is a harness bug report; E1's model-level numbers are then suspect and must be re-run after the fix. |

Direction rule for reading E1: treat envelope cost as a **working finding** only if the
paired direction (strict-only successes > toolcall-only successes) is consistent in at
least two of the three models *and* the pooled exact McNemar p < 0.05. Otherwise report it
as "not established at this n", including when p < 0.05 appears in a single model only —
that pattern is inside the noise budget of 12 tasks.

### 2.2 E2 — grader adequacy

Candidate universe per task: every seeded-bad expression, three programmatic mutations of
each, the reference fix, every alternate fix, the optional hand-written adversarial entry,
and every admitted E1 patch (see §4 and §5).

| # | endpoint | how it is computed | hypothesis it tests | falsifier — what would refute it |
|---|---|---|---|---|
| E2-A | weak false-accept rate, pooled and per seeded-bad family | `grade_weak == HOLDS` ∧ `grade_obligation == VIOLATED`, over the negative candidates; Wilson 95% CI per family | **H2a.** The cheap grader silently accepts broken patches, and the rate is structured by family (symptom suppression and AST wrappers worse than feature deletion). | Pooled rate at or below the pre-declared 10% floor with a Wilson upper bound under 25%, or perfectly flat per-family rates. Then the weak grader is adequate for these task shapes and the "grader adequacy" concern does not reproduce here. |
| E2-B | obligation false-reject rate on reference / alternate fixes | `grade_obligation == VIOLATED` over positive controls; Wilson 95% CI | **H2b.** The obligation grader does not reject genuinely correct fixes, so it is safe as an automatic filter. | **Any** rejection of a reference fix is a finding: either the grader is over-strict or the task is mis-authored. Either way it is a blocker for using the obligation grader as a reward, and it is triaged by reading the reasons and the task file. |
| E2-C | paired McNemar (weak vs obligation) and Cohen's κ | same candidates scored by both graders; binary `HOLDS`/not; exact McNemar on discordant pairs; κ on the binary labels | **H2c.** The two graders disagree non-randomly, in the direction weak-accepts / obligation-rejects, so the extra rules buy real signal rather than noise. | Discordance is symmetric (b ≈ c) or κ ≥ 0.8 with rates that agree. Then the obligation grader mostly restates the weak one and adds cost without information. |
| E2-D | per task-family breakdown | same false-accept measure grouped by `family` (authz, containment, cookie, …) | **H2d.** Grader blind spots are task-family dependent, which tells us where an automated reward is least safe. | Rates are indistinguishable across families at this n. Report as "no family structure detected", not as "uniform safety". |
| E2-E | hand-written adversarial candidates, reported separately | listed individually: weak verdict, obligation verdict, reasons | **H2e (diagnostic).** The obligation rule set has structural blind spots that a syntactically clean, semantically wrong patch can exploit. | All adversarial candidates are rejected with an informative reason. Then the rule set covers the failure shapes its authors imagined — a bounded, not a global, reassurance. |

Direction rule: the grader gap is a **working finding** if the pooled weak false-accept
rate ≥ 25% (a one-in-four silent false accept is already unusable as a reward) and its
Wilson lower bound excludes 0. The 25% figure is a convention chosen so the conclusion does
not hinge on a single candidate. The adversarial count is *never* converted into a rate: it
is an existence demonstration.

### 2.3 E3 — polarity

| # | endpoint | how it is computed | hypothesis it tests | falsifier — what would refute it |
|---|---|---|---|---|
| E3-A | converse-guard rate per (model, variant) | `converse_guard == CONVERSE` / units; Wilson 95% CI; per (model, variant) | **H3a.** Converse guards occur at a non-trivial rate, and the rate depends on whether the polarity fact was stated. | Rate at or below 10% in both variants for all models: polarity failure is anecdotal here, and the mitigation has nothing to act on. |
| E3-B | Fisher exact, explicit vs neutral | unpaired 2x2 (converse/other × variant), two-sided | **H3b.** Stating the fact reduces the converse rate. | p not small **and** the two rates within a couple of points: the fact is not doing work. A reversed effect (explicit worse) is a stronger finding than a null and must be reported as such. |
| E3-C | paired McNemar, explicit vs neutral | matched on (task, model, decoding); outcome = CONVERSE; exact two-sided binomial | **H3c.** The effect is within-unit: the same task flips polarity because of the prompt, not because different tasks were sampled. | Symmetric discordance (b ≈ c, p ≈ 1). Then the variant does not change the model's behaviour on matched tasks and H3b fails with better control. |
| E3-D | audit log of every expression | one row per unit: task, model, variant, decoding, classification, expression | **H3d (instrument).** The lexical classifier's labels can be checked by a human, so a reported rate is auditable rather than trusted. | Skipped or truncated logging. Then the rate is uninterpretable and must not be reported — the audit log is a prerequisite, not an appendix. |

Direction rule: polarity failure is a **working finding** if the pooled converse rate is
≥ 10% and the explicit-vs-neutral direction is stable per model; the mitigation counts as
working only if McNemar p < 0.05 in the predicted direction with the paired counts also
favourable. If UNKNOWN extraction exceeds 20% of units, E3-A is **void** (it is then mostly
measuring admission, which E1 already covers) and is reported as void instead of as a rate.

---

## 3. Sampling, seeds, decodings

| experiment | decodings | temperature | seed |
|---|---|---|---|
| E1 | `decoding=0` (greedy) | 0.0 | 20261120 |
| E1 | `decoding=1` (warm) | 0.3 | 20261121 |
| E3 | `decoding=0` | 0.0 | 20261120 |
| E3 | `decoding=1` | 0.3 | 20261121 |
| E3 | `decoding=2` | 0.7 | 20261122 |

The seed is `--seed` (default 20261120) plus the decoding index, so it is reproducible from
the CLI and is recorded in every record and in the manifest. A seed is *recorded*, not
*guaranteed*: vendors may ignore it or route to a different backend. A re-run is a new
sample, not a replay, and is never pooled with the first run (different out file or a new
`results/` directory).

The full design is 216 E1 units (12 tasks × 3 envelopes × 3 models × 2 decodings), 0 model
calls for E2 (candidates are local) and 180 E3 units (10 polarity-sensitive tasks × 2
variants × 3 models × 3 decodings) — **396 model calls**. No adaptive sampling.

---

## 4. E2 candidate construction

For each seeded-bad entry `E` in family `F`, the corpus contains:

1. `E` itself (`seeded`);
2. `swap_op` — swap the **first** comparison operator (`==`↔`!=`, `<`↔`>=`, `>`↔`<=`);
3. `mutation:invert_membership` — invert the **first** membership/identity test
   (`in`↔`not in`, `is`↔`is not`);
4. `mutation:weaken_guard` — `(E) or True`, the canonical "add a tautology so it always
   passes" move;
5. `reference` — `reference_fix`, a positive control;
6. `alternate` — each entry of `alternate_fixes`, positive controls;
7. `adversarial` — the task's optional hand-written entry (string, `{expression, why}`, or a
   list of either), reported in its own report section;
8. `e1_patch` — every admitted E1 patch for that task, deduplicated by (task, model,
   expression), reported in its own section.

Rules: a mutation is kept **only if it still parses as a single Python expression**
(`ast.parse(..., mode="eval")`); mutations equal to the original are dropped; every
candidate is then graded by `execute_patch` + both graders, so the pass/fail decision stays
harness-owned. `swap_op` and `invert_membership` act on the first textual match, which is a
deliberate simplification: it keeps the mutation's intent legible and makes the candidate
reproducible from the record alone.

**What these mutations cannot reach.** They are *syntactic neighbourhood* probes. A patch
that removes a check by restructuring the expression (extracting a helper, reordering
conjuncts, moving the guard into the caller) is not reachable by any of the three
mutations. So the per-family false-accept rate is a property of *this mutation family*, not
of the family's real-world failure mode. This is why §2.2 refuses to compare families in a
ranking sense and reports the rate per family with its n.

---

## 5. Record and manifest contract

One JSONL record per **attempted** unit — including units that failed admission, failed
compilation, or raised. A pipeline that only records successes cannot distinguish "the
model was wrong" from "the harness crashed", which is precisely the distinction the
experiment exists to make.

Common keys (all three experiments): `unit`, `experiment`, `task_id`, `model`, `decoding`,
`admitted`, `admission_reason`, `raw_expression`, `verdict`, `reasons`, `prompt_tokens`,
`completion_tokens`, `http_status`, `ts_utc`, `seed`. E1 adds `envelope`; E3 adds `variant`;
E2 adds `candidate_kind`, `seeded_family`, `family`, `weak_verdict`, `compiled`, `source`,
`detail` (and `model = "none"` for locally built candidates). `raw_reply` keeps the full
model text so an admission decision can be re-litigated without re-calling the API.

`unit` is the resume key and is stable across runs: E1 `task|model|envelope|decoding`, E3
`task|model|variant|decoding`, E2 `task|kind|seeded_family|model|sha256(expression)[:12]`.
`--resume` skips any unit key already present. Duplicate keys are impossible by construction
within a run and asserted in the tests; a truncated trailing line (killed process) is ignored
on resume, so that unit is re-attempted rather than silently lost.

The sibling manifest (`<out>.manifest.json`) records: `git_sha` (`git rev-parse HEAD`, or
`"unknown"`), UTC timestamp, the model list, the sha256 of every task file, the argv, the
seed, the mock flag, the decoding table, the Python version, `records_written`,
`records_skipped`, and `task_sha256_drift` (task files that changed since the previous run of
the same out file). The manifest is written once at start (`status: running`) and once at
completion (`status: complete`), so a killed run still leaves evidence of what it was.

The report's `## Provenance` section prints these manifest fields; combined with the record
files, every number in `results/REPORT.md` is recomputable offline.

---

## 6. Stopping rules (pre-committed)

The design is a fixed factorial. There is no optional stopping and no peeking at partial
results for significance: a run either completes its unit list or is stopped by one of the
mechanical conditions below, and the record file shows exactly how far it got.

1. **Full factorial or abort.** Every unit in the design is attempted. A stopped run is
   resumed with `--resume`; the missing units are exactly the absent keys.
2. **Consecutive-failure abort.** Ten consecutive units raising the same exception stops the
   run (`status: aborted_consecutive_errors`) instead of burning the budget. A single
   exception is recorded and the run continues: one odd task is data, a hundred is a bug.
3. **Transport-failure rule (operator).** If the report's `http_status` distribution shows
   more than 20% non-2xx responses in an experiment (mock: `http_status == 0` is expected),
   the run is not a sample. Discard it to a scratch directory, fix the transport, re-run. No
   partial data is analysed because the missing units are unlikely to be missing at random.
4. **Task-sha and config drift.** If `task_sha256_drift` is non-empty on resume, task files
   changed under the run. If `config_drift` is non-empty, the seed, the model list or the
   mock flag changed. In both cases the runner warns and records the drift in the manifest;
   mixed conditions are not pooled, so the run finishes into a fresh `--out` directory.
5. **Mock is never pooled with real.** Mock artifacts live under `results/mock/`, are
   produced by canned replies keyed on (model, messages), and are only read by
   `summarize.py --mock`. No report may mix the two. Mock mode does not vary with
   temperature or seed (neither is part of the mock key), so the decoding axis is degenerate
   by construction in mock data — a property of the fixture, not a finding.
6. **No mid-run prompt edits.** If a prompt is changed for any reason, the affected
   experiment is re-run from unit zero into a new results directory. The prompt text is
   reconstructible from `git_sha` + task hashes; a mid-run edit makes two incompatible
   instruments look like one.
7. **Void, not undefined.** If UNKNOWN exceeds 20% in E3 (or an experiment is aborted with
   fewer than half the units), the endpoint is reported as **void** with the reason, not as
   a rate over the units that happened to succeed.

---

## 7. Analysis policy

- **Wilson** score intervals rather than Wald: rates sit near 0 or 1 and n is tens, where Wald
  intervals leave the unit interval and collapse at 0/1. `n = 0` renders as `n/a`, never as
  `0%`.
- **Exact McNemar** (two-sided binomial on discordant pairs) for all paired comparisons,
  because concordant pairs dominate and discordant counts are single digits. Verified against
  `scipy.stats.binomtest` at build time; the report needs no scipy.
- **Fisher exact** (two-sided, hypergeometric enumeration) for the unpaired 2x2, because
  expected counts are small. Verified against `scipy.stats.fisher_exact` at build time.
- **Cohen's κ** for the weak/obligation double labels, since raw agreement is inflated by the
  base rate of rejected candidates; degenerate convention documented in the function.
- **No multiplicity correction.** Three experiments, many endpoints, small n: p-values are
  descriptive flags and are reported unadjusted and exact. The pre-committed direction rules
  in §2 are the guard against post-hoc pattern hunting, not a family-wise claim.
- **Every number recomputed** from the JSONL by `analysis/summarize.py`; no constant in the
  report. The report prints "no records" rather than a plausible-looking zero.
- **Nulls are results.** A flat envelope effect, a zero false-accept rate, or a converse rate
  of zero are all reportable outcomes and are written up with the same care as the
  confirmations.

---

## 8. What each result would show — and would not show

**E1 — envelope.**
A drop in joint success from `strict` to `toolcall`, consistent across models with a
significant paired test, *shows* that this pipeline loses usable patches for reasons that are
not about the model's security reasoning: the same model, on the same task, at the same
decoding, produces a patch under one output contract and no patch under another.
It does *not* show that the models "understand" less under a diff format, that the loss is
recoverable with retries, or that the ordering generalises to other tasks. Envelopes differ
in output format *and* in instruction text, so the causal factor is not isolated (§9, T3).

**E2 — grader adequacy.**
A high weak false-accept rate with a low obligation false-reject rate *shows* that a
test-case-only reward and an obligation-aware reward are different instruments, and quantifies
how differently they score the same fixed candidate corpus. It does *not* show that the
obligation grader is correct: its rules are a program, and E2-E exists precisely to
demonstrate shapes those rules miss. It also does not measure model behaviour — the corpus is
hand-built, and the E1-patch subset is the only model-derived slice, reported separately.

**E3 — polarity.**
A converse-guard rate that moves with the stated polarity fact *shows* that some failures are
orientation failures rather than competence failures, and that the orientation is
prompt-controllable on matched tasks. It does *not* show that a converse guard in a real patch
would have the same cause, that the classifier's labels are right (the audit log is the
check), or that "just say the fact" is a safe mitigation in production — a model that needs
the fact stated may still be wrong when the fact is contested or absent.

**Across all three.** Nothing here ranks the three models. Multiple endpoints, three models,
12 tasks: a model that looks better here is "better on this corpus under these conventions",
not a general deployment or capability ranking.

---

## 9. Threats to validity

| id | threat | how it is bounded here |
|---|---|---|
| T1 | Envelope parser bugs masquerade as model failures | E1-E counts strict ⊆ fence violations and reports them as instrument defects |
| T2 | Grader bugs masquerade as model quality | reference/alternate positive controls (E2-B); every record keeps obligation `reasons` for triage |
| T3 | Envelopes differ in prompt text, not only format | declared, not solved: E1-C (conditional on admission) separates "cannot produce" from "produced wrong", but the instruction-vs-format confound remains |
| T4 | Prompt leakage (markers, reference fix) | the polarity markers and the reference fix never appear in any prompt |
| T5 | Mock treated as evidence | separate directory, `mock` flag in the manifest, report banner, degenerate decoding axis documented |
| T6 | Vendor nondeterminism / routing | seeds recorded, not claimed as guarantees; re-runs are new samples |
| T7 | Contamination (models saw similar code) | acknowledged: the tasks are proxies and no train/test split is possible at 12 tasks |
| T8 | Task authoring errors | positive controls turn authoring bugs into visible false rejects, triaged by reason |
| T9 | Multiple comparisons | pre-committed direction rules; p-values descriptive and unadjusted |
| T10 | One serving path | scope statement: results describe this aggregator's endpoint |
| T11 | E2 mutations are syntactic | per-family statements are about the mutation family; no semantic reach claim |
| T12 | E3 classifier is lexical | every expression logged; a rate is only reported alongside its audit table |
| T13 | Resume mixing incompatible runs | unit keys + `task_sha256_drift` + git sha in the manifest; stopping rule 4/6 |

---

## 10. Reproducibility

- Fixed seeds per decoding, recorded per record and per manifest.
- Deterministic prompts: no timestamps, no randomness, stable ordering, and the case list is
  rendered from the task file.
- `--resume` makes a long VPS run safe to interrupt; the missing units are exactly the absent
  keys.
- `make test` is offline and needs neither network nor key: it asserts the statistics against
  scipy-verified reference values, the JSONL schema, the manifest fields, resume idempotence
  (byte-identical file, `records_written == 0`), and that the report contains its three
  endpoint tables.
- `make dryrun` exercises the full pipeline against canned mock replies. It proves the
  plumbing, and nothing about any model.
- The three runners inline their shared scaffolding (CLI, resume, manifest) so each script is
  self-contained. The test suite asserts all three expose the same contract, so drift fails a
  test rather than quietly producing three different experiments.
