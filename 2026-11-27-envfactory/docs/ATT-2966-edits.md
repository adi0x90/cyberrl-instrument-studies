# ATT-2966 — Required edits to `01-RESEARCH-AND-LANDSCAPE.md`

**Produced:** 27 November 2026 by `source-verifier`.
**Applies to:** `/Users/adi/Desktop/attify-strategy-ops/aditya-gupta-positioning/jobs/anthropic-rl-scientist-engineer-cyber/part3-cyber-rl-environment-factory/01-RESEARCH-AND-LANDSCAPE.md`
**The target file was not modified.** This document contains the exact find/replace pairs; apply them by hand or with the guarded script in Appendix A.

**How to read a block:** `FIND` is the exact current text (single-line unless marked). `REPLACE` is what it becomes. Line numbers are from the 237-line file as of 27 Nov 2026 and are hints only — match on text, not line number.

**Counts:** 12 `[F]`, 3 `[S-UNVERIFIED]`, 1 `CUT` across the 16 table-C rows; all 6 anchors verified, 2 needing quote repairs.

---

## Part 1 — Marker changes (§1–§3, inline `[S]` → verified tag)

### 1.1 §1.2 — Auditing Reward Hackability (line 33)

**FIND**
```
https://arxiv.org/abs/2606.16062)) **[S]** — so Part 3 must compare
```
**REPLACE**
```
https://arxiv.org/abs/2606.16062)) **[F]** (fetched 27 Nov 2026: 22.4% of 49 SWE-bench Verified tasks accept a Docker-verified patch that behaves differently from the gold patch) — so Part 3 must compare
```

### 1.2 §1.3 — Mozilla (line 37): Anthropic link

**FIND**
```
mozilla-firefox-security)) **[S]** — confirm quote wording on fetch; corroborated by
```
**REPLACE**
```
mozilla-firefox-security)) **[F]** (fetched 27 Nov 2026: 22 vulnerabilities / 14 high-severity confirmed, and the task-verifier sentence is verbatim; the 6 Mar 2026 date comes from the page's `datePublished` metadata, not the body prose); corroborated by
```

### 1.3 §1.3 — Mozilla (line 37): Mozilla link

**FIND**
```
anthropic-red-team/)) **[S]**.
```
**REPLACE**
```
anthropic-red-team/)) **[F]** (Mozilla's own count: "14 high-severity bugs and issued 22 CVEs"; it does not restate the task-verifier definition).
```

### 1.4 §2.1 Stage 1 — EnvCraft (line 78)

**FIND**
```
EnvCraft auto-synthesis of 139 executable environments / ~20k tasks incl. finite-state-machine and firmware-style scenarios ([arXiv:2609.05576](https://arxiv.org/abs/2609.05576)) **[S]**
```
**REPLACE**
```
EnvCraft auto-synthesis of 139 interactive environments / ~20K tasks incl. a constraint-guided finite-state-machine family and an IoT device/sensor-management environment ([arXiv:2609.05576](https://arxiv.org/abs/2609.05576)) **[F]** (fetched 27 Nov 2026 — the paper contains no firmware, embedded, hardware, PLC, SCADA or OT scenario family, so "firmware-style" is cut)
```

### 1.5 §2.1 Stage 2 — Mozilla dual verifier marker (line 79)

**FIND**
```
Mozilla's dual task verifier (vuln-gone + regression) **[S]**
```
**REPLACE**
```
Mozilla's dual task verifier (vuln-gone + regression) **[F]**
```

### 1.6 §2.1 Stage 2 — LLMs Gaming Verifiers marker (line 79)

**FIND**
```
([arXiv:2604.15149](https://arxiv.org/abs/2604.15149)) **[S]**; Aditya's Part 1
```
**REPLACE**
```
([arXiv:2604.15149](https://arxiv.org/abs/2604.15149)) **[F]**; Aditya's Part 1
```

### 1.7 §2.1 Stage 5 — cut the Krova vendor figure (line 82)

**FIND**
```
; Firecracker snapshot-restore ~24.5 ms p50 bare-metal vs ~642 ms cold boot ([2026 microVM benchmark](https://krova.cloud/blog/firecracker-microvm-benchmarks-real-performance-2026)) **[S]**
```
**REPLACE**
```

```
*(Delete the whole clause including the leading `; `. The Krova page contains neither 24.5 nor 642; it reports a 125 ms cold start and explicitly says Firecracker's snapshot/restore is not in the core. Use the DeltaBox-table figures instead — already present in the DeltaBox clause.)*

### 1.8 §2.1 Stage 5 — DeltaBox precision + marker (line 82)

**FIND**
```
DeltaBox millisecond CRIU+overlayfs checkpoint/rollback (1.86 ms fast restore; Firecracker 200 ms–2 s checkpoint / 120–700 ms restore) ([arXiv:2605.22781](https://arxiv.org/html/2605.22781)) **[S]**
```
**REPLACE**
```
DeltaBox 10.83 ms checkpoint hidden under inference + 1.86 ms template-fork restore (the CRIU lazy-pages fallback is 9.29 ms; the paper's own comparison table gives Firecracker VM snapshot 200 ms–2 s checkpoint / 120–700 ms restore) ([arXiv:2605.22781](https://arxiv.org/html/2605.22781)) **[F]**
```

### 1.9 §2.1 Stage 5 — BunshinRL baseline + marker (line 82)

**FIND**
```
−23% end-to-end rollout latency, up to −43% peak memory vs CRIU forks ([arXiv:2611.10453](https://arxiv.org/html/2611.10453v1)) **[S]**
```
**REPLACE**
```
−23% end-to-end rollout latency, up to −43% peak memory vs AgentENV on WebArena (−30% vs CRIU on OSWorld) ([arXiv:2611.10453](https://arxiv.org/html/2611.10453v1)) **[F]**
```

### 1.10 §2.1 Stage 5 — Harbor / Prime split (line 82)

**FIND**
```
Harbor's verifier contract (`tests/test.sh` → `reward.txt`) and Prime Intellect's async taskset/harness/runtime decomposition as public rollout infrastructure ([harbor](https://github.com/harbor-framework/harbor), [verifiers v1](https://github.com/PrimeIntellect-ai/verifiers)) **[S]**
```
**REPLACE**
```
Prime Intellect's async taskset/harness/runtime decomposition as public rollout infrastructure ([verifiers v1](https://www.primeintellect.ai/blog/verifiers-v1)) **[F]**; Harbor as a public eval/rollout framework ([harbor](https://github.com/harbor-framework/harbor)) **[F]**, though its stated `tests/test.sh` → `reward.txt` reward contract is **[S-UNVERIFIED]** — not present in the fetched README
```

### 1.11 §3.1 — LLMs Gaming Verifiers marker (line 103)

**FIND**
```
[arXiv:2604.15149](https://arxiv.org/abs/2604.15149) **[S]** |
```
**REPLACE**
```
[arXiv:2604.15149](https://arxiv.org/abs/2604.15149) **[F]** |
```

### 1.12 §3.3 item 2 — EnvCraft / CRLF citation (line 115)

**FIND**
```
([I]; EnvCraft **[S]**, [CRLF](https://link.springer.com/chapter/10.1007/978-3-032-16089-8_27) **[S]**)
```
**REPLACE**
```
([I]; EnvCraft **[F]** — finite-state-machine scenarios only, no firmware; [CRLF](https://link.springer.com/chapter/10.1007/978-3-032-16089-8_27) **[S-UNVERIFIED]** — Springer served a bot challenge on both HTML and PDF, so OT-pentest RL prior art remains unconfirmed)
```

---

## Part 2 — Claim corrections (quotes and numbers that the source contradicts or does not support)

### 2.1 §1.5 — the OT quotation is not in the source (line 47)

**FIND**
```
it notes these systems "can be difficult to take offline and risky to change".
```
**REPLACE**
```
it notes these systems "often cannot be taken offline to patch" and that "changes are risky".
```

### 2.2 §1.7 — drop "shortcut" (line 60)

**FIND**
```
and warns that rewards for unintended shortcuts can generalize beyond their original environment.
```
**REPLACE**
```
and warns that rewards for unintended workarounds can generalize beyond their original environment (the page's wording: "If training rewards something we didn't intend—such as finding loopholes or working around a restriction—the model learns that the workaround pays off and may then apply it elsewhere.").
```

### 2.3 §3.3 item 2 — same bad quotation (line 115)

**FIND**
```
that are "difficult to take offline and risky to change" **[F]**
```
**REPLACE**
```
that "often cannot be taken offline to patch" and where "changes are risky" **[F]**
```

### 2.4 §2.1 Stage 2 — narrow the open-cell claim (line 79)

The §4.3 negative search is not clean: Minerva (arXiv 2602.00513) is a cyber-domain RLVR verifier pipeline with task-specific verifiers.

**FIND**
```
has no cyber-domain implementation in the public record;
```
**REPLACE**
```
has no published cyber-domain *Tier-2 drift instrument* in the public record — a cyber-domain RLVR verifier pipeline does exist (Minerva, [arXiv:2602.00513](https://arxiv.org/abs/2602.00513)), so this must stay a claim about the two-tier in-loop instrument, not about cyber verification generally;
```

### 2.5 §2.1 Stage 3 — narrow "Almost entirely open" (line 80)

The §4.4 negative search is not clean: a model-free pre-training verifier audit (NeurIPS 2026) and runtime environment repair (arXiv 2611.32037) both exist.

**FIND**
```
Nothing in the public record implements this; CATCH measures monitor erosion but has no recertification gate
```
**REPLACE**
```
Adjacent mechanisms exist but none is a capability-indexed recertification gate for cyber RL: a model-free pre-training verifier audit predicts reward hacking before training ([The Verifier Gap, NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/169869)), online environment repair fixes verifier/harness defects at runtime ([arXiv:2611.32037](https://arxiv.org/abs/2611.32037)), and CATCH measures monitor erosion but has no recertification gate
```

### 2.6 §2.1 Stage 3 — verdict cell (line 80)

**FIND**
```
| **Almost entirely open.** This is the stage closest
```
**REPLACE**
```
| **Substantially open, not empty.** Adjacent audit/repair mechanisms exist; none indexes certification to policy capability. This is the stage closest
```

---

## Part 3 — §5 table C, line-by-line (adds a 5th "Verdict" column)

### 3.0 Header and separator (lines 218–219)

**FIND**
```
| Source | URL | What it claims | Risk if unfetched |
|---|---|---|---|
```
**REPLACE**
```
| Source | URL | What it claims | Risk if unfetched | Verdict (fetched 27 Nov 2026) |
|---|---|---|---|---|
```

### 3.1 (line 220) Anthropic — Mozilla

**FIND**
```
| Anthropic — Partnering with Mozilla to improve Firefox's security | https://www.anthropic.com/news/mozilla-firefox-security | 6 Mar 2026; Opus 4.6, 22 vulns / 14 high-sev; task-verifier definition | Load-bearing for Stage 2 |
```
**REPLACE**
```
| Anthropic — Partnering with Mozilla to improve Firefox's security | https://www.anthropic.com/news/mozilla-firefox-security | 6 Mar 2026; Opus 4.6, 22 vulns / 14 high-sev; task-verifier definition | Load-bearing for Stage 2 | **[F]** — 22/14 and the definition sentence verbatim; date from page metadata |
```

### 3.2 (line 221) Mozilla blog

**FIND**
```
| Mozilla — Hardening Firefox with Anthropic's Red Team | https://blog.mozilla.org/en/firefox/hardening-firefox-anthropic-red-team/ | Corroborating Mozilla-side account | Low |
```
**REPLACE**
```
| Mozilla — Hardening Firefox with Anthropic's Red Team | https://blog.mozilla.org/en/firefox/hardening-firefox-anthropic-red-team/ | Corroborating Mozilla-side account | Low | **[F]** — corroborates "14 high-severity bugs and issued 22 CVEs"; no verifier text |
```

### 3.3 (line 222) EnvCraft

**FIND**
```
| EnvCraft — Synthesizing Executable Environments in Agentic RL | https://arxiv.org/abs/2609.05576 | 139 environments / ~20k tasks; FSM + firmware-style scenarios | Stage 1 prior-art claim |
```
**REPLACE**
```
| EnvCraft — Synthesizing Executable Environments in Agentic RL | https://arxiv.org/abs/2609.05576 | 139 environments / ~20K tasks; constraint-guided finite-state-machine family; IoT device/sensor-management environment | Stage 1 prior-art claim | **[F]** — 139/~20K and FSM confirmed; **"firmware-style" cut** (0 hits for firmware/embedded/hardware/PLC/SCADA) |
```

### 3.4 (line 223) CRLF

**FIND**
```
| CRLF — Sim2Real RL environment for automated IT/OT pentesting | https://link.springer.com/chapter/10.1007/978-3-032-16089-8_27 | OT-layer pentest RL environment | Stage 1/4 prior-art check |
```
**REPLACE**
```
| CRLF — Sim2Real RL environment for automated IT/OT pentesting | https://link.springer.com/chapter/10.1007/978-3-032-16089-8_27 | OT-layer pentest RL environment | Stage 1/4 prior-art check | **[S-UNVERIFIED]** — bot wall ("Client Challenge") on HTML and PDF; `web_fetch` blocked by cross-origin redirect to idp.springer.com. Do not cite. |
```

### 3.5 (line 224) LLMs Gaming Verifiers

**FIND**
```
| LLMs Gaming Verifiers (extensional vs isomorphic verification) | https://arxiv.org/abs/2604.15149 | Property-based verifier failure modes | Stage 2 support |
```
**REPLACE**
```
| LLMs Gaming Verifiers (extensional vs isomorphic verification) | https://arxiv.org/abs/2604.15149 | Property-based verifier failure modes | Stage 2 support | **[F]** — "extensional"/"isomorphic" and Isomorphic Perturbation Testing confirmed |
```

### 3.6 (line 225) Auditing Reward Hackability

**FIND**
```
| Auditing Reward Hackability in Code RL Training Environments | https://arxiv.org/abs/2606.16062 | Rates of open code-RL environments accepting wrong patches | Stage 2 support |
```
**REPLACE**
```
| Auditing Reward Hackability in Code RL Training Environments | https://arxiv.org/abs/2606.16062 | Rates of open code-RL environments accepting wrong patches | Stage 2 support | **[F]** — "22.4% of 49 SWE-bench Verified tasks accept a Docker-verified patch that behaves differently from the gold patch" |
```

### 3.7 (line 226) DeltaBox

**FIND**
```
| DeltaBox — millisecond checkpoint/rollback agent sandbox | https://arxiv.org/html/2605.22781 | CRIU+overlayfs 1.86 ms restore; Firecracker comparison | Stage 5 numbers |
```
**REPLACE**
```
| DeltaBox — millisecond checkpoint/rollback agent sandbox | https://arxiv.org/html/2605.22781 | 10.83 ms checkpoint; 1.86 ms template-fork restore (CRIU fallback 9.29 ms); Firecracker 200 ms–2 s / 120–700 ms comparison | Stage 5 numbers | **[F]** — but 1.86 ms is the fork fast path, **not CRIU**; do not label it "CRIU" |
```

### 3.8 (line 227) BunshinRL

**FIND**
```
| BunshinRL — shared-state fork rollout system | https://arxiv.org/html/2611.10453v1 | −23% latency, −43% peak memory vs CRIU forks | Stage 5 numbers |
```
**REPLACE**
```
| BunshinRL — shared-state fork rollout system | https://arxiv.org/html/2611.10453v1 | −23% end-to-end rollout latency; up to −43% peak memory vs AgentENV (−30% vs CRIU) | Stage 5 numbers | **[F]** — 43% baseline is AgentENV/WebArena, **not CRIU** |
```

### 3.9 (line 228) Krova

**FIND**
```
| Krova — Firecracker microVM benchmarks 2026 (vendor blog) | https://krova.cloud/blog/firecracker-microvm-benchmarks-real-performance-2026 | 24.5 ms p50 snapshot restore vs 642 ms cold boot | Vendor blog — replace or mark anecdotal |
```
**REPLACE**
```
| ~~Krova — Firecracker microVM benchmarks 2026 (vendor blog)~~ | https://krova.cloud/blog/firecracker-microvm-benchmarks-real-performance-2026 | **CUT — claim not in source.** Page has no snapshot-restore figure and no 642 ms; it reports 125 ms cold start, <5 MB/instance, 4500+ concurrent VMs, and states snapshot/restore "are not baked into the core" | Would be a fabricated number if kept | **CUT** |
```

### 3.10 (line 229) DigitalOcean

**FIND**
```
| DigitalOcean — microVM architecture (warm restore design) | https://www.digitalocean.com/blog/microvm-architecture | <200 ms warm path design | Vendor blog — same caution |
```
**REPLACE**
```
| DigitalOcean — microVM architecture (warm restore design) | https://www.digitalocean.com/blog/microvm-architecture | <200 ms warm path, p50, their internal testing | Vendor blog — same caution | **[F]** — "under 200 ms at p50 in our internal testing"; keep the "internal testing" qualifier and the exclusion of cold caches |
```

### 3.11 (line 230) Harbor

**FIND**
```
| Harbor framework (verifier reward contract) | https://github.com/harbor-framework/harbor | `tests/test.sh` → `reward.txt` contract; rewardkit | Infrastructure citation |
```
**REPLACE**
```
| Harbor framework (verifier reward contract) | https://github.com/harbor-framework/harbor | `tests/test.sh` → `reward.txt` contract; rewardkit | Infrastructure citation | **[S-UNVERIFIED]** for the contract — `reward` has 0 hits in the fetched README (repo itself is real, Apache-2.0, 7.5k★) |
```

### 3.12 (line 231) Prime Intellect

**FIND**
```
| Prime Intellect — verifiers v1 (taskset/harness/runtime) | https://github.com/PrimeIntellect-ai/verifiers + https://www.primeintellect.ai/blog/verifiers-v1 | Async RL environment SDK decomposition | Infrastructure citation |
```
**REPLACE**
```
| Prime Intellect — verifiers v1 (taskset/harness/runtime) | https://www.primeintellect.ai/blog/verifiers-v1 | Async RL environment SDK decomposition | Infrastructure citation | **[F]** — "decomposes environments into composable tasksets, harnesses, and runtimes"; cite the blog, since `taskset` is not on the GitHub landing page |
```

### 3.13 (line 232) COLM rtd

**FIND**
```
| Replayable trajectory dataflows (COLM '26) | https://github.com/PeterWrighten/rtd | Rescoring/diffing trajectories under verifier change | Stage 3 support |
```
**REPLACE**
```
| Replayable trajectory dataflows (COLM '26) | https://github.com/PeterWrighten/rtd | Rescoring/diffing trajectories under verifier change | Stage 3 support | **[F]** — "[COLM26] [AIMS] Replayable Trajectory Dataflow: Auditing Reward Measurement under Optimization and Drift in RLVR"; rescore/diff/localize confirmed |
```

### 3.14 (line 233) 2605.02909

**FIND**
```
| Evaluating the Impact of Systematic Verification Error on RLVR | https://arxiv.org/html/2605.02909v2 | Systematic vs random verifier error effects | Stage 2 support |
```
**REPLACE**
```
| Evaluating the Impact of Systematic Verification Error on RLVR | https://arxiv.org/html/2605.02909v2 | Systematic vs random verifier error effects | Stage 2 support | **[F]** — "Delay, Plateau, or Collapse…"; systematic false positives → plateau/collapse, false negatives ≈ random noise |
```

### 3.15 (line 234) AgentV-RL

**FIND**
```
| Agentic Verifier (AgentV-RL, ACL Findings 2026) | https://aclanthology.org/2026.findings-acl.1156.pdf | Two-agent adversarial reward modeling | Stage 2 support |
```
**REPLACE**
```
| Agentic Verifier (AgentV-RL, ACL Findings 2026) | https://aclanthology.org/2026.findings-acl.1156.pdf | Two-agent (forward + backward) agentic verifier reward modeling | Stage 2 support | **[F]** — "coordinates two specialized agents: the forward agent and the backward agent"; **"adversarial" is not the paper's framing** (0 hits) |
```

### 3.16 (line 235) NitroBox

**FIND**
```
| NitroBox — fast sandbox for large-scale RL training | https://openreview.net/pdf/8591066d27454d88cf846fa77f0d2dde3159f0a2.pdf | MicroVM sandbox latency trade-offs | Stage 5 support |
```
**REPLACE**
```
| NitroBox — fast sandbox for large-scale RL training | https://openreview.net/pdf/8591066d27454d88cf846fa77f0d2dde3159f0a2.pdf | MicroVM sandbox latency trade-offs | Stage 5 support | **[S-UNVERIFIED]** — HTTP 403 (OpenReview error page), then a "Verifying your browser" challenge via `web_fetch`. Do not cite. |
```

---

## Part 4 — Checklist and ledger-header bookkeeping

### 4.1 §4 item 1 intro (line 131)

**FIND**
```
1. **Fetch and quote-check the six anchor sources** (all currently [F]/[P] except two):
```
**REPLACE**
```
1. **Fetch and quote-check the six anchor sources** (all six fetched 27 Nov 2026 and quote-checked — see `evidence/verification-ledger.md` Table 1):
```

### 4.2 §4 item 1 Mozilla bullet (line 134)

**FIND**
```
**[S→must fetch]** — confirm 6 Mar 2026 date, 22 vulns / 14 high-severity, and the exact task-verifier definition sentence (this is the load-bearing quote for Stage 2).
```
**REPLACE**
```
**[F]** — confirmed 27 Nov 2026: 6 Mar 2026 (page metadata), 22 vulns / 14 high-severity, and the exact task-verifier definition sentence. The load-bearing quote for Stage 2 is safe to use verbatim.
```

### 4.3 §4 item 3 — record the result (line 140)

**FIND**
```
If found, Stage 2's open-cell claim must be narrowed to the cyber-domain specificity. Record either way.
```
**REPLACE**
```
If found, Stage 2's open-cell claim must be narrowed to the cyber-domain specificity. Record either way. **Done 27 Nov 2026 (4 query formulations):** *not* a clean absence — Minerva ([arXiv:2602.00513](https://arxiv.org/abs/2602.00513)) is a cyber-domain RLVR verifier pipeline, and async decoupled verifier infrastructure is published (verl Reward Loop / VeRL-Omni HTTP scorer, ASG-SI 2512.23760). Stage 2's claim was consequently narrowed to the *Tier-2 in-loop drift instrument*; see `evidence/verification-ledger.md` Table 3.
```

### 4.4 §4 item 4 — record the result (line 141)

**FIND**
```
Any published mechanism narrows Stage 3's "almost entirely open" verdict.
```
**REPLACE**
```
Any published mechanism narrows Stage 3's "almost entirely open" verdict. **Done 27 Nov 2026 (6 query formulations; 3 returned only environmental-certification noise after a ddg rate-limit fallback):** adjacent mechanisms do exist — *The Verifier Gap: A Model-Free Audit Predicts Reward Hacking Before Training* (NeurIPS 2026) and *Online Environment Repair for Coding Agents* ([arXiv:2611.32037](https://arxiv.org/abs/2611.32037)). The verdict was downgraded to "substantially open, not empty".
```

### 4.5 §5 preamble (line 150)

**FIND**
```
Only sources actually verified to date. **Do not carry [S] rows into the published post without fetching.**
```
**REPLACE**
```
Only sources actually verified to date. **Do not carry [S] rows into the published post without fetching.** Table C was fetched and quote-checked on 27 Nov 2026 under ATT-2966 (`evidence/verification-ledger.md`): 12 `[F]`, 3 `[S-UNVERIFIED]`, 1 `CUT`.
```

### 4.6 Table C heading (line 216)

**FIND**
```
### C. Surfaced in 27 Nov 2026 searches — verify body on fetch before quoting [S]
```
**REPLACE**
```
### C. Surfaced in 27 Nov 2026 searches — body fetched and quote-checked 27 Nov 2026 (ATT-2966)
```

### 4.7 Closing line (line 237)

**FIND**
```
**Internal; not for publication.** This memo anchors Part 3's outline and the follow-up Linear research issues; the [S] rows are the research agent's first work items.
```
**REPLACE**
```
**Internal; not for publication.** This memo anchors Part 3's outline and the follow-up Linear research issues; the table-C rows were fetched and quote-checked on 27 Nov 2026 (ATT-2966) — 12 `[F]`, 3 `[S-UNVERIFIED]`, 1 `CUT`. Do not cite the three unverified rows or the cut vendor figure.
```

---

## Appendix A — guarded apply script

Every `FIND` above is unique in the file. To apply mechanically, save this as `apply.py` beside the memo and run `python3 apply.py`; it refuses to write unless every block matches **exactly once**.

```python
import sys, pathlib
P = pathlib.Path("01-RESEARCH-AND-LANDSCAPE.md")
src = P.read_text()
BLOCKS = [
    # (label, find, replace) — paste from Part 1–4 above, in order.
    # 1.1
    ("https://arxiv.org/abs/2606.16062)) **[S]** — so Part 3 must compare",
     "https://arxiv.org/abs/2606.16062)) **[F]** (fetched 27 Nov 2026: 22.4% of 49 SWE-bench Verified tasks accept a Docker-verified patch that behaves differently from the gold patch) — so Part 3 must compare"),
    # ... one tuple per block ...
]
for f, r in BLOCKS:
    n = src.count(f)
    if n != 1:
        sys.exit(f"ABORT: match count {n} for: {f[:70]!r}")
    src = src.replace(f, r)
P.write_text(src)
print("applied", len(BLOCKS), "blocks")
```

---

## Appendix B — optional precision tweaks (not marker changes)

1. **§1.1 (line 26)** is already `[F]` and substantively correct. Optional: change `"more than 10%"` phrasing to match the source exactly — the post says **"over 10% of environments in our production mix"**. No change needed to the `[F]` marker.
2. **§5 table A row for the reward seeker** labels the trio "(80 envs, 40%, 41%)". Optional clarity: "(80 envs; 40% of episodes hacked; 41% reward-tampering rate)" — the 41% is a tampering rate and the unlabelled figure invites misreading.
3. **§1.3 (line 37)** says "nearly a fifth of all high-severity Firefox vulnerabilities remediated in 2025" — this is the Anthropic post's own claim and is quoted accurately; keep as Anthropic's claim, not an independent fact.
4. **Dates from metadata** (Mozilla, 6 Mar 2026): if the published post needs a visible date, cite the page's `datePublished` / `article:published_time` rather than asserting the body states it.
