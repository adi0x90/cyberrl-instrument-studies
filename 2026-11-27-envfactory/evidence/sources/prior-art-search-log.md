# Prior-art negative searches — raw search log (ATT-2966, 27 Nov 2026)

Agent: source-verifier. Tool: `advanced_search` (plugin engine override where noted).
Search snippets are NOT evidence; hits below are recorded to establish *existence*, and only
pages I then fetched into `evidence/sources/` are cited as verified. Fetch helper: `evidence/sources/_fetch.sh`.

## §4 item 3 — two-tier / async verifier in cyber RL

| # | Query (verbatim) | Engine requested | Engine actually used | Result summary |
|---|---|---|---|---|
| 3.1 | `tiered verifier cyber RL` | ddg | ddg | Hits: arXiv 2602.00513 *Minerva: RLVR for Cyber Threat Intelligence*; arXiv 2611.07639 *Measuring the Marginal Value of Verifier-Based RL*; PrimeIntellect-ai/verifiers; hud.ai "Verifier and Reward Design for RL Environments"; generic RLVR explainers |
| 3.2 | `async verifier reward drift training loop` | ddg | ddg | Hits: verl "Reward Loop" docs; VeRL-Omni "Async Reward" (sample-level streaming reward / HTTP scorer); **RTD — Replayable Trajectory Dataflow (COLM 2026)**; PeterWrighten/rtd repo |
| 3.3 | `verifier regression gate RL environment` | ddg | ddg | Hits: PrimeIntellect-ai/verifiers; arXiv 2610.19298 *Verifier-Grounded Agentic RL for Provably Correct Code*; RL-environment explainers (no regression-gate artifact) |
| 3.4 | `two-tier verifier synchronous reward asynchronous audit cyber reinforcement learning environment` | deepseek-official | deepseek-official | Hits: ASG-SI (arXiv 2512.23760) "designed to integrate with asynchronous agentic RL infrastructures where execution and training are decoupled"; HF Space "Dual-grader reward system with independent sub-functions and anti-hacking verifier"; arXiv 2609.09219 |

## §4 item 4 — recertification / trust-score artifact

| # | Query (verbatim) | Engine requested | Engine actually used | Result summary |
|---|---|---|---|---|
| 4.1 | `environment certification RL` | ddg | **tavily (ddg rate-limited/anti-bot)** | Junk: environmental/EHS professional certifications. No RL hit. |
| 4.2 | `capability-indexed environment audit` | ddg | **tavily (ddg rate-limited)** | Junk: SOC audit, environmental compliance audit, Ecovadis. No RL hit. |
| 4.3 | `environment trust score decay` | ddg | **tavily (ddg rate-limited)** | Junk: "natural environment", dictionary definitions. No RL hit. |
| 4.4 | `RL training environment recertification audit trigger when policy capability improves reward hacking` | deepseek-official | deepseek-official | Hits: arXiv 2611.32037 *Online Environment Repair for Coding Agents*; Zenodo 22763022 *A Reproducible Weak-RSI Loop: From Behavioral Audit to Recursive Training*; arXiv 2611.09828; **NeurIPS 2026 *The Verifier Gap: A Model-Free Audit Predicts Reward Hacking Before Training***; arXiv 2609.35677 *Verifier Errors in RLVR*; HF dataset ai-safety-institute/reward-hacking-sdf-default |
| 4.5 | `environment trust score decay reinforcement learning training environment` | anysearch | anysearch | Generic RL tutorials / vendor explainers. No artifact hit. |
| 4.6 | `certification stamp re-audit RL environment pass-rate band regression gate published mechanism` | deepseek-official | deepseek-official | Hits: Zenodo *Written Down, Never Checked: Auditing a Reinforcement-Learning Environment, Its Research Pipeline, and the Audit Itself*; HF repo2rlenv release notes "graded F2P/P2P reward + scale audit"; arXiv 2610.08319 ("2,132 certify (60.29%, Wilson 95% 58.67–61.89)"); arXiv 2610.33451 (audit results written into a "recipe stamp") |

## Search limits (read before citing "confirmed absence")

1. **ddg was rate-limited mid-run.** Queries 4.1–4.3 were silently re-routed to tavily, which
   interpreted "environment" as the natural environment and returned unrelated results. Those
   three rows are *engine failures*, not evidence of absence.
2. The doc's literal phrasings — "tiered verifier", "capability-indexed environment audit",
   "environment trust score decay" — return **zero** RL-relevant hits *as exact phrases*. The
   concept-space queries (3.4, 4.4, 4.6) return relevant artifacts under different names.
3. Search coverage is one session, one date (27 Nov 2026), 10 queries. Not exhaustive; not a
   substitute for a venue-level sweep (NeurIPS/ICML/COLM/ACL proceedings, 2025–2026).
4. Mechanism, not name: the "almost entirely open" verdict in §2.1 Stage 3 is **not**
   supported as stated once *The Verifier Gap* (pre-training model-free audit) and *Online
   Environment Repair* (runtime environment repair) are counted.
