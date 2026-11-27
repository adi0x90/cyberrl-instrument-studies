# Run logs — permanently unavailable

The RunPod run logs (`gen7b.log`, `gen14b.log`, `genapi.log`, `score7b.log`, `score14b.log`,
`scoreapi.log`, `download.log`, smoke logs) lived at `/workspace/grpo/` on pod
`grpo-a100-80gb` (`nbyu2eaqnlawqt`).

The pod was stopped before they were pulled, and **terminated on 2026-11-27** to stop the
$60/month volume charge. The logs are therefore gone and cannot be regenerated from this
archive.

What survives is what matters: the 1,920 raw completions (`evidence/bon/bon_*.jsonl`), the
2,880 scored verdicts (`evidence/bon/scored_*.jsonl`), the API spend accounting embedded in
those files, and every result derived from them. The logs only described progress, not outcomes.
