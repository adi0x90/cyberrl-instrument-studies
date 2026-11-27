# FILE-MANIFEST

342 files. Generated 2026-11-27. Paths are relative to this directory.

| # | path | type | size |
|---|---|---|---|
| 1 | `./00-INDEX.md` | Markdown doc | 5.8 kB |
| 2 | `./Makefile` | build/config | 0.5 kB |
| 3 | `./README.md` | Markdown doc | 4.9 kB |
| 4 | `./analysis/__init__.py` | Python code | 0.0 kB |
| 5 | `./analysis/build_report.py` | Python code | 14.6 kB |
| 6 | `./analysis/recompute.py` | Python code | 6.9 kB |
| 7 | `./bon/__init__.py` | Python code | 0.0 kB |
| 8 | `./bon/analyze_native.py` | Python code | 4.3 kB |
| 9 | `./bon/generate_api.py` | Python code | 5.0 kB |
| 10 | `./bon/generate_native.py` | Python code | 4.9 kB |
| 11 | `./bon/generate_vllm.py` | Python code | 3.7 kB |
| 12 | `./bon/grade_native.py` | Python code | 8.4 kB |
| 13 | `./bon/models_ext.py` | Python code | 2.8 kB |
| 14 | `./bon/native_prompts.py` | Python code | 3.1 kB |
| 15 | `./bon/prompts.py` | Python code | 3.1 kB |
| 16 | `./bon/quarantine_sim.py` | Python code | 6.1 kB |
| 17 | `./bon/report_b1.py` | Python code | 6.8 kB |
| 18 | `./bon/report_native.py` | Python code | 10.9 kB |
| 19 | `./bon/score_pool.py` | Python code | 2.9 kB |
| 20 | `./bon/select_drift.py` | Python code | 5.8 kB |
| 21 | `./bon/tier2.py` | Python code | 2.3 kB |
| 22 | `./docs/ATT-2966-edits.md` | Markdown doc | 24.0 kB |
| 23 | `./docs/FREEZE.json` | JSON data | 0.5 kB |
| 24 | `./docs/SPEC.md` | Markdown doc | 23.5 kB |
| 25 | `./docs/SPEC.sha256` | hash | 0.1 kB |
| 26 | `./envcert/__init__.py` | Python code | 0.0 kB |
| 27 | `./envcert/cli.py` | Python code | 32.4 kB |
| 28 | `./evidence/a1/a1_latency.json` | JSON data | 8.2 kB |
| 29 | `./evidence/a1/a1_latency_control.json` | JSON data | 0.9 kB |
| 30 | `./evidence/a1/a1_latency_parallel.json` | JSON data | 8.3 kB |
| 31 | `./evidence/a1/a1_loo.json` | JSON data | 0.8 kB |
| 32 | `./evidence/a1/a1_native.json` | JSON data | 4.5 kB |
| 33 | `./evidence/a1/a1_summary.json` | JSON data | 7.1 kB |
| 34 | `./evidence/a1/a1_verdicts.jsonl` | JSONL evidence | 4579.2 kB |
| 35 | `./evidence/a2/a2_tests.json` | JSON data | 3.0 kB |
| 36 | `./evidence/a2/audit-authlib-jwt-dupkey-canonicalization.json` | JSON data | 5.3 kB |
| 37 | `./evidence/a2/audit-coremqtt-publish-varint-payloadlen.json` | JSON data | 5.3 kB |
| 38 | `./evidence/a2/audit-d1.json` | JSON data | 5.8 kB |
| 39 | `./evidence/a2/audit-d2.json` | JSON data | 5.3 kB |
| 40 | `./evidence/a2/audit-etcd-revocation-skew.json` | JSON data | 5.2 kB |
| 41 | `./evidence/a2/audit-fastmcp-canonical-authz-alias.json` | JSON data | 5.3 kB |
| 42 | `./evidence/a2/audit-redis-setrange-grow-undercount.json` | JSON data | 5.3 kB |
| 43 | `./evidence/a2/deployment-drifted.json` | JSON data | 0.6 kB |
| 44 | `./evidence/a2/deployment.json` | JSON data | 0.6 kB |
| 45 | `./evidence/a2/observations-high-pass.jsonl` | JSONL evidence | 3.0 kB |
| 46 | `./evidence/a2/observations.jsonl` | JSONL evidence | 3.0 kB |
| 47 | `./evidence/a2/stamp-verify-fixture.json` | JSON data | 1.4 kB |
| 48 | `./evidence/a3/a3_reset.json` | JSON data | 14.6 kB |
| 49 | `./evidence/bon/bon_api.jsonl` | JSONL evidence | 377.6 kB |
| 50 | `./evidence/bon/bon_qwen14b.jsonl` | JSONL evidence | 173.0 kB |
| 51 | `./evidence/bon/bon_qwen7b.jsonl` | JSONL evidence | 171.2 kB |
| 52 | `./evidence/bon/graded_authlib.jsonl` | JSONL evidence | 159.7 kB |
| 53 | `./evidence/bon/graded_fastmcp.jsonl` | JSONL evidence | 226.6 kB |
| 54 | `./evidence/bon/native_authlib.jsonl` | JSONL evidence | 88.2 kB |
| 55 | `./evidence/bon/native_fastmcp.jsonl` | JSONL evidence | 162.1 kB |
| 56 | `./evidence/bon/scored_api.jsonl` | JSONL evidence | 498.3 kB |
| 57 | `./evidence/bon/scored_qwen14b.jsonl` | JSONL evidence | 118.1 kB |
| 58 | `./evidence/bon/scored_qwen7b.jsonl` | JSONL evidence | 122.8 kB |
| 59 | `./evidence/evidence.b2sum` | hash manifest | 57.8 kB |
| 60 | `./evidence/logs/NOTE.md` | Markdown doc | 0.7 kB |
| 61 | `./evidence/packs/INDEX.json` | JSON data | 2.0 kB |
| 62 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/files/alternate/authlib/oauth2/rfc7523/jwt_bearer.py` | Python code | 12.6 kB |
| 63 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/files/vulnerable/authlib/oauth2/rfc7523/jwt_bearer.py` | Python code | 12.6 kB |
| 64 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/manifest.json` | JSON data | 6.2 kB |
| 65 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/canary-prompt.md` | Markdown doc | 2.0 kB |
| 66 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/injection-spec.md` | Markdown doc | 6.2 kB |
| 67 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/manifest.json` | JSON data | 6.2 kB |
| 68 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/poc/harness.py` | Python code | 5.8 kB |
| 69 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/poc/poc.py` | Python code | 4.8 kB |
| 70 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/poc/run-poc.sh` | shell script | 0.6 kB |
| 71 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/replay.md` | Markdown doc | 4.7 kB |
| 72 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/variants/aud-polyglot/run.sh` | shell script | 0.3 kB |
| 73 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/meta/pack/variants/expired-assertion-replay/run.sh` | shell script | 0.3 kB |
| 74 | `./evidence/packs/authlib-jwt-dupkey-canonicalization/provenance.json` | JSON data | 3.3 kB |
| 75 | `./evidence/packs/coremqtt-publish-varint-payloadlen/files/alternate/source/core_mqtt_serializer.c` | C source | 228.3 kB |
| 76 | `./evidence/packs/coremqtt-publish-varint-payloadlen/files/vulnerable/source/core_mqtt_serializer.c` | C source | 228.5 kB |
| 77 | `./evidence/packs/coremqtt-publish-varint-payloadlen/manifest.json` | JSON data | 11.1 kB |
| 78 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/.gitignore` | build/config | 0.0 kB |
| 79 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/canary-prompt.md` | Markdown doc | 5.6 kB |
| 80 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/injection-spec.md` | Markdown doc | 5.9 kB |
| 81 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/manifest.json` | JSON data | 11.1 kB |
| 82 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/poc/poc.c` | C source | 9.8 kB |
| 83 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/poc/run-poc.sh` | shell script | 1.7 kB |
| 84 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/replay.md` | Markdown doc | 6.0 kB |
| 85 | `./evidence/packs/coremqtt-publish-varint-payloadlen/meta/pack/publish-varint-payloadlen/variants/run-variants.sh` | shell script | 1.9 kB |
| 86 | `./evidence/packs/coremqtt-publish-varint-payloadlen/provenance.json` | JSON data | 3.0 kB |
| 87 | `./evidence/packs/etcd-revocation-skew/files/alternate/server/auth/range_perm_cache.go` | Go source | 7.1 kB |
| 88 | `./evidence/packs/etcd-revocation-skew/files/alternate/server/auth/store.go` | Go source | 34.6 kB |
| 89 | `./evidence/packs/etcd-revocation-skew/files/vulnerable/server/auth/store.go` | Go source | 34.6 kB |
| 90 | `./evidence/packs/etcd-revocation-skew/manifest.json` | JSON data | 6.3 kB |
| 91 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/canary-prompt.md` | Markdown doc | 1.5 kB |
| 92 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/injection-spec.md` | Markdown doc | 5.6 kB |
| 93 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/manifest.json` | JSON data | 6.3 kB |
| 94 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/poc/poc_revocation_skew_test.go` | Go source | 4.7 kB |
| 95 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/poc/run-poc.sh` | shell script | 4.5 kB |
| 96 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/replay.md` | Markdown doc | 4.4 kB |
| 97 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/variants/lib.sh` | shell script | 2.5 kB |
| 98 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/variants/v1-range-write.sh` | shell script | 1.3 kB |
| 99 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/variants/v2-second-session.sh` | shell script | 1.4 kB |
| 100 | `./evidence/packs/etcd-revocation-skew/meta/pack/etcd-revocation-skew/variants/v3-range-middle.sh` | shell script | 1.3 kB |
| 101 | `./evidence/packs/etcd-revocation-skew/provenance.json` | JSON data | 4.2 kB |
| 102 | `./evidence/packs/fastmcp-canonical-authz-alias/files/alternate/fastmcp_slim/fastmcp/server/middleware/authorization.py` | Python code | 21.2 kB |
| 103 | `./evidence/packs/fastmcp-canonical-authz-alias/files/alternate/fastmcp_slim/fastmcp/server/server.py` | Python code | 114.9 kB |
| 104 | `./evidence/packs/fastmcp-canonical-authz-alias/files/vulnerable/fastmcp_slim/fastmcp/server/middleware/authorization.py` | Python code | 20.8 kB |
| 105 | `./evidence/packs/fastmcp-canonical-authz-alias/manifest.json` | JSON data | 7.9 kB |
| 106 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/canary-prompt.md` | Markdown doc | 2.6 kB |
| 107 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/injection-spec.md` | Markdown doc | 5.9 kB |
| 108 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/manifest.json` | JSON data | 7.9 kB |
| 109 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/poc/exploit_canonical_authz.py` | Python code | 3.4 kB |
| 110 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/poc/harness.py` | Python code | 2.7 kB |
| 111 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/poc/run-poc.sh` | shell script | 1.1 kB |
| 112 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/replay.md` | Markdown doc | 4.8 kB |
| 113 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/variants/variant1_user_delete_pair.py` | Python code | 2.3 kB |
| 114 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/variants/variant2_denylist_callable.py` | Python code | 2.6 kB |
| 115 | `./evidence/packs/fastmcp-canonical-authz-alias/meta/pack/fastmcp-canonical-authz-alias/variants/variant3_chained_policy.py` | Python code | 2.7 kB |
| 116 | `./evidence/packs/fastmcp-canonical-authz-alias/provenance.json` | JSON data | 3.7 kB |
| 117 | `./evidence/packs/redis-setrange-grow-undercount/files/alternate/src/t_string.c` | C source | 34.1 kB |
| 118 | `./evidence/packs/redis-setrange-grow-undercount/files/vulnerable/src/t_string.c` | C source | 34.0 kB |
| 119 | `./evidence/packs/redis-setrange-grow-undercount/manifest.json` | JSON data | 8.1 kB |
| 120 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/canary-prompt.md` | Markdown doc | 2.4 kB |
| 121 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/injection-spec.md` | Markdown doc | 6.6 kB |
| 122 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/manifest.json` | JSON data | 8.1 kB |
| 123 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/poc/poc.py` | Python code | 8.5 kB |
| 124 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/poc/run-poc.sh` | shell script | 2.6 kB |
| 125 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/replay.md` | Markdown doc | 3.3 kB |
| 126 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/variants/variant1.sh` | shell script | 0.4 kB |
| 127 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/variants/variant2.sh` | shell script | 0.4 kB |
| 128 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/variants/variant3.sh` | shell script | 0.3 kB |
| 129 | `./evidence/packs/redis-setrange-grow-undercount/meta/pack/setrange-grow-undercount/variants/variant4.sh` | shell script | 0.4 kB |
| 130 | `./evidence/packs/redis-setrange-grow-undercount/provenance.json` | JSON data | 3.5 kB |
| 131 | `./evidence/reference-e2.b2sum` | hash manifest | 0.2 kB |
| 132 | `./evidence/reference-e2.jsonl` | JSONL evidence | 232.8 kB |
| 133 | `./evidence/search-log.md` | Markdown doc | 4.5 kB |
| 134 | `./evidence/sources/_fetch.sh` | shell script | 2.6 kB |
| 135 | `./evidence/sources/agentv-rl-acl-findings.raw.html` | raw HTML | 1034.1 kB |
| 136 | `./evidence/sources/agentv-rl-acl-findings.txt` | text | 140.5 kB |
| 137 | `./evidence/sources/anthropic-alignment-security.raw.html` | raw HTML | 214.4 kB |
| 138 | `./evidence/sources/anthropic-alignment-security.txt` | text | 27.2 kB |
| 139 | `./evidence/sources/anthropic-attack-surface-reduction.raw.html` | raw HTML | 177.1 kB |
| 140 | `./evidence/sources/anthropic-attack-surface-reduction.txt` | text | 18.5 kB |
| 141 | `./evidence/sources/anthropic-cyber-mission.raw.html` | raw HTML | 197.1 kB |
| 142 | `./evidence/sources/anthropic-cyber-mission.txt` | text | 17.5 kB |
| 143 | `./evidence/sources/anthropic-mozilla-firefox.raw.html` | raw HTML | 174.3 kB |
| 144 | `./evidence/sources/anthropic-mozilla-firefox.txt` | text | 14.4 kB |
| 145 | `./evidence/sources/anthropic-reward-seeker.raw.html` | raw HTML | 1019.1 kB |
| 146 | `./evidence/sources/anthropic-reward-seeker.txt` | text | 197.7 kB |
| 147 | `./evidence/sources/anthropic-unintended-model-actions.raw.html` | raw HTML | 187.8 kB |
| 148 | `./evidence/sources/anthropic-unintended-model-actions.txt` | text | 21.3 kB |
| 149 | `./evidence/sources/auditing-reward-hackability-arxiv-2606.16062.raw.html` | raw HTML | 43.2 kB |
| 150 | `./evidence/sources/auditing-reward-hackability-arxiv-2606.16062.txt` | text | 11.1 kB |
| 151 | `./evidence/sources/bunshinrl-arxiv-2611.10453.raw.html` | raw HTML | 141.5 kB |
| 152 | `./evidence/sources/bunshinrl-arxiv-2611.10453.txt` | text | 54.9 kB |
| 153 | `./evidence/sources/colm-rtd-github.raw.html` | raw HTML | 307.8 kB |
| 154 | `./evidence/sources/colm-rtd-github.txt` | text | 16.8 kB |
| 155 | `./evidence/sources/crlf-springer-chapter.raw.html` | raw HTML | 3.0 kB |
| 156 | `./evidence/sources/crlf-springer-chapter.txt` | text | 0.4 kB |
| 157 | `./evidence/sources/crlf-springer-pdf-alt.raw.html` | raw HTML | 3.0 kB |
| 158 | `./evidence/sources/crlf-springer-pdf-alt.txt` | text | 0.4 kB |
| 159 | `./evidence/sources/deltabox-arxiv-2605.22781.raw.html` | raw HTML | 293.3 kB |
| 160 | `./evidence/sources/deltabox-arxiv-2605.22781.txt` | text | 94.0 kB |
| 161 | `./evidence/sources/digitalocean-microvm-architecture.raw.html` | raw HTML | 299.1 kB |
| 162 | `./evidence/sources/digitalocean-microvm-architecture.txt` | text | 39.0 kB |
| 163 | `./evidence/sources/envcraft-arxiv-2609.05576-html.raw.html` | raw HTML | 654.5 kB |
| 164 | `./evidence/sources/envcraft-arxiv-2609.05576-html.txt` | text | 118.6 kB |
| 165 | `./evidence/sources/envcraft-arxiv-2609.05576.raw.html` | raw HTML | 43.2 kB |
| 166 | `./evidence/sources/envcraft-arxiv-2609.05576.txt` | text | 7.8 kB |
| 167 | `./evidence/sources/harbor-framework-github.raw.html` | raw HTML | 279.3 kB |
| 168 | `./evidence/sources/harbor-framework-github.txt` | text | 7.5 kB |
| 169 | `./evidence/sources/harbor-raw-readme.raw.html` | raw HTML | 3.0 kB |
| 170 | `./evidence/sources/harbor-raw-readme.txt` | text | 3.0 kB |
| 171 | `./evidence/sources/krova-firecracker-benchmarks.raw.html` | raw HTML | 196.8 kB |
| 172 | `./evidence/sources/krova-firecracker-benchmarks.txt` | text | 13.3 kB |
| 173 | `./evidence/sources/llms-gaming-verifiers-arxiv-2604.15149.raw.html` | raw HTML | 43.6 kB |
| 174 | `./evidence/sources/llms-gaming-verifiers-arxiv-2604.15149.txt` | text | 8.3 kB |
| 175 | `./evidence/sources/minerva-arxiv-2602.00513.raw.html` | raw HTML | 42.5 kB |
| 176 | `./evidence/sources/minerva-arxiv-2602.00513.txt` | text | 7.9 kB |
| 177 | `./evidence/sources/mozilla-blog-hardening-firefox.raw.html` | raw HTML | 67.3 kB |
| 178 | `./evidence/sources/mozilla-blog-hardening-firefox.txt` | text | 10.9 kB |
| 179 | `./evidence/sources/nitrobox-openreview-pdf.raw.html` | raw HTML | 12.7 kB |
| 180 | `./evidence/sources/nitrobox-openreview-pdf.txt` | text | 2.1 kB |
| 181 | `./evidence/sources/online-env-repair-arxiv-2611.32037.raw.html` | raw HTML | 43.5 kB |
| 182 | `./evidence/sources/online-env-repair-arxiv-2611.32037.txt` | text | 8.4 kB |
| 183 | `./evidence/sources/primeintellect-blog-verifiers-v1.raw.html` | raw HTML | 148.6 kB |
| 184 | `./evidence/sources/primeintellect-blog-verifiers-v1.txt` | text | 17.8 kB |
| 185 | `./evidence/sources/primeintellect-verifiers-github.raw.html` | raw HTML | 255.7 kB |
| 186 | `./evidence/sources/primeintellect-verifiers-github.txt` | text | 6.7 kB |
| 187 | `./evidence/sources/prior-art-search-log.md` | Markdown doc | 4.6 kB |
| 188 | `./evidence/sources/verification-error-rlvr-arxiv-2605.02909.raw.html` | raw HTML | 324.5 kB |
| 189 | `./evidence/sources/verification-error-rlvr-arxiv-2605.02909.txt` | text | 65.8 kB |
| 190 | `./evidence/sources/verifier-gap-neurips.raw.html` | raw HTML | 90.3 kB |
| 191 | `./evidence/sources/verifier-gap-neurips.txt` | text | 4.1 kB |
| 192 | `./evidence/verification-ledger.md` | Markdown doc | 24.6 kB |
| 193 | `./microbench/__init__.py` | Python code | 0.0 kB |
| 194 | `./microbench/reset_bench.py` | Python code | 14.6 kB |
| 195 | `./microbench/run_a3.py` | Python code | 4.8 kB |
| 196 | `./propverifier/__init__.py` | Python code | 0.1 kB |
| 197 | `./propverifier/fastrun.py` | Python code | 3.8 kB |
| 198 | `./propverifier/gen.py` | Python code | 18.8 kB |
| 199 | `./propverifier/grade_property.py` | Python code | 10.4 kB |
| 200 | `./propverifier/harness/__init__.py` | Python code | 0.7 kB |
| 201 | `./propverifier/harness/envelopes.py` | Python code | 19.0 kB |
| 202 | `./propverifier/harness/graders.py` | Python code | 9.8 kB |
| 203 | `./propverifier/harness/models.py` | Python code | 13.3 kB |
| 204 | `./propverifier/harness/polarity.py` | Python code | 9.5 kB |
| 205 | `./propverifier/harness/tasks_io.py` | Python code | 23.9 kB |
| 206 | `./propverifier/latency.py` | Python code | 3.0 kB |
| 207 | `./propverifier/run_a1.py` | Python code | 17.1 kB |
| 208 | `./propverifier/run_a1_native.py` | Python code | 12.7 kB |
| 209 | `./propverifier/tasks/authz_delegated_scope_02.json` | JSON data | 5.6 kB |
| 210 | `./propverifier/tasks/authz_delegated_write_01.json` | JSON data | 5.3 kB |
| 211 | `./propverifier/tasks/bounds_content_length_01.json` | JSON data | 5.2 kB |
| 212 | `./propverifier/tasks/bounds_header_budget_02.json` | JSON data | 5.1 kB |
| 213 | `./propverifier/tasks/canonicalization_identity_fold_02.json` | JSON data | 4.4 kB |
| 214 | `./propverifier/tasks/canonicalization_signed_path_01.json` | JSON data | 5.1 kB |
| 215 | `./propverifier/tasks/containment_cidr_block_02.json` | JSON data | 6.0 kB |
| 216 | `./propverifier/tasks/containment_path_prefix_01.json` | JSON data | 5.0 kB |
| 217 | `./propverifier/tasks/cookie_host_prefix_01.json` | JSON data | 5.3 kB |
| 218 | `./propverifier/tasks/cookie_hostonly_forward_02.json` | JSON data | 5.2 kB |
| 219 | `./propverifier/tasks/state_drift_quorum_generation_02.json` | JSON data | 4.9 kB |
| 220 | `./propverifier/tasks/state_drift_revocation_epoch_01.json` | JSON data | 4.9 kB |
| 221 | `./requirements.txt` | text | 0.1 kB |
| 222 | `./results/B1.json` | JSON data | 36.9 kB |
| 223 | `./results/B1.md` | Markdown doc | 9.9 kB |
| 224 | `./results/B1_native.json` | JSON data | 65.4 kB |
| 225 | `./results/B1_native.md` | Markdown doc | 84.3 kB |
| 226 | `./results/B1_quarantine.json` | JSON data | 3.2 kB |
| 227 | `./results/REPORT.md` | Markdown doc | 9.7 kB |
| 228 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s01.diff` | other | 0.5 kB |
| 229 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s01.meta.json` | JSON data | 0.3 kB |
| 230 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s01.variant.txt` | text | 0.6 kB |
| 231 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s03.diff` | other | 0.7 kB |
| 232 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s03.meta.json` | JSON data | 0.3 kB |
| 233 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s03.variant.txt` | text | 0.6 kB |
| 234 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s05.diff` | other | 2.0 kB |
| 235 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s05.meta.json` | JSON data | 0.3 kB |
| 236 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s05.variant.txt` | text | 0.6 kB |
| 237 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s06.diff` | other | 0.7 kB |
| 238 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s06.meta.json` | JSON data | 0.3 kB |
| 239 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s06.variant.txt` | text | 0.6 kB |
| 240 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s09.diff` | other | 2.0 kB |
| 241 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s09.meta.json` | JSON data | 0.3 kB |
| 242 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s09.variant.txt` | text | 0.6 kB |
| 243 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s10.diff` | other | 0.8 kB |
| 244 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s10.meta.json` | JSON data | 0.3 kB |
| 245 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s10.variant.txt` | text | 0.6 kB |
| 246 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s12.diff` | other | 2.2 kB |
| 247 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s12.meta.json` | JSON data | 0.3 kB |
| 248 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s12.variant.txt` | text | 0.6 kB |
| 249 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s16.diff` | other | 0.5 kB |
| 250 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s16.meta.json` | JSON data | 0.3 kB |
| 251 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s16.variant.txt` | text | 0.6 kB |
| 252 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s19.diff` | other | 0.9 kB |
| 253 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s19.meta.json` | JSON data | 0.3 kB |
| 254 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s19.variant.txt` | text | 0.6 kB |
| 255 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s23.diff` | other | 0.4 kB |
| 256 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s23.meta.json` | JSON data | 0.3 kB |
| 257 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s23.variant.txt` | text | 0.6 kB |
| 258 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s24.diff` | other | 1.0 kB |
| 259 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s24.meta.json` | JSON data | 0.3 kB |
| 260 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s24.variant.txt` | text | 0.6 kB |
| 261 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s25.diff` | other | 1.6 kB |
| 262 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s25.meta.json` | JSON data | 0.3 kB |
| 263 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s25.variant.txt` | text | 0.6 kB |
| 264 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s27.diff` | other | 1.2 kB |
| 265 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s27.meta.json` | JSON data | 0.3 kB |
| 266 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s27.variant.txt` | text | 0.6 kB |
| 267 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s29.diff` | other | 1.5 kB |
| 268 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s29.meta.json` | JSON data | 0.3 kB |
| 269 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__deepseek-v4.1-flash-1017__s29.variant.txt` | text | 0.6 kB |
| 270 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s00.diff` | other | 0.6 kB |
| 271 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s00.meta.json` | JSON data | 0.3 kB |
| 272 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s00.variant.txt` | text | 0.6 kB |
| 273 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s01.diff` | other | 0.7 kB |
| 274 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s01.meta.json` | JSON data | 0.3 kB |
| 275 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s01.variant.txt` | text | 0.6 kB |
| 276 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s02.diff` | other | 0.6 kB |
| 277 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s02.meta.json` | JSON data | 0.3 kB |
| 278 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s02.variant.txt` | text | 0.6 kB |
| 279 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s03.diff` | other | 0.7 kB |
| 280 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s03.meta.json` | JSON data | 0.3 kB |
| 281 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s03.variant.txt` | text | 0.6 kB |
| 282 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s04.diff` | other | 0.6 kB |
| 283 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s04.meta.json` | JSON data | 0.3 kB |
| 284 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s04.variant.txt` | text | 0.6 kB |
| 285 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s06.diff` | other | 0.9 kB |
| 286 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s06.meta.json` | JSON data | 0.3 kB |
| 287 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s06.variant.txt` | text | 0.6 kB |
| 288 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s07.diff` | other | 0.6 kB |
| 289 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s07.meta.json` | JSON data | 0.3 kB |
| 290 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s07.variant.txt` | text | 0.6 kB |
| 291 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s14.diff` | other | 0.7 kB |
| 292 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s14.meta.json` | JSON data | 0.3 kB |
| 293 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s14.variant.txt` | text | 0.6 kB |
| 294 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s17.diff` | other | 0.9 kB |
| 295 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s17.meta.json` | JSON data | 0.3 kB |
| 296 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s17.variant.txt` | text | 0.6 kB |
| 297 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s19.diff` | other | 0.6 kB |
| 298 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s19.meta.json` | JSON data | 0.3 kB |
| 299 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s19.variant.txt` | text | 0.6 kB |
| 300 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s20.diff` | other | 0.9 kB |
| 301 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s20.meta.json` | JSON data | 0.3 kB |
| 302 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s20.variant.txt` | text | 0.6 kB |
| 303 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s21.diff` | other | 1.2 kB |
| 304 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s21.meta.json` | JSON data | 0.3 kB |
| 305 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s21.variant.txt` | text | 0.6 kB |
| 306 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s24.diff` | other | 0.6 kB |
| 307 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s24.meta.json` | JSON data | 0.3 kB |
| 308 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s24.variant.txt` | text | 0.6 kB |
| 309 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s25.diff` | other | 0.7 kB |
| 310 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s25.meta.json` | JSON data | 0.3 kB |
| 311 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s25.variant.txt` | text | 0.6 kB |
| 312 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s27.diff` | other | 3.0 kB |
| 313 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s27.meta.json` | JSON data | 0.3 kB |
| 314 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s27.variant.txt` | text | 0.6 kB |
| 315 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s28.diff` | other | 0.6 kB |
| 316 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s28.meta.json` | JSON data | 0.3 kB |
| 317 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s28.variant.txt` | text | 0.6 kB |
| 318 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s29.diff` | other | 0.8 kB |
| 319 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s29.meta.json` | JSON data | 0.3 kB |
| 320 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s29.variant.txt` | text | 0.6 kB |
| 321 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s30.diff` | other | 1.4 kB |
| 322 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s30.meta.json` | JSON data | 0.3 kB |
| 323 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s30.variant.txt` | text | 0.6 kB |
| 324 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s31.diff` | other | 0.6 kB |
| 325 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s31.meta.json` | JSON data | 0.3 kB |
| 326 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__glm-5.3-flash__s31.variant.txt` | text | 0.6 kB |
| 327 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s02.diff` | other | 0.4 kB |
| 328 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s02.meta.json` | JSON data | 0.3 kB |
| 329 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s02.variant.txt` | text | 0.6 kB |
| 330 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s22.diff` | other | 0.5 kB |
| 331 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s22.meta.json` | JSON data | 0.3 kB |
| 332 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s22.variant.txt` | text | 0.6 kB |
| 333 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s28.diff` | other | 0.7 kB |
| 334 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s28.meta.json` | JSON data | 0.3 kB |
| 335 | `./results/adjudication/authlib/authlib-jwt-dupkey-canonicalization__hy4-flash__s28.variant.txt` | text | 0.6 kB |
| 336 | `./results/claims.json` | JSON data | 4.2 kB |
| 337 | `./scripts/export_packs.py` | Python code | 11.0 kB |
| 338 | `./scripts/fetch.sh` | shell script | 2.6 kB |
| 339 | `./scripts/run_a2_tests.py` | Python code | 7.7 kB |
| 340 | `./tests/__init__.py` | Python code | 0.0 kB |
| 341 | `./tests/test_graders_baseline.py` | Python code | 16.0 kB |
| 342 | `./verify.py` | Python code | 10.0 kB |
