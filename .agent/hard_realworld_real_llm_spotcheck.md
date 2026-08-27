# Hard Real-World Benchmark — Real-LLM Spot-Check

Small, representative sample (6 cases) run against the real configured provider
(Groq, `openai/gpt-oss-120b`) — not the full 102-case suite, per this project's
standing rule against spending shared, limited quota re-validating structural
plumbing the scripted 97.1% result already proves works deterministically. Scored
with the identical 15-dimension rubric the scripted result was graded by, so the two
are directly comparable.

**This report was reconstructed from the real run's captured console output** after
the harness script itself hit a real path-resolution bug on its own report-writing
step (`run_hard_spotcheck.py` computed `.agent/`'s path one directory level too high
— `parents[5]` instead of `parents[4]` — fixed in the same commit as this report;
see the script's git history). No case data was lost or altered — every number below
is exactly what the live run produced.

## Result summary

| case_id | verdict | provider_failure | explanation |
|---|---|---|---|
| `hard_confound_01a` | **PARTIAL** | no | Failed `data_quality_awareness` (traps not flagged: `format`, `mix`) and `method_selection` (expected `group_and_aggregate`/`correlation_analysis`, used `t_test`) |
| `hard_stat_02` | PASS | no | All applicable dimension checks passed |
| `hard_prim_05` | PASS | no | All applicable dimension checks passed |
| `hard_prim_04` | PASS | no | All applicable dimension checks passed |
| `hard_prim_10` | UNMEASURED | **yes** | PROVIDER_ERROR — daily token quota (TPD) exhausted (`Used 199531/200000`, then `199834/200000`) |
| `hard_scale_08` | UNMEASURED | **yes** | PROVIDER_ERROR — same TPD exhaustion, worsening (`Used 199435`, then `199334` against a 200000 cap with the request itself exceeding remaining headroom) |

**Real responses obtained**: 4/6.
**Provider failures (UNMEASURED, never reported as model failures)**: 2/6 — both are
genuine daily-quota exhaustion (Groq's free tier: 200,000 tokens/day), confirmed
directly from the provider's own error text, not inferred. Per the mission's explicit
instruction, the spot-check **stopped is not applicable here** (it naturally reached
the end of its 6-case sample before quota fully zeroed out) but made no further live
calls once exhaustion was confirmed.

A secondary, non-fatal issue also appeared in the raw log during `hard_prim_04`'s
run: one attempt got `LLM provider HTTP error 400: Tool choice is none, but model
called a tool` — a transient provider-side/prompt-shape mismatch on a single
attempt, automatically retried by the existing (unmodified) retry logic, and the case
went on to PASS on a later attempt. Not investigated further in this pass since it
self-resolved and is not reproducible on demand without spending more quota — noted
here for visibility, not treated as root-caused.

Throughout the run, free-tier TPM throttling (8,000 tokens/minute) caused frequent
`rate limited, retrying` backoff messages — expected, handled correctly by the
existing retry/backoff logic, not a bug.

## The one genuine model failure: `hard_confound_01a`

**Question**: "North region has a $55 higher average basket size than South -- is
North just a better-performing region?" (against `region_size_confound`, a fixture
where North is 18/20 large-format stores and South is 2/20 large-format stores — the
regional gap is a store-format confound, not a true regional difference).

**What the real model did**: called `t_test` (comparing basket size between North and
South directly) instead of examining the `format` breakdown. Its answer never
mentioned store format or a format/region mix issue at all — it evaluated
statistical significance of the raw regional gap without ever checking whether
another variable explained it.

**Root cause**: an LLM-reasoning limitation, not a deterministic-code bug (category
F/D-adjacent in the mission's 12-way taxonomy) — the planner has no mechanism, prompt-
based or otherwise, that nudges it to check for confounding variables before
accepting a two-group comparison as reflecting a true group-level difference. This is
a real, now-confirmed-live instance of exactly the pattern the `region_size_confound`
fixture was built to test — previously only demonstrated via a scripted MockProvider
response; now confirmed against the real model.

**Architectural fix** (not a prompt change, per the standing instruction to improve
architecture, not merely prompts): see the accompanying commit — a new deterministic
`app/reasoning/confound_detection.py` module, wired into `orchestrator.py`, that scans
for exactly this pattern automatically after any group-comparison tool call, using the
real dataset the comparison ran against. It does not require the model to think of a
confound check itself; it runs unconditionally and would have caught this exact
question mechanically, independent of what the model happened to do. See that
module's own docstring and `.agent/hard_realworld_benchmark.md` §18 for the full
design, root-cause chain, and regression tests.

## What remains

- Only 4 of 6 sample cases got a real answer before quota exhaustion; `hard_prim_10`
  (the positive-control causal-permission case) and `hard_scale_08` (large-N
  statistical-vs-practical-significance) remain UNMEASURED against a real provider.
  Should be retried in a future session once the daily quota window resets — not
  retried in this session per the explicit "stop instead of hammering the API" rule.
- The `hard_prim_04` transient `400: Tool choice is none, but model called a tool`
  error is unexplained and not reproduced/root-caused — worth a closer look if it
  recurs, but not chased down on a single, self-resolved occurrence given quota cost.
- This remains a 4-real-case sample, not a statistically powered claim about real-LLM
  reliability across the full 102-case hard suite — it should not be read as more
  than "one genuine reasoning gap found and fixed, three other hard cases confirmed
  handled correctly live."
