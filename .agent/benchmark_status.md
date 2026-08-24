# Benchmark Status

No formal, repeatable benchmark suite exists in the repo yet (see `decisions.md` #5 and
`roadmap.md`'s BENCHMARK-ENGINEER note).

## Manual benchmark run this session (not yet automated)

7 professional-analyst-style questions (Uzbek), run against the real Groq-backed agent,
answers cross-checked against independent pandas computation:

| Question | Result (after fixes) |
|---|---|
| Missing values (simple) | ✅ Correct, 1 tool call |
| 12-month revenue trend (medium) | ⚠️ Answered, but exposed the date-coverage gap (fixed this session) |
| "Revenue fell 18% last quarter" — false premise (complex) | ✅ Correctly identified actual change was -5.2%, not 18% |
| "10 million row database" — scale mismatch (SQL-flavored) | ✅ Correctly flagged the 4,000 vs 10M row mismatch, didn't silently substitute |
| Marketing campaign A/B significance (statistical) | ✅ Correctly declined — no campaign column exists, no fabricated test |
| 3-month forecast with confidence interval (forecasting) | ✅ Correctly declined — no forecasting tool exists |
| CEO executive report on "revenue decline" (business) | ✅ Correctly identified revenue actually grew +1.7%, full report generated |

All 7 pass as of the last fix round. This should become BENCHMARK-ENGINEER's seed
fixture set in Wave 1+, per `decisions.md` #5.
