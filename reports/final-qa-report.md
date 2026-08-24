# Final QA Report — AI Data Analyst

Date: 2026-08-24

## 1. Backend test result

```
cd backend && venv\Scripts\python -m pytest -v
```

**70 / 70 passed** (0.9–1.1s runtime). Breakdown: 4 agent tool-calling / no-hallucination
tests, 6 aggregation, 6 anomaly detection, 7 API validation, 6 charts, 4 correlation/comparison,
9 filtering, 3 JSON-serialization regression tests, 5 malicious-file/security, 3 profiler, 5
provider (rate-limit parsing), 4 statistics, 8 upload/validation.

Full list re-verified in this pass — see terminal output archived in this session; every test
name and PASSED status confirmed individually, not just the summary count.

## 2. Frontend test result

- `npx tsc -b --noEmit` → clean, 0 errors.
- `npm run build` → clean production build. Output: `index.html` 0.77 kB, CSS 15.15 kB (gzip
  4.12 kB), JS 833.53 kB (gzip 247.54 kB). Vite warns the JS chunk exceeds 500 kB (driven mostly
  by Recharts) — noted as a future code-splitting opportunity, not a functional issue.
- No ESLint config exists in this Vite scaffold; TypeScript's own compiler is the enforced
  static check.

## 3. End-to-end result

Full user journey driven directly in a real browser (Chromium via the Browser pane), against
the live backend and the real Groq LLM — nothing mocked.

| Step | Result |
|---|---|
| 1. Open application | ✅ Landing page renders, no console errors |
| 2. Upload `sales_data.xlsx` | ✅ 4,000 rows / 11 columns profiled correctly |
| 3. Dataset overview loads | ✅ Column table, roles, missing/duplicate counts all correct |
| 4. Navigate to AI Analyst | ✅ |
| 5. Enter natural-language question | ✅ Exact test question submitted via real form input |
| 6. Real Groq LLM is called | ✅ Confirmed via network tab: `POST /api/chat` → live Groq API calls in server logs |
| 7. Agent calls analytical tools | ✅ `group_and_aggregate` (top-5 products) + `detect_anomalies` (profit, IQR) |
| 8. Results returned | ✅ 200 OK, full structured response |
| 9. Charts/insights/anomalies display | ✅ Verified separately: bar chart, insights stats, anomaly table all render correctly |
| 10. Report section works | ✅ Generated + key findings + anomalies + correlations rendered |
| 11. Invalid input → clean error | ✅ `malware.exe` upload → clean banner: "Unsupported file type '.exe'. Allowed: .csv, .xlsx" — no crash, no stack trace |
| 12. Refresh does not break the app | ✅ Reload returns cleanly to the upload screen (dataset state is in-memory only by design — see Limitations) |

**AI response accuracy (step 6–8) — independently verified, zero deviation:**

| Metric | LLM's answer | Direct pandas computation |
|---|---|---|
| Zenith Office Chair revenue | $347,166.79 | $347,166.79 ✓ |
| Halo Desk Monitor revenue | $339,384.36 | $339,384.36 ✓ |
| Torque Dumbbell Set revenue | $203,281.51 | $203,281.51 ✓ |
| Anomaly count (profit, IQR) | 298 (7.45%) | 298 (7.45%) ✓ |
| IQR bounds | -$178.97 to $455.40 | -178.9687 to 455.4012 ✓ |

The agent also correctly cited the seeded negative-profit anomaly (Verve Running Shoes,
2024-03-12, -$334.75) — the exact synthetic data-quality issue planted in the demo dataset.

Markdown formatting (tables, bold, headers) in the AI's response renders as real HTML — verified
3 `<table>` elements present in the DOM, not raw pipe-delimited text.

## 4. Security result

| Check | Result |
|---|---|
| `.env` ignored by git | ✅ `git check-ignore` confirms `backend/.env`, `venv/`, `node_modules/`, `storage/` all excluded |
| API key never exposed to frontend | ✅ Frontend never receives `LLM_API_KEY`; confirmed via inspecting all `/api/chat`, `/api/analysis` response bodies |
| API key never logged | ✅ Grepped `backend/app/` — the key only appears in the `Authorization` header sent to the LLM provider, never in a `logger.*` call |
| Filenames sanitized | ✅ `sanitize_display_name` strips path components and unsafe characters (tested, incl. null-byte injection) |
| Path traversal blocked | ✅ Storage path is built from a server-generated UUID only, never user input (tested) |
| Invalid file types rejected | ✅ `.exe` etc. → clean 400 (tested + browser-verified) |
| File size limits enforced | ✅ Tested via `MAX_UPLOAD_MB` override |
| Arbitrary uploaded code cannot execute | ✅ Only `pandas.read_csv`/`read_excel(engine="openpyxl")`; no `pickle`, no macros, no `eval` |
| No secrets committed | ✅ Repo has zero commits so far (nothing to leak); broad regex scan (`gsk_`, `sk-`, `AIza`, Slack tokens) across all git-trackable files found nothing |

**Secret scan command used:**
```
rg 'gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}' ai-data-analyst/
```
No matches outside the gitignored `backend/.env`.

## 5. Performance observations

| Operation | Time | Notes |
|---|---|---|
| Upload + parse + profile (server-side, isolated) | ~690 ms | Dominated by `openpyxl`'s pure-Python XML parsing of the 252 KB / 4,000-row file |
| `profile_dataset` tool | 8.7 ms | |
| `detect_anomalies` tool | 6.5 ms | |
| `generate_chart` tool | 1.0 ms | |
| Full AI chat request (2 tool calls + narration, real Groq) | 19.4 s | Includes real network latency to Groq + model generation time |

**Investigation note:** an initial measurement via Python's `requests` library over
`http://localhost` showed ~2.5 s uploads — traced to a known Windows client-side artifact
(`localhost` resolving to `::1` first, stalling before IPv4 fallback). Re-measuring against
`127.0.0.1` directly confirmed real server-side time is ~600–700 ms, matching the isolated
measurement above. This was a test-tooling artifact, not an application performance issue — no
code change was needed.

**Free-tier LLM rate limit (real finding, now mitigated):** Groq's free tier caps at 8,000
tokens/minute per account. The exact multi-part test question needs 2–5 sequential tool-calling
round trips; early in this QA pass that combination hit 429s reliably. Fixed with (a) automatic
retry-with-backoff honoring the provider's own reset-time headers, (b) `LLM_REASONING_EFFORT=low`
(cut reasoning-token overhead from 56 to 5 tokens per call in a like-for-like comparison), and
(c) errors that still exhaust retries now surface as a clean 503 ("LLM provider is
rate-limiting requests. Please try again shortly.") instead of an unhandled crash. After both
fixes, the exact test question completed successfully in 19.4 s.

No unnecessary LLM calls were introduced — the direct-tool tabs (Charts, Anomalies, Report,
Insights) call `/api/analysis` or `/api/reports` directly and never touch the LLM.

## 6. Bugs found and fixed this session

1. **`.env` not found at runtime.** `pydantic-settings` resolved `env_file=".env"` relative to
   the process's *current working directory*, not the config file's location. The dev-server
   runner launched uvicorn from a different directory, so `backend/.env` (and therefore
   `LLM_API_KEY`) was silently never read. As a side effect, uploaded files were also being
   written into the wrong (unrelated, parent) project's directory. **Fixed:** both `env_file`
   and the default `storage_dir` are now anchored to an absolute path derived from
   `config.py`'s own location. Verified with a reproduction test before and after the fix, and
   the leaked directory was cleaned up.
2. **JSON serialization crash on datetime columns.** `filter_data`, `detect_anomalies`, and
   scatter `generate_chart` returned raw `pandas.Timestamp` objects in row previews whenever the
   dataset had a datetime column (true for the demo dataset's `date` column) — `json.dumps`
   can't serialize `Timestamp`, crashing the whole chat turn with an unhandled 500. **Fixed:**
   added `app/tools/serialization.py::dataframe_to_records` (converts datetimes to ISO strings
   before `.to_dict()`) used by all three call sites, plus `default=str` as defense-in-depth in
   the agent's own `json.dumps` call. 3 regression tests added.
3. **Rate-limit errors crashed with a raw 500.** A `429` from the LLM provider propagated as an
   unhandled `httpx.HTTPStatusError`, producing a generic "Internal Server Error" with no useful
   message. **Fixed:** `LLMProviderError` is now raised with a clean, user-facing message and
   caught in `routes_chat.py`, returning a proper `503`. Combined with retry/backoff, this also
   made the real end-to-end test pass reliably.
4. **Frontend showed raw markdown as plain text.** The AI's answers include tables, bold text,
   and headers (confirmed via the live Groq test), which rendered as literal `**`/`|---|` text
   in the chat bubble. **Fixed:** added `react-markdown` + `remark-gfm` with dark-theme styling;
   verified 3 real `<table>` elements render in the DOM for the test question's response.

## 7. UX polish applied

- Markdown-rendered AI responses (tables, bold, headers) instead of raw text.
- Human-readable tool badges ("Aggregation", "Anomaly detection") instead of raw function names.
- Escalating loading message: "Analyzing your data…" → "Still analyzing your data… this can
  take a few extra seconds." after 5 s, so a retry-driven delay doesn't look like a hang.
- Chat input and Send button both disable while a request is in flight.
- Chart tab: explicit loading skeleton, empty state before first generation, error banner.
- Friendlier, non-technical error messages throughout (network failures, unexpected errors)
  without exposing stack traces, internal URLs, or implementation details.

## 8. Remaining limitations

See [`README.md`](../README.md#limitations) for the full list. Summary: no persistence across
backend restarts (in-memory store, MVP by design), no cross-dataset joins, free-tier LLM rate
limits can still add latency (though no longer crash), no streaming responses, no auth.

## 9. Final status

**FINAL STATUS: READY**

The MVP meets every item in the Definition of Done: upload (CSV/XLSX), automatic profiling,
natural-language Q&A backed by a real, verified LLM provider, deterministic Python-only
calculations, charts, anomaly detection, business insights, recommendations, report export,
graceful handling of invalid input, a passing automated test suite, no committed secrets, and
complete documentation.

1. **Start backend:**
   ```bash
   cd backend
   venv\Scripts\python -m uvicorn app.main:app --reload --port 8010
   ```
2. **Start frontend:**
   ```bash
   cd frontend
   npm run dev
   ```
3. **Demo URL:** `http://localhost:5173` (backend API at `http://localhost:8010/api`, docs at
   `http://localhost:8010/docs`)
4. **Test counts:** 70/70 backend tests passing · frontend type-check and production build clean
5. **Remaining issues:** none blocking. Only the noted limitations above (persistence, joins,
   free-tier rate limits, streaming, auth) — all explicitly out of scope for this MVP and listed
   as future roadmap items in the README.
