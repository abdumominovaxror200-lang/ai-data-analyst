# Dependency Graph — Wave Sequencing

Status: PROPOSED, derived from the Wave 0 audit. Confirms the user's proposed wave
structure against what actually exists in the codebase.

## Hard dependencies (must be sequential)

```
DATA-ARCHITECT (Dataset/Schema/DataSource contracts)
       │
       ├──▶ SQL-ENGINEER            (needs a DataSource to query against)
       ├──▶ LARGE-DATA-ENGINEER     (needs Dataset/Schema to chunk/stream against)
       └──▶ DATA-VALIDATION-ENGINEER (needs stable contracts to build ground-truth fixtures against)
```

```
TOOLING-ENGINEER (tool_router.py contract: schema shape, registration pattern)
       │
       ▼
 every new tool (SQL, stats, forecasting, clustering) registers through this contract
       │
       ▼
AGENT-ARCHITECT (agent.py — the loop that actually calls them)
       │
       ▼
BENCHMARK-ENGINEER (can only test capabilities once they're wired into the live loop)
```

This second chain is the one the audit flags as highest-risk: `agent.py` and
`tool_router.py` already exist and work (validated this session against real Groq
traffic). Every new capability team depends on TOOLING-ENGINEER's registration contract
staying stable — a breaking change there risks regressing the working agent loop and
its benchmark results. Recommend TOOLING-ENGINEER publish the tool-schema contract
(already implicitly defined by the 10 existing tools in `tool_router.py`) as an explicit
doc before Wave 2 teams start registering new tools against it.

## Can run in parallel (no shared files, no data dependency)

Wave 1 candidates, confirmed independent by the audit (no file overlap):

```
DATA-ARCHITECT ─┐
SQL-ENGINEER    │  (SQL-ENGINEER can build against a *stub* contract in parallel,
LARGE-DATA      │   integrate once DATA-ARCHITECT's real contract lands — reduces
STATISTICS      │   the hard-sequential cost above to an integration step, not a
QA              │   full blocking wait)
SECURITY        ─┘
```

Wave 2 candidates (all depend on Wave 1's DATA-ARCHITECT contracts + STATISTICS
foundation, but not on each other):

```
EDA-ANALYST ─┐
FORECASTING  │
ADVANCED-ANALYTICS │  independent of each other; each needs DATA-ARCHITECT's
VISUALIZATION │  contracts (Wave 1) to exist first
BUSINESS-ANALYTICS ─┘
```

Wave 3 candidates (all touch or depend on the agent/tooling core):

```
AGENT-ARCHITECT   ← sole writer of agent.py, must integrate Wave 1+2 tools
TOOLING           ← sole writer of tool_router.py, must register Wave 1+2 tools
REASONING/VALIDATION ← builds on AGENT-ARCHITECT's existing constraint-checking prompt
CONTEXT/TOKEN     ← builds on AGENT-ARCHITECT's existing dedup/stagnation-stop logic
```

Note: unlike a from-scratch build, Wave 3's foundation is **not empty** — AGENT-ARCHITECT
is extending working code, not writing a new loop. REASONING/VALIDATION and CONTEXT/TOKEN
should treat the existing `agent.py` behavior (constraint prompt, coverage warnings,
dedup) as a baseline to preserve, not replace — regression against the 7-question
benchmark from this session is the concrete acceptance bar.

Wave 4 (production hardening) — independent of each other, depend on Waves 1–3 having
something to harden:

```
PERFORMANCE ─┐
SECURITY     │  independent; PERFORMANCE specifically needs LARGE-DATA's chunking
DEVOPS       │  work (Wave 1) to have something to benchmark at 1M+/10M+ rows
REPORTING    │
DOCUMENTATION ─┘
```

## Gap the audit found that the proposed graph doesn't cover

**Frontend is not assigned to any team.** New SQL results, forecasts, and cluster
outputs will all need a frontend surface (new tabs/components), but the proposed org
chart has no FRONTEND-ENGINEER role. This should be resolved before Wave 2 —
recommend either (a) adding a FRONTEND-ENGINEER role, or (b) splitting frontend work
by capability (VISUALIZATION-ENGINEER owns chart-rendering components,
REPORTING-ENGINEER owns report/export UI, BUSINESS-ANALYST's new tools get a
generic result-renderer maintained by whichever team ships first). Decision needed —
logged in `decisions.md`.
