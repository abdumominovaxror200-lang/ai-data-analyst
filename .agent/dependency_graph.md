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

Wave 3 — Phase 3A and 3B both complete, both built as a single direct orchestrator
pass rather than the originally-proposed parallel-agent waves (see `decisions.md`):

```
Phase 3A: TOOLING INTEGRATION — ✅ COMPLETE
    22 previously-unregistered Wave 1+2 tools now live in tool_router.py (32 total).
       │
       ▼
Phase 3B: REASONING ARCHITECTURE FOUNDATION — ✅ COMPLETE
    backend/app/reasoning/{contracts,categories,causation_guard,premise_validator,
    question_parser,planner,executor,verifier,synthesizer,orchestrator}.py.
    61 new tests, 629/629 full suite. 100% additive -- agent.py/tool_router.py
    untouched. See completed_tasks.md for full detail.
       │
       ▼
Phase 3C (NOT STARTED, stopped here per explicit instruction):
    - Wire ReasoningOrchestrator into an API route (or a mode flag on the
      existing chat route -- still an open decision, see decisions.md #1a).
    - Score the 12-case reasoning_questions.json benchmark for real (no score
      claimed yet); expand toward 100-150 cases.
    - Still-pending Wave 2 agents (BUSINESS-ANALYTICS, VISUALIZATION,
      DATA-QUALITY, QA-PROFESSIONAL-BENCHMARK) remain independently launchable
      whenever resumed -- not blocked by Phase 3B, no file overlap.
```

Note: unlike a from-scratch build, this foundation is **not empty** — `agent.py`'s
existing constraint-checking prompt, dedup/stagnation-stop, and trust-boundary wrapping
are all reused as-is by the reasoning layer's `executor.py`, not rebuilt. Regression
against the existing 505-test suite plus the new reasoning-quality benchmark
(`reasoning-layer-design.md` §10) is the acceptance bar. **Not yet launched** — waiting
for approval per the user's explicit "design first, stop before implementing" instruction.

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
