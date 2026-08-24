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

Wave 3 — **superseded by the detailed plan in `reasoning-layer-design.md` §11–12.**
Summary (see that doc for the full graph):

```
Phase 3A: TOOLING INTEGRATION — ✅ COMPLETE (see completed_tasks.md).
    22 previously-unregistered Wave 1+2 tools now live in tool_router.py
    (32 total). This prerequisite is CLEARED — the planner below now has
    the real, full tool catalog to plan against, not the original 10.
       │
       ▼
REASONING-ARCHITECT (contracts.py + orchestrator.py skeleton)   ← Phase 3a, sequential
       │
       ├──▶ QUESTION-PARSING-ENGINEER   (question_parser.py, premise_validator.py)
       ├──▶ PLANNING-ENGINEER           (planner.py)              Phase 3b,
       ├──▶ VERIFICATION-ENGINEER       (verifier.py, uncertainty.py)  parallel,
       ├──▶ SYNTHESIS-ENGINEER          (synthesizer.py)           no file overlap
       ├──▶ REASONING-QA-ENGINEER       (tests + reasoning benchmark)
       └──▶ REASONING-SECURITY-ENGINEER (review only)
       │
       ▼
Orchestrator integration (executor.py wiring, API route, full regression)  ← Phase 3c
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
