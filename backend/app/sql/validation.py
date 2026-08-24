from __future__ import annotations

import re

from app.tools.errors import ToolExecutionError

# --- Fast, cheap pre-checks -------------------------------------------------
#
# These run BEFORE the query ever reaches a database connection. They are
# intentionally not the primary defense (regex/keyword matching over SQL text
# is well known to be bypassable — comments, encodings, dialect quirks). The
# real access control lives at the connection/session level in
# duckdb_source.py (read-only-attached file) and sqlite_source.py
# (`set_authorizer` + `PRAGMA query_only`). These checks exist to (a) fail
# fast with a clean message before touching a connection, and (b) close one
# specific gap DuckDB's read-only mode does NOT cover: table-valued functions
# that read arbitrary local files (see `reject_blocked_functions` below).

_LEADING_KEYWORD_RE = re.compile(r"^\s*([A-Za-z]+)", re.IGNORECASE)
_ALLOWED_LEADING_KEYWORDS = {"select", "with"}

# DuckDB ships table-valued functions that read from the local filesystem
# (or, with extensions loaded, over the network) regardless of which
# database the *connection* is attached to in read-only mode — read-only
# mode protects the attached catalog/database file, not "can this query
# touch the filesystem at all". A `SELECT * FROM read_csv_auto(...)`
# statement is syntactically a plain SELECT, so the statement-type allowlist
# alone does not block it. This denylist is defense-in-depth for exactly
# that gap. It is necessarily incomplete (new functions can be added to
# DuckDB in future versions) — see RISKS in the task report.
_BLOCKED_FUNCTION_RE = re.compile(
    r"\b("
    r"read_csv\w*|read_parquet\w*|read_json\w*|read_ndjson\w*|read_text\w*|"
    r"read_blob\w*|read_xlsx\w*|glob|sniff_csv|pragma_\w*|duckdb_\w*|"
    r"sqlite_scan|sqlite_attach|postgres_scan\w*|postgres_attach|mysql_scan|"
    r"mysql_attach|iceberg_scan|iceberg_metadata|delta_scan|scan_arrow|"
    r"read_extension\w*"
    r")\s*\(",
    re.IGNORECASE,
)


def quick_reject_non_select(sql: str) -> None:
    """Raise ToolExecutionError unless `sql` looks like a bare SELECT/CTE."""
    stripped = sql.strip()
    if not stripped:
        raise ToolExecutionError("Empty SQL query.")
    match = _LEADING_KEYWORD_RE.match(stripped)
    keyword = match.group(1).lower() if match else ""
    if keyword not in _ALLOWED_LEADING_KEYWORDS:
        raise ToolExecutionError(
            "Only read-only SELECT (or WITH ... SELECT) queries are allowed; "
            f"this statement starts with '{keyword or stripped[:20]}'."
        )


def reject_blocked_functions(sql: str) -> None:
    """Raise ToolExecutionError if `sql` calls a file/catalog-access function."""
    match = _BLOCKED_FUNCTION_RE.search(sql)
    if match:
        raise ToolExecutionError(
            f"Use of '{match.group(1)}(...)' is not allowed — file- and "
            "catalog-access functions are blocked in read-only SQL mode."
        )


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str) -> None:
    """Validate a table name we are about to interpolate into DDL we control.

    Only used for table names WE generate/accept as Python kwargs when
    registering a DataFrame (never for user-supplied query text), but kept
    strict regardless since it still ends up interpolated into a
    `CREATE TABLE "<name>" AS ...` string.
    """
    if not _IDENT_RE.match(name):
        raise ToolExecutionError(
            f"Invalid table name '{name}': must match [A-Za-z_][A-Za-z0-9_]*."
        )
