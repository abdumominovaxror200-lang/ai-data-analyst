# Tool Inventory — Phase 3A (Tooling Integration)

Every tool now registered in `backend/app/agent/tool_router.py` (`TOOL_SCHEMAS` +
`_HANDLERS`), as of this pass. 10 pre-existing + 22 newly wired (Wave 1 + Wave 2
capabilities that existed on disk but were not LLM-callable) = **32 total**.

All tools share these properties unless noted otherwise: **deterministic** (pure
pandas/numpy/scipy/statsmodels/sklearn computation, no LLM involvement — the
"LLM never invents numbers" rule), **cannot modify data** (every tool receives
`record.df` read-only; none writes back to the `DatasetRecord` or any file), and
every result returned through `ToolRouter.execute` is wrapped with the
`[UNTRUSTED DATA]` marker (`agent.py`'s `_wrap_tool_payload`) before it ever
reaches the LLM — verified for every tool class in
`backend/tests/test_tool_registration.py`.

## Pre-existing (already registered before this pass)

| Tool | Module | Input contract | Output contract | Failure modes |
|---|---|---|---|---|
| `profile_dataset` | `profiler.py` | none | shape, column_info, date_ranges, text/bool/numeric/categorical/date column buckets | none (never raises on a loaded dataset) |
| `describe_data` | `statistics.py` | `columns?`, `filters?` | per-column mean/median/std/min/max/top-values | unknown column, empty filter result |
| `filter_data` | `filtering.py` | `filters` (required) | match count + bounded preview | unknown column, bad operator, type mismatch |
| `group_and_aggregate` | `aggregation.py` | `group_by`, `agg_column`, `agg_func?`, `filters?`, `top_n?` | grouped/aggregated rows | unknown column, non-numeric agg target |
| `compare_periods` | `comparison.py` | 2 date ranges + `value_column` | delta/pct-change + coverage warnings | unparseable dates, no overlap with data |
| `correlation_analysis` | `correlation.py` | `columns?`, `method?`, `filters?` | top correlated pairs (bounded) | <2 numeric columns |
| `detect_anomalies` | `anomaly.py` | `column`, `method?`, `threshold?` | outlier rows/bounds (IQR or z-score) | non-numeric column, all-null column |
| `generate_chart` | `charts.py` | `chart_type`, `x`, `y?`, ... | chart-ready series data | unsupported chart_type, unknown column |
| `generate_business_insights` | `insights.py` | `filters?` | bundled stats/anomalies/correlations summary | empty dataset after filters |
| `generate_report` | `report.py` | none | full structured report | none |

## Newly registered this pass

### Statistical hypothesis testing (`app/tools/hypothesis.py`, Wave 1, `scipy.stats`)

| Tool | Input contract | Output contract | Min data | Failure modes |
|---|---|---|---|---|
| `t_test` | `column`, `group_column`+`group_a`+`group_b` OR `popmean`, `alpha?` | statistic, p_value, dof, significant | 2 samples/group | missing group columns/popmean, <2 samples/group, unknown column |
| `chi_square_test` | `column_a`, `column_b`, `alpha?` | statistic, p_value, dof, contingency_table | 2×2 categories | <2 distinct categories in either column |
| `anova_test` | `value_column`, `group_column`, `alpha?` | statistic, p_value, per-group stats | 2 samples × 2 groups | <2 groups, <2 samples in any group |
| `confidence_interval` | `column`, `confidence?` | mean, bounds, margin of error | 2 samples | <2 samples, confidence out of (0,1) |
| `effect_size` | `column`, `group_column`, `group_a`, `group_b` | Cohen's d + magnitude label | 2 samples/group | pooled std == 0, <2 samples/group |

Resource usage: O(n) per column, negligible even at large-data scale. No SQL/file access.

### Regression (`app/tools/regression.py`, `regression_diagnostics.py`; Wave 1 + Wave 2, `statsmodels`)

| Tool | Input contract | Output contract | Min data | Failure modes |
|---|---|---|---|---|
| `linear_regression` | `target_column`, `feature_columns`, `alpha?` | coefficients, p-values, R², summary | rows > features+1 | collinear features (singular matrix), non-numeric columns |
| `regression_diagnostics` | same as above | base regression + Shapiro-Wilk normality, Breusch-Pagan heteroscedasticity, VIF multicollinearity | ≥8 complete rows | same as above; Shapiro-Wilk skipped above 5000 rows (documented, not silent) |
| `outlier_analysis_multivariate` | `columns` (≥2), `method?`, `contamination?` | flagged rows (bounded to 50), scores, threshold | rows > columns | <2 columns, singular covariance matrix (auto-falls back to pseudo-inverse) |

Resource usage: OLS fit is O(n·k²); fine at demo scale (4K rows), untested at 10M-row
scale in this pass — large-data-scale regression is a known gap, not silently assumed safe.

### Forecasting (`app/tools/forecasting.py`, Wave 2, `statsmodels.tsa`)

| Tool | Input contract | Output contract | Min data | Failure modes |
|---|---|---|---|---|
| `train_test_split_timeseries` | `date_column`, `value_column`, `test_size?` | train/test row counts + date ranges | 10 points | too few points for the split |
| `decompose_timeseries` | `date_column`, `value_column`, `period?`, `model?` | trend/seasonal/residual summaries + one seasonal cycle | 2×period points | unparseable dates, non-positive values with `model="multiplicative"`, un-inferrable period |
| `forecast` | `date_column`, `value_column`, `periods`, `method?`, `confidence_level?` | point forecast + prediction interval per period | 10 points | `periods` > observed history (hard refusal, not extrapolation), duplicate timestamps |
| `backtest_forecast` | `date_column`, `value_column`, `method?`, `horizon?` | MAE/RMSE/MAPE on held-out data | 10 points | same as above |

Resource usage: ARIMA grid search is a small fixed 6-candidate set (bounded fitting
cost); ETS is a single fit. Both are request-time-appropriate, not exhaustive
auto-ARIMA. Refuses (does not silently guess) below `_MIN_POINTS=10` or when
`periods` exceeds observed history.

### Clustering / dimensionality reduction (`app/tools/clustering.py`, Wave 2, `scikit-learn`)

| Tool | Input contract | Output contract | Min data | Failure modes |
|---|---|---|---|---|
| `kmeans_cluster` | `columns` (≥2), `n_clusters?` | per-cluster size/centroid, silhouette score | `n_clusters × 2` rows | zero-variance column, <2 numeric columns, duplicate column names |
| `pca_reduce` | `columns` (≥2), `n_components?` | per-component explained variance + loadings | `n_components` rows | `n_components` > min(columns, rows) |

Resource usage: K-means auto-k fits up to 7 models (k=2..8) via silhouette
search — bounded, not unbounded search. Standardization (`StandardScaler`) applied
before fitting in both tools.

### Customer segmentation (`app/tools/segmentation.py`, Wave 2, pandas only)

| Tool | Input contract | Output contract | Min data | Failure modes |
|---|---|---|---|---|
| `rfm_analysis` | `customer_column`, `date_column`, `value_column`, `reference_date?` | per-segment (10 standard RFM labels) summaries, never per-customer rows | 5 distinct customers | <5 customers, no rows on/before reference_date |
| `cohort_analysis` | `customer_column`, `date_column`, `value_column?`, `period?` | retention or value matrix, capped 24×24 | any | invalid `period` value |
| `churn_risk_analysis` | `customer_column`, `date_column`, `reference_date?`, `churn_threshold_days?` | active/at_risk/churned counts + inferred threshold | 2 customers with 2+ transactions (if threshold not given) | cannot infer threshold from <2 qualifying customers |

Resource usage: groupby-based, O(n) in row count; `cohort_analysis`'s output is
explicitly bounded regardless of dataset history length.

### Automated EDA (`app/tools/eda.py`, Wave 2, composes `profiler`/`anomaly`/`correlation` + `scipy.stats`)

| Tool | Input contract | Output contract | Failure modes |
|---|---|---|---|
| `automated_eda` | `filters?` | schema, missingness, duplicates, cardinality, distributions, outliers, relationships, prioritized `potential_problems` list (capped 25) | empty dataset after filters |
| `analyze_cardinality` | `filters?` | per-column classification (constant/near_constant/boolean_like/continuous_numeric/unique_id_like/high_cardinality/low_cardinality_categorical) | empty dataset after filters |
| `analyze_distributions` | `columns?`, `filters?` | skewness/kurtosis (numeric) or entropy/balance (categorical) per column | unknown column name |

Resource usage: composes existing tools rather than recomputing; per-column detail
capped at 60 columns, outlier scan capped at 12 columns in `automated_eda`.

### Read-only SQL (`app/sql/duckdb_source.py`, `sqlite_source.py`, Wave 1 — bridged into the
tool router this pass via new `_run_sql_query`/`_explain_sql_query` functions in
`tool_router.py` itself; the SQL engines were not modified)

| Tool | Input contract | Output contract | Data source | Failure modes |
|---|---|---|---|---|
| `run_sql_query` | `sql` (SELECT only), `engine?` (`duckdb` default or `sqlite`) | columns, rows (capped), row_count, truncated flag | constructs a fresh `DuckDBDataSource`/`SQLiteDataSource` over `record.df` per call, closed after | any write/DDL/PRAGMA/ATTACH/multi-statement attempt (`ToolExecutionError`, engine-level + parser-level block — unmodified from Wave 1), query timeout (30s default), memory ceiling (512MB default), unknown table/column |
| `explain_sql_query` | `sql`, `engine?` | query plan, no execution | same | same validation, no execution-time failure modes |

**Security constraints (inherited unmodified from the existing SQL layer — not
re-implemented or bypassed by this bridge):** read-only enforcement is engine-level
(DuckDB `read_only=True` connection + parser statement-type check + function
denylist; SQLite `set_authorizer` + `PRAGMA query_only` + one-statement-per-execute).
See `backend/docs/security/sql-layer-threat-model.md` for the full threat model —
unchanged by this integration.

**Resource usage — the one place this pass adds a new constraint, not just wires up
an existing one:** `_SQL_TOOL_MAX_ROWS = 500`, tighter than each DataSource's own
default of 10,000, because a tool result here is JSON-dumped directly into the LLM's
conversation context — this project has a documented history of 413 "payload too
large" failures from oversized tool payloads. The underlying engines' own
timeout/memory limits (unchanged: 30s / 512MB) still apply independently. Verified
in `test_tool_registration.py::test_sql_tool_truncates_when_result_exceeds_cap`.

**Not tool-registered in this pass (deliberately out of scope):**
- `backend/app/large_data/**` — this is ingestion-layer infrastructure (chunked CSV
  reading, sampling, memory-guarded aggregation operating on file paths), not an
  analysis-on-already-loaded-DataFrame capability. It is also not yet wired into the
  upload path (`app/datasets/storage.py` still fully loads every file via
  `pd.read_csv`/`pd.read_excel` regardless of size, bounded only by `max_rows`) —
  registering it as an LLM tool now would expose a capability the rest of the system
  doesn't actually support end-to-end yet. This is a separate integration decision,
  not an oversight.
- `backend/app/data/**` (DATA-ARCHITECT's `Dataset`/`Schema`/`DataSource` contracts) —
  an internal abstraction layer, not something the LLM calls directly.
