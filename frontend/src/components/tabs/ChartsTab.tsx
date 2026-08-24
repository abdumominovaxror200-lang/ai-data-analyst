import { useState, type ReactNode } from "react";
import { apiErrorMessage, runAnalysis } from "../../api/client";
import type { ChartData, DatasetProfile } from "../../types";
import { ChartRenderer } from "../ChartRenderer";
import { Button, Card, CardHeader, EmptyState, ErrorBanner, Spinner } from "../primitives";

const CHART_TYPES = ["bar", "line", "histogram", "scatter", "pie"] as const;
const AGG_FUNCS = ["sum", "mean", "median", "count", "min", "max"] as const;

export function ChartsTab({ profile }: { profile: DatasetProfile }) {
  const allColumns = profile.column_info.map((c) => c.name);
  const [chartType, setChartType] = useState<(typeof CHART_TYPES)[number]>("bar");
  const [x, setX] = useState(profile.categorical_columns[0] ?? allColumns[0] ?? "");
  const [y, setY] = useState(profile.numeric_columns[0] ?? "");
  const [aggFunc, setAggFunc] = useState<(typeof AGG_FUNCS)[number]>("sum");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chart, setChart] = useState<ChartData | null>(null);

  const needsY = chartType !== "histogram";

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, unknown> = { chart_type: chartType, x };
      if (needsY) {
        params.y = y;
        if (chartType !== "scatter") params.agg_func = aggFunc;
      }
      const res = await runAnalysis(profile.dataset_id, "generate_chart", params);
      setChart(res.result as ChartData);
    } catch (err) {
      setError(apiErrorMessage(err));
      setChart(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title="Build a chart" subtitle="Chart data is computed by a Python tool — never rendered from raw, unaggregated LLM output." />
        <div className="flex flex-wrap items-end gap-3 px-5 py-4">
          <Field label="Chart type">
            <Select value={chartType} onChange={(v) => setChartType(v as typeof chartType)} options={CHART_TYPES} />
          </Field>
          <Field label={chartType === "histogram" ? "Column" : "X axis"}>
            <Select value={x} onChange={setX} options={allColumns} />
          </Field>
          {needsY && (
            <Field label="Y axis / value">
              <Select value={y} onChange={setY} options={profile.numeric_columns} />
            </Field>
          )}
          {needsY && chartType !== "scatter" && (
            <Field label="Aggregation">
              <Select value={aggFunc} onChange={(v) => setAggFunc(v as typeof aggFunc)} options={AGG_FUNCS} />
            </Field>
          )}
          <Button onClick={generate} disabled={loading || !x || (needsY && !y)}>
            {loading ? <Spinner className="h-4 w-4" /> : "Generate"}
          </Button>
        </div>
      </Card>

      {error && <ErrorBanner message={error} />}

      {loading && (
        <Card>
          <div className="flex h-[280px] flex-col items-center justify-center gap-2 text-ink-400">
            <Spinner className="h-5 w-5" />
            <p className="text-xs">Computing chart data…</p>
          </div>
        </Card>
      )}

      {!loading && chart && (
        <Card>
          <CardHeader title={`${chart.x_label ?? ""}${chart.y_label ? ` vs ${chart.y_label}` : ""}`} subtitle={`${chartType} chart`} />
          <div className="px-5 py-4">
            <ChartRenderer chart={chart} />
          </div>
        </Card>
      )}

      {!loading && !chart && !error && (
        <Card>
          <EmptyState
            title="No chart yet"
            subtitle="Choose a chart type and columns above, then click Generate."
          />
        </Card>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-medium uppercase tracking-wide text-ink-400">{label}</span>
      {children}
    </label>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: readonly string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="min-w-[9rem] rounded-lg border border-ink-700 bg-ink-900/60 px-2.5 py-2 text-sm text-ink-100 focus:border-accent focus:outline-none"
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}
