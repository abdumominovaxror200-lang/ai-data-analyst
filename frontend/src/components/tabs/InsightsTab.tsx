import { useEffect, useState } from "react";
import { apiErrorMessage, runAnalysis } from "../../api/client";
import type { DatasetProfile } from "../../types";
import { Badge, Card, CardHeader, EmptyState, ErrorBanner, Spinner, StatTile } from "../primitives";

interface Insights {
  profile_summary: { rows: number; columns: number; numeric_columns: string[]; categorical_columns: string[] };
  statistics: { row_count: number; columns: Record<string, any> };
  anomalies: Array<{ column: string; anomaly_count: number; anomaly_pct: number; method: string }>;
  top_correlations: Array<{ column_a: string; column_b: string; correlation: number }>;
  data_quality_issues: string[];
}

export function InsightsTab({ profile }: { profile: DatasetProfile }) {
  const [data, setData] = useState<Insights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    runAnalysis(profile.dataset_id, "generate_business_insights")
      .then((res) => {
        if (!cancelled) setData(res.result as Insights);
      })
      .catch((err) => {
        if (!cancelled) setError(apiErrorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [profile.dataset_id]);

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-ink-400">
        <Spinner className="h-4 w-4" /> Computing insights…
      </div>
    );
  if (error) return <ErrorBanner message={error} />;
  if (!data) return <EmptyState title="No insights available" />;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Rows analyzed" value={data.profile_summary.rows.toLocaleString()} />
        <StatTile label="Anomalous columns" value={data.anomalies.length} />
        <StatTile label="Strong correlations" value={data.top_correlations.length} />
        <StatTile label="Data quality flags" value={data.data_quality_issues.length} />
      </div>

      <Card>
        <CardHeader title="Statistical summary" subtitle="Computed by describe_data over all numeric columns." />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-ink-700/60 text-xs uppercase tracking-wide text-ink-400">
                <th className="px-5 py-2.5 font-medium">Column</th>
                <th className="px-5 py-2.5 font-medium">Mean</th>
                <th className="px-5 py-2.5 font-medium">Median</th>
                <th className="px-5 py-2.5 font-medium">Std dev</th>
                <th className="px-5 py-2.5 font-medium">Min</th>
                <th className="px-5 py-2.5 font-medium">Max</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.statistics.columns).map(([col, s]: [string, any]) => (
                <tr key={col} className="border-b border-ink-700/30 last:border-0 hover:bg-ink-800/30">
                  <td className="px-5 py-2.5 font-medium text-ink-100">{col}</td>
                  <td className="px-5 py-2.5 tabular-nums text-ink-300">{fmt(s.mean)}</td>
                  <td className="px-5 py-2.5 tabular-nums text-ink-300">{fmt(s.median)}</td>
                  <td className="px-5 py-2.5 tabular-nums text-ink-300">{fmt(s.std)}</td>
                  <td className="px-5 py-2.5 tabular-nums text-ink-300">{fmt(s.min)}</td>
                  <td className="px-5 py-2.5 tabular-nums text-ink-300">{fmt(s.max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader title="Strongest correlations" />
        {data.top_correlations.length === 0 ? (
          <EmptyState title="No strong correlations found" />
        ) : (
          <div className="space-y-2 px-5 py-4">
            {data.top_correlations.slice(0, 8).map((pair, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="text-ink-200">
                  {pair.column_a} <span className="text-ink-500">↔</span> {pair.column_b}
                </span>
                <Badge tone={Math.abs(pair.correlation) > 0.7 ? "accent" : "neutral"}>{pair.correlation.toFixed(3)}</Badge>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHeader title="Data quality" />
        {data.data_quality_issues.length === 0 ? (
          <EmptyState title="No data quality issues detected" />
        ) : (
          <ul className="space-y-1.5 px-5 py-4 text-sm text-ink-200">
            {data.data_quality_issues.map((issue, i) => (
              <li key={i} className="flex items-center gap-2">
                <Badge tone="warn">!</Badge> {issue}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
