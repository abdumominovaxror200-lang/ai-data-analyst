import { useState } from "react";
import { apiErrorMessage, runAnalysis } from "../../api/client";
import type { DatasetProfile } from "../../types";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorBanner, Spinner } from "../primitives";

interface AnomalyResult {
  method: string;
  column: string;
  threshold: number;
  bounds: { lower: number; upper: number } | null;
  anomaly_count: number;
  anomaly_pct: number;
  anomalies: Array<Record<string, any>>;
}

export function AnomaliesTab({ profile }: { profile: DatasetProfile }) {
  const [column, setColumn] = useState(profile.numeric_columns[0] ?? "");
  const [method, setMethod] = useState<"iqr" | "zscore">("iqr");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnomalyResult | null>(null);

  const detect = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runAnalysis(profile.dataset_id, "detect_anomalies", { column, method });
      setResult(res.result as AnomalyResult);
    } catch (err) {
      setError(apiErrorMessage(err));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  if (profile.numeric_columns.length === 0) {
    return <EmptyState title="No numeric columns" subtitle="Anomaly detection requires at least one numeric column." />;
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title="Detect anomalies" subtitle="Statistical outlier detection — IQR (robust to skew) or z-score." />
        <div className="flex flex-wrap items-end gap-3 px-5 py-4">
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-ink-400">Column</span>
            <select
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              className="min-w-[10rem] rounded-lg border border-ink-700 bg-ink-900/60 px-2.5 py-2 text-sm text-ink-100 focus:border-accent focus:outline-none"
            >
              {profile.numeric_columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-ink-400">Method</span>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as "iqr" | "zscore")}
              className="rounded-lg border border-ink-700 bg-ink-900/60 px-2.5 py-2 text-sm text-ink-100 focus:border-accent focus:outline-none"
            >
              <option value="iqr">IQR</option>
              <option value="zscore">Z-score</option>
            </select>
          </label>
          <Button onClick={detect} disabled={loading || !column}>
            {loading ? <Spinner className="h-4 w-4" /> : "Detect"}
          </Button>
        </div>
      </Card>

      {error && <ErrorBanner message={error} />}

      {result && (
        <Card>
          <CardHeader
            title={`${result.anomaly_count} anomalies in "${result.column}"`}
            subtitle={`${result.anomaly_pct}% of values · ${result.method.toUpperCase()} method${
              result.bounds ? ` · normal range ${result.bounds.lower} – ${result.bounds.upper}` : ""
            }`}
            action={<Badge tone={result.anomaly_count > 0 ? "bad" : "good"}>{result.anomaly_count > 0 ? "Outliers found" : "Clean"}</Badge>}
          />
          {result.anomalies.length === 0 ? (
            <EmptyState title="No anomalies detected" subtitle="Every value in this column falls within the expected range." />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-ink-700/60 text-xs uppercase tracking-wide text-ink-400">
                    {Object.keys(result.anomalies[0]).map((key) => (
                      <th key={key} className="whitespace-nowrap px-5 py-2.5 font-medium">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.anomalies.map((row, i) => (
                    <tr key={i} className="border-b border-ink-700/30 last:border-0 hover:bg-ink-800/30">
                      {Object.entries(row).map(([key, value]) => (
                        <td
                          key={key}
                          className={`whitespace-nowrap px-5 py-2.5 tabular-nums ${
                            key === result.column ? "font-semibold text-bad" : "text-ink-300"
                          }`}
                        >
                          {String(value ?? "—")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
