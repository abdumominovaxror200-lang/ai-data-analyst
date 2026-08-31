import type { DataCaveats, LimitationOut } from "../types";
import { Card } from "./primitives";

export function DataCaveatsCard({ caveats, limitations = [] }: { caveats: DataCaveats; limitations?: LimitationOut[] }) {
  const incomplete = caveats.column_coverage.filter((item) => item.coverage_pct < 100);
  return (
    <Card className="p-4">
      <p className="text-sm font-medium text-ink-100">Data caveats</p>
      <div className="mt-2 space-y-1 text-xs text-ink-300">
        <p>Coverage: {incomplete.length ? incomplete.map((item) => `${item.column} ${item.coverage_pct}%`).join(", ") : "100% for every column"}</p>
        <p>Duplicates: {caveats.duplicate_row_count} rows ({caveats.duplicate_pct}%)</p>
        <p>Actual date range: {Object.keys(caveats.actual_date_ranges).length ? Object.entries(caveats.actual_date_ranges).map(([column, range]) => `${column}: ${range.min ?? "unknown"} to ${range.max ?? "unknown"}`).join("; ") : "No date column detected"}</p>
        <p>Rows dropped: {caveats.rows_dropped}. {caveats.rows_dropped_note}</p>
        <p>Type anomalies: {caveats.type_anomalies.length ? caveats.type_anomalies.join("; ") : "None detected"}</p>
      </div>
      {limitations.length > 0 && <p className="mt-2 text-xs text-warn">{limitations.map((item) => item.text).join(" ")}</p>}
    </Card>
  );
}
