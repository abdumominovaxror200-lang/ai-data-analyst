import { useState } from "react";
import { apiErrorMessage, generateReport } from "../../api/client";
import type { DatasetProfile, ReportPayload } from "../../types";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorBanner, Spinner, StatTile } from "../primitives";

export function ReportTab({ profile }: { profile: DatasetProfile }) {
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateReport(profile.dataset_id);
      setReport(res.report);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    if (!report) return;
    const md = renderMarkdown(report);
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${report.filename.replace(/\.[^.]+$/, "")}-report.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title="Business report"
          subtitle="A structured summary of statistics, anomalies, and correlations — computed, not narrated."
          action={
            <div className="flex gap-2">
              {report && (
                <Button variant="secondary" onClick={download}>
                  Download .md
                </Button>
              )}
              <Button onClick={generate} disabled={loading}>
                {loading ? <Spinner className="h-4 w-4" /> : report ? "Regenerate" : "Generate report"}
              </Button>
            </div>
          }
        />
      </Card>

      {error && <ErrorBanner message={error} />}

      {!report && !loading && (
        <EmptyState title="No report generated yet" subtitle="Click “Generate report” to produce a structured business summary." />
      )}

      {report && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatTile label="Rows" value={report.overview.rows.toLocaleString()} />
            <StatTile label="Columns" value={report.overview.columns} />
            <StatTile label="Missing values" value={report.overview.missing_total.toLocaleString()} />
            <StatTile label="Duplicate rows" value={report.overview.duplicate_rows.toLocaleString()} />
          </div>

          <Card>
            <CardHeader title="Key findings" />
            <ul className="space-y-2 px-5 py-4 text-sm text-ink-100">
              {report.key_findings.map((finding, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  {finding}
                </li>
              ))}
            </ul>
          </Card>

          {report.anomalies.length > 0 && (
            <Card>
              <CardHeader title="Anomalies by column" />
              <div className="space-y-2 px-5 py-4">
                {report.anomalies.map((a, i) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="text-ink-200">{a.column}</span>
                    <Badge tone="bad">
                      {a.anomaly_count} outliers ({a.anomaly_pct}%)
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {report.correlations.length > 0 && (
            <Card>
              <CardHeader title="Notable correlations" />
              <div className="space-y-2 px-5 py-4">
                {report.correlations.slice(0, 6).map((pair, i) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="text-ink-200">
                      {pair.column_a} ↔ {pair.column_b}
                    </span>
                    <Badge tone="accent">{pair.correlation.toFixed(3)}</Badge>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function renderMarkdown(report: ReportPayload): string {
  const lines = [
    `# Business Report — ${report.filename}`,
    "",
    `Generated: ${report.generated_at}`,
    "",
    "## Overview",
    `- Rows: ${report.overview.rows}`,
    `- Columns: ${report.overview.columns}`,
    `- Missing values: ${report.overview.missing_total}`,
    `- Duplicate rows: ${report.overview.duplicate_rows}`,
    "",
    "## Key findings",
    ...report.key_findings.map((f) => `- ${f}`),
    "",
  ];
  if (report.anomalies.length) {
    lines.push("## Anomalies", ...report.anomalies.map((a) => `- ${a.column}: ${a.anomaly_count} outliers (${a.anomaly_pct}%)`), "");
  }
  if (report.correlations.length) {
    lines.push(
      "## Correlations",
      ...report.correlations.slice(0, 10).map((p) => `- ${p.column_a} vs ${p.column_b}: r = ${p.correlation}`),
      ""
    );
  }
  return lines.join("\n");
}
