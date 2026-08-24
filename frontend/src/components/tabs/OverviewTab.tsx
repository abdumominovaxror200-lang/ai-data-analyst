import type { DatasetProfile } from "../../types";
import { Badge, Card, CardHeader, StatTile } from "../primitives";

const ROLE_TONE: Record<string, "accent" | "good" | "warn" | "neutral"> = {
  numeric: "accent",
  categorical: "good",
  datetime: "warn",
  boolean: "neutral",
  text: "neutral",
};

export function OverviewTab({ profile }: { profile: DatasetProfile }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Rows" value={profile.rows.toLocaleString()} />
        <StatTile label="Columns" value={profile.columns} />
        <StatTile
          label="Missing values"
          value={profile.missing_total.toLocaleString()}
          hint={profile.missing_total === 0 ? "Clean dataset" : undefined}
        />
        <StatTile label="Duplicate rows" value={profile.duplicate_rows.toLocaleString()} />
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatTile label="Numeric columns" value={profile.numeric_columns.length} />
        <StatTile label="Categorical columns" value={profile.categorical_columns.length} />
        <StatTile label="Date columns" value={profile.date_columns.length} />
      </div>

      <Card>
        <CardHeader title="Columns" subtitle="Type, missing data, and cardinality for every column." />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-ink-700/60 text-xs uppercase tracking-wide text-ink-400">
                <th className="px-5 py-2.5 font-medium">Column</th>
                <th className="px-5 py-2.5 font-medium">Type</th>
                <th className="px-5 py-2.5 font-medium">Role</th>
                <th className="px-5 py-2.5 font-medium">Missing</th>
                <th className="px-5 py-2.5 font-medium">Unique</th>
              </tr>
            </thead>
            <tbody>
              {profile.column_info.map((col) => (
                <tr key={col.name} className="border-b border-ink-700/30 last:border-0 hover:bg-ink-800/30">
                  <td className="px-5 py-2.5 font-medium text-ink-100">{col.name}</td>
                  <td className="px-5 py-2.5 font-mono text-xs text-ink-400">{col.dtype}</td>
                  <td className="px-5 py-2.5">
                    <Badge tone={ROLE_TONE[col.role] ?? "neutral"}>{col.role}</Badge>
                  </td>
                  <td className="px-5 py-2.5 text-ink-300">
                    {col.missing_count > 0 ? `${col.missing_count} (${col.missing_pct}%)` : "—"}
                  </td>
                  <td className="px-5 py-2.5 text-ink-300">{col.unique_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
