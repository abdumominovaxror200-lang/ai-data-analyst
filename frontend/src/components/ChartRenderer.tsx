import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartData } from "../types";
import { EmptyState } from "./primitives";

const PALETTE = ["#5b8def", "#3fbf8f", "#e0a63c", "#e0616f", "#8fb2f5", "#9d7ff0", "#4fc3d9"];

const tooltipStyle = {
  background: "#131a29",
  border: "1px solid #212d42",
  borderRadius: 8,
  fontSize: 12,
  color: "#dde3ec",
};

function toRows(chart: ChartData): Array<Record<string, number | string>> {
  if (!chart.labels || !chart.series?.length) return [];
  return chart.labels.map((label, i) => {
    const row: Record<string, number | string> = { label };
    for (const s of chart.series!) row[s.name] = s.values[i];
    return row;
  });
}

export function ChartRenderer({ chart }: { chart: ChartData }) {
  if (chart.chart_type === "scatter") {
    if (!chart.points?.length) return <EmptyState title="No points to plot" />;
    return (
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#212d42" strokeDasharray="3 3" />
          <XAxis dataKey="x" name={chart.x_label} stroke="#5b6d8c" fontSize={12} tickLine={false} />
          <YAxis dataKey="y" name={chart.y_label} stroke="#5b6d8c" fontSize={12} tickLine={false} />
          <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3", stroke: "#3c4d6b" }} />
          <Scatter data={chart.points} fill={PALETTE[0]} />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  const rows = toRows(chart);
  if (!rows.length) return <EmptyState title="No data to display" />;
  const seriesName = chart.series?.[0]?.name ?? "value";

  if (chart.chart_type === "pie") {
    const pieData = rows.map((r) => ({ name: String(r.label), value: Number(r[seriesName]) }));
    return (
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={95} paddingAngle={2}>
            {pieData.map((_, i) => (
              <Cell key={i} fill={PALETTE[i % PALETTE.length]} stroke="#0f1420" strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip contentStyle={tooltipStyle} />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (chart.chart_type === "line") {
    return (
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#212d42" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" stroke="#5b6d8c" fontSize={11} tickLine={false} minTickGap={24} />
          <YAxis stroke="#5b6d8c" fontSize={12} tickLine={false} width={48} />
          <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: "#3c4d6b" }} />
          <Line type="monotone" dataKey={seriesName} stroke={PALETTE[0]} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  // bar + histogram
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke="#212d42" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" stroke="#5b6d8c" fontSize={11} tickLine={false} minTickGap={16} interval="preserveStartEnd" />
        <YAxis stroke="#5b6d8c" fontSize={12} tickLine={false} width={48} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(91,141,239,0.08)" }} />
        <Bar dataKey={seriesName} fill={PALETTE[0]} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
