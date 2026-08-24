export type ColumnRole = "numeric" | "categorical" | "datetime" | "boolean" | "text";

export interface ColumnInfo {
  name: string;
  dtype: string;
  role: ColumnRole;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
}

export interface DatasetProfile {
  dataset_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_info: ColumnInfo[];
  numeric_columns: string[];
  categorical_columns: string[];
  date_columns: string[];
  boolean_columns: string[];
  missing_total: number;
  duplicate_rows: number;
  uploaded_at: string;
}

export interface UploadResponse {
  dataset_id: string;
  profile: DatasetProfile;
}

export interface ToolCallRecord {
  tool: string;
  params: Record<string, unknown>;
  result: Record<string, any>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChartSeries {
  name: string;
  values: number[];
}

export interface ChartPoint {
  x: number;
  y: number;
}

export interface ChartData {
  chart_type: "line" | "bar" | "histogram" | "scatter" | "pie";
  x_label?: string;
  y_label?: string;
  labels?: string[];
  series?: ChartSeries[];
  points?: ChartPoint[];
}

export interface ChatResponse {
  answer: string;
  tool_calls: ToolCallRecord[];
  charts: ChartData[];
}

export interface AnalysisResponse {
  tool: string;
  result: Record<string, any>;
  elapsed_ms: number;
}

export interface ReportPayload {
  dataset_id: string;
  filename: string;
  generated_at: string;
  overview: { rows: number; columns: number; missing_total: number; duplicate_rows: number };
  statistics: { row_count: number; columns: Record<string, any> };
  anomalies: Array<Record<string, any>>;
  correlations: Array<{ column_a: string; column_b: string; correlation: number }>;
  key_findings: string[];
}

export interface ReportResponse {
  dataset_id: string;
  generated_at: string;
  report: ReportPayload;
}
