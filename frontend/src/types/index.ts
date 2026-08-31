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
  metrics: MetricDefinition[];
}

export interface MetricDefinition {
  name: string;
  kind: "measure" | "count" | "rate" | "ratio";
  numerator_column?: string | null;
  denominator_column?: string | null;
  numerator_aggregation?: "sum" | "count" | "mean" | null;
  denominator_aggregation?: "sum" | "count" | "count_distinct" | "row_count" | null;
  unit?: string | null;
  status: "resolved" | "needs_definition";
  reason: string;
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
  data_caveats: DataCaveats;
  limitations: LimitationOut[];
}

export interface ColumnCoverage {
  column: string;
  non_null_rows: number;
  total_rows: number;
  coverage_pct: number;
}

export interface DataCaveats {
  column_coverage: ColumnCoverage[];
  duplicate_row_count: number;
  duplicate_pct: number;
  actual_date_ranges: Record<string, { min?: string | null; max?: string | null }>;
  rows_dropped: number;
  rows_dropped_note: string;
  type_anomalies: string[];
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

// --- Reasoning layer (`/api/reason`) types.
// Mirrors backend/app/schemas.py's Reason*/EvidenceOut/etc. Pydantic models
// exactly (field names, optionality, literal unions) so the API contract stays
// traceable end to end. Additive only — see frontend/src/api/client.ts. ---

export type EvidenceType = "FACT" | "CALCULATED_RESULT" | "STATISTICAL_RESULT";

export type FindingClassification =
  | "FACT"
  | "CALCULATED_RESULT"
  | "STATISTICAL_RESULT"
  | "HYPOTHESIS"
  | "ASSUMPTION"
  | "UNKNOWN";

export type UncertaintyLevel = "known" | "estimated" | "uncertain" | "unavailable";

export type RecommendationConfidence = "high" | "medium" | "low";

export interface UncertaintyOut {
  level: UncertaintyLevel;
  point_estimate?: number | null;
  interval_low?: number | null;
  interval_high?: number | null;
  confidence_level?: number | null;
  method?: string | null;
}

export interface EvidenceOut {
  id: string;
  source_tool: string;
  evidence_type: EvidenceType;
  metric?: string | null;
  result_summary: Record<string, any>;
  sample_size?: number | null;
}

export interface FindingOut {
  id: string;
  statement: string;
  classification: FindingClassification;
  cross_checked: boolean;
  uncertainty?: UncertaintyOut | null;
  supporting_evidence: string[]; // EvidenceOut.id values
}

export interface LimitationOut {
  category: string;
  text: string;
  severity: string;
  affected_findings: string[]; // FindingOut.id values
}

export interface HypothesisOut {
  id: string;
  description: string;
  is_causal: boolean;
  status: string;
  evidence_for: string[]; // EvidenceOut.id values
  evidence_against: string[];
}

export interface RecommendationOut {
  recommendation: string;
  expected_business_effect?: string | null;
  confidence?: RecommendationConfidence | null;
  assumptions: string[];
  risks: string[];
  supporting_findings: string[]; // FindingOut.id values
}

export interface ReasonResponse {
  answer: string;
  intent: string;
  evidence: EvidenceOut[];
  findings: FindingOut[];
  limitations: LimitationOut[];
  data_caveats: DataCaveats;
  hypotheses: HypothesisOut[];
  recommendation?: RecommendationOut | null;
  tools_used: string[];
  reasoning_trace: string[];
  principle_violations: string[];
}
