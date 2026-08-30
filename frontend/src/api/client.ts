import axios from "axios";
import type {
  AnalysisResponse,
  ChatMessage,
  ChatResponse,
  DatasetProfile,
  ReasonResponse,
  ReportResponse,
  UploadResponse,
} from "../types";

function normalizeApiBase(rawValue: string): string {
  const value = rawValue.trim().replace(/^Value:\s*/i, "").replace(/\/+$/, "");
  if (value.startsWith("/")) return value;

  const url = new URL(value);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("VITE_API_BASE_URL must use http or https");
  }
  return value;
}

const API_BASE = normalizeApiBase(import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api");

export const api = axios.create({ baseURL: API_BASE });

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<UploadResponse>("/datasets/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getDataset(id: string): Promise<DatasetProfile> {
  const { data } = await api.get<DatasetProfile>(`/datasets/${id}`);
  return data;
}

export async function runAnalysis(
  datasetId: string,
  tool: string,
  params: Record<string, unknown> = {}
): Promise<AnalysisResponse> {
  const { data } = await api.post<AnalysisResponse>("/analysis", { dataset_id: datasetId, tool, params });
  return data;
}

export async function sendChatMessage(
  datasetId: string,
  message: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>("/chat", { dataset_id: datasetId, message, history });
  return data;
}

export async function generateReport(datasetId: string): Promise<ReportResponse> {
  const { data } = await api.post<ReportResponse>("/reports", { dataset_id: datasetId });
  return data;
}

export async function runReasoning(datasetId: string, message: string): Promise<ReasonResponse> {
  const { data } = await api.post<ReasonResponse>("/reason", { dataset_id: datasetId, message });
  return data;
}

export function apiErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (!err.response) return "Could not reach the server. Check your connection and try again.";
    return "Something went wrong. Please try again.";
  }
  return err instanceof Error ? err.message : "Something went wrong.";
}
