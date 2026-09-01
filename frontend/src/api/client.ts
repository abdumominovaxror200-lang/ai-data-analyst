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

export const DATASET_EXPIRED_MESSAGE =
  "The server was restarted and this uploaded dataset is no longer available. Please upload the file again.";

export class DatasetExpiredError extends Error {
  constructor() {
    super(DATASET_EXPIRED_MESSAGE);
    this.name = "DatasetExpiredError";
  }
}

type DatasetExpiredListener = (datasetId: string) => void;
const expiredDatasetIds = new Set<string>();
const datasetExpiredListeners = new Set<DatasetExpiredListener>();

export function subscribeToDatasetExpiry(listener: DatasetExpiredListener): () => void {
  datasetExpiredListeners.add(listener);
  return () => datasetExpiredListeners.delete(listener);
}

export function markDatasetAvailable(datasetId: string): void {
  expiredDatasetIds.delete(datasetId);
}

function requireAvailableDataset(datasetId: string): void {
  if (expiredDatasetIds.has(datasetId)) throw new DatasetExpiredError();
}

api.interceptors.response.use(undefined, async (error: unknown) => {
  if (axios.isAxiosError(error) && error.response?.status === 404) {
    let detail = error.response.data?.detail;
    if (typeof Blob !== "undefined" && error.response.data instanceof Blob) {
      try {
        detail = JSON.parse(await error.response.data.text()).detail;
      } catch {
        // Preserve the ordinary API error below when a blob is not JSON.
      }
    }
    const match = typeof detail === "string" ? /^Dataset '([^']+)' not found\.?$/.exec(detail) : null;
    const datasetId = match?.[1];
    if (datasetId) {
      if (!expiredDatasetIds.has(datasetId)) {
        expiredDatasetIds.add(datasetId);
        datasetExpiredListeners.forEach((listener) => listener(datasetId));
      }
      return Promise.reject(new DatasetExpiredError());
    }
  }
  return Promise.reject(error);
});

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<UploadResponse>("/datasets/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getDataset(id: string): Promise<DatasetProfile> {
  requireAvailableDataset(id);
  const { data } = await api.get<DatasetProfile>(`/datasets/${id}`);
  return data;
}

export async function runAnalysis(
  datasetId: string,
  tool: string,
  params: Record<string, unknown> = {}
): Promise<AnalysisResponse> {
  requireAvailableDataset(datasetId);
  const { data } = await api.post<AnalysisResponse>("/analysis", { dataset_id: datasetId, tool, params });
  return data;
}

export async function sendChatMessage(
  datasetId: string,
  message: string,
  history: ChatMessage[]
): Promise<ChatResponse> {
  requireAvailableDataset(datasetId);
  const { data } = await api.post<ChatResponse>("/chat", { dataset_id: datasetId, message, history });
  return data;
}

export async function generateReport(datasetId: string): Promise<ReportResponse> {
  requireAvailableDataset(datasetId);
  const { data } = await api.post<ReportResponse>("/reports", { dataset_id: datasetId });
  return data;
}

export async function downloadExcelReport(datasetId: string): Promise<Blob> {
  requireAvailableDataset(datasetId);
  const { data } = await api.get<Blob>(`/reports/${datasetId}/excel`, { responseType: "blob" });
  return data;
}

export async function runReasoning(datasetId: string, message: string): Promise<ReasonResponse> {
  requireAvailableDataset(datasetId);
  const { data } = await api.post<ReasonResponse>("/reason", { dataset_id: datasetId, message });
  return data;
}

export function apiErrorMessage(err: unknown): string {
  if (err instanceof DatasetExpiredError) return DATASET_EXPIRED_MESSAGE;
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (!err.response) return "Could not reach the server. Check your connection and try again.";
    return "Something went wrong. Please try again.";
  }
  return err instanceof Error ? err.message : "Something went wrong.";
}
