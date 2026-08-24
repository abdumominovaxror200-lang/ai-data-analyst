import { useCallback, useRef, useState } from "react";
import { apiErrorMessage, uploadDataset } from "../api/client";
import type { UploadResponse } from "../types";
import { Button, ErrorBanner, Spinner } from "./primitives";

const FEATURES = [
  { title: "Ask in plain language", desc: "“Which region underperformed?” — no formulas required." },
  { title: "Real numbers, always", desc: "Every figure comes from deterministic Python — the AI never invents data." },
  { title: "Anomalies & correlations", desc: "IQR / z-score outlier detection and correlation analysis, automatically." },
  { title: "Export a report", desc: "Turn findings into a structured business report in one click." },
];

export function UploadView({ onUploaded }: { onUploaded: (res: UploadResponse) => void }) {
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setLoading(true);
      try {
        const res = await uploadDataset(file);
        onUploaded(res);
      } catch (err) {
        setError(apiErrorMessage(err));
      } finally {
        setLoading(false);
      }
    },
    [onUploaded]
  );

  return (
    <div className="flex min-h-screen flex-col items-center bg-ink-950 px-6 py-16">
      <div
        className="pointer-events-none fixed inset-0 opacity-40"
        style={{
          background:
            "radial-gradient(600px circle at 20% 10%, rgba(91,141,239,0.12), transparent 60%), radial-gradient(500px circle at 85% 30%, rgba(63,191,143,0.08), transparent 60%)",
        }}
      />
      <div className="relative z-10 w-full max-w-2xl text-center">
        <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-ink-700 bg-ink-850/70 px-3 py-1 text-xs font-medium text-ink-300">
          <span className="h-1.5 w-1.5 rounded-full bg-good" />
          AI Data Analyst
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-ink-50 sm:text-5xl">
          Upload a spreadsheet.
          <br />
          <span className="text-accent-soft">Get a business analyst.</span>
        </h1>
        <p className="mx-auto mt-4 max-w-md text-sm text-ink-300">
          Drop in a CSV or Excel file. The agent profiles it, answers questions, finds anomalies, and
          writes up findings — backed entirely by real Python calculations.
        </p>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files?.[0];
            if (file) handleFile(file);
          }}
          className={`mt-8 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl2 border-2 border-dashed px-8 py-14 transition-colors ${
            dragOver ? "border-accent bg-accent/5" : "border-ink-700 bg-ink-850/40 hover:border-ink-600"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
          {loading ? (
            <>
              <Spinner className="h-6 w-6 text-accent" />
              <p className="text-sm text-ink-200">Uploading and profiling dataset…</p>
            </>
          ) : (
            <>
              <div className="rounded-full bg-ink-700/60 p-3 text-accent-soft">
                <UploadIcon />
              </div>
              <p className="text-sm font-medium text-ink-100">Drag & drop a .csv or .xlsx file</p>
              <p className="text-xs text-ink-400">or click to browse · max 25 MB</p>
            </>
          )}
        </label>

        {error && (
          <div className="mt-4 text-left">
            <ErrorBanner message={error} />
          </div>
        )}

        <div className="mt-4">
          <Button variant="secondary" onClick={() => inputRef.current?.click()} disabled={loading}>
            Choose a file
          </Button>
        </div>

        <div className="mt-14 grid grid-cols-1 gap-3 text-left sm:grid-cols-2">
          {FEATURES.map((f) => (
            <div key={f.title} className="rounded-xl2 border border-ink-700/60 bg-ink-850/40 px-4 py-3">
              <p className="text-sm font-medium text-ink-50">{f.title}</p>
              <p className="mt-0.5 text-xs text-ink-400">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
