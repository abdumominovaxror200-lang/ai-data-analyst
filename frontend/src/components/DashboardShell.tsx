import type { PropsWithChildren, ReactNode } from "react";
import type { DatasetProfile } from "../types";
import { Badge } from "./primitives";

export type TabKey = "overview" | "chat" | "charts" | "insights" | "anomalies" | "report";

const TABS: { key: TabKey; label: string; icon: ReactNode }[] = [
  { key: "overview", label: "Overview", icon: <IconGrid /> },
  { key: "chat", label: "AI Analyst", icon: <IconChat /> },
  { key: "charts", label: "Charts", icon: <IconChart /> },
  { key: "insights", label: "Insights", icon: <IconSpark /> },
  { key: "anomalies", label: "Anomalies", icon: <IconAlert /> },
  { key: "report", label: "Report", icon: <IconDoc /> },
];

export function DashboardShell({
  profile,
  active,
  onTabChange,
  onReset,
  children,
}: PropsWithChildren<{
  profile: DatasetProfile;
  active: TabKey;
  onTabChange: (tab: TabKey) => void;
  onReset: () => void;
}>) {
  return (
    <div className="flex min-h-screen bg-ink-950">
      <aside className="flex w-60 shrink-0 flex-col border-r border-ink-700/60 bg-ink-900/60 px-3 py-4">
        <div className="mb-6 flex items-center gap-2 px-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm font-bold text-white">
            A
          </div>
          <span className="text-sm font-semibold text-ink-50">AI Data Analyst</span>
        </div>

        <nav className="flex flex-col gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => onTabChange(tab.key)}
              className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors ${
                active === tab.key
                  ? "bg-accent/15 text-accent-soft"
                  : "text-ink-300 hover:bg-ink-800/80 hover:text-ink-100"
              }`}
            >
              <span className={active === tab.key ? "text-accent-soft" : "text-ink-400"}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto space-y-2 border-t border-ink-700/60 pt-3">
          <button
            onClick={onReset}
            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs font-medium text-ink-400 hover:bg-ink-800/80 hover:text-ink-100"
          >
            <IconSwap /> Upload a different dataset
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-700/60 bg-ink-900/40 px-6 py-3.5">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-ink-50">{profile.filename}</p>
            <p className="text-xs text-ink-400">
              {profile.rows.toLocaleString()} rows · {profile.columns} columns
            </p>
          </div>
          <div className="flex items-center gap-2">
            {profile.duplicate_rows > 0 && <Badge tone="warn">{profile.duplicate_rows} duplicate rows</Badge>}
            {profile.missing_total > 0 ? (
              <Badge tone="warn">{profile.missing_total.toLocaleString()} missing values</Badge>
            ) : (
              <Badge tone="good">No missing values</Badge>
            )}
          </div>
        </header>

        <main className="flex-1 overflow-y-auto px-6 py-6">{children}</main>
      </div>
    </div>
  );
}

function IconGrid() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <rect x="3" y="3" width="8" height="8" rx="1.5" />
      <rect x="13" y="3" width="8" height="8" rx="1.5" />
      <rect x="3" y="13" width="8" height="8" rx="1.5" />
      <rect x="13" y="13" width="8" height="8" rx="1.5" />
    </svg>
  );
}
function IconChat() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
    </svg>
  );
}
function IconChart() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M3 3v18h18" strokeLinecap="round" />
      <path d="M7 16l4-6 3 4 5-8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function IconSpark() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" strokeLinecap="round" />
    </svg>
  );
}
function IconAlert() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" strokeLinejoin="round" />
      <path d="M12 9v4M12 17h.01" strokeLinecap="round" />
    </svg>
  );
}
function IconDoc() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" strokeLinejoin="round" />
      <path d="M14 3v6h6M8 13h8M8 17h8M8 9h2" strokeLinecap="round" />
    </svg>
  );
}
function IconSwap() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M17 2l4 4-4 4M3 6h18M7 22l-4-4 4-4M21 18H3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
