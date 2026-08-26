import type { PropsWithChildren, ReactNode } from "react";

export function Card({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return (
    <div
      className={`rounded-xl2 border border-ink-700/60 bg-ink-850/60 shadow-panel backdrop-blur-sm ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-ink-700/60 px-5 py-4">
      <div>
        <h3 className="text-sm font-semibold tracking-tight text-ink-50">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-ink-300">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  type = "button",
  className = "",
}: PropsWithChildren<{
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}>) {
  const base =
    "inline-flex items-center justify-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40";
  const variants = {
    primary: "bg-accent text-white hover:bg-accent-soft shadow-[0_0_0_1px_rgba(91,141,239,0.4)]",
    secondary: "bg-ink-700 text-ink-50 hover:bg-ink-600 border border-ink-600",
    ghost: "text-ink-200 hover:bg-ink-700/60",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: PropsWithChildren<{ tone?: "neutral" | "good" | "warn" | "bad" | "accent" | "stat" | "hypo"; className?: string }>) {
  const tones = {
    neutral: "bg-ink-700 text-ink-200",
    good: "bg-good/15 text-good",
    warn: "bg-warn/15 text-warn",
    bad: "bg-bad/15 text-bad",
    accent: "bg-accent/15 text-accent-soft",
    // Added for the reasoning layer's 6-way finding classification, which needs
    // more visually distinct categories than the original 5 tones cover.
    stat: "bg-stat/15 text-stat",
    hypo: "bg-hypo/15 text-hypo",
  };
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${tones[tone]} ${className}`}>
      {children}
    </span>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl2 border border-ink-700/60 bg-ink-850/60 px-4 py-3.5">
      <p className="text-[11px] font-medium uppercase tracking-wide text-ink-300">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-ink-50">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-ink-400">{hint}</p>}
    </div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function EmptyState({ title, subtitle, icon }: { title: string; subtitle?: string; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
      {icon && <div className="mb-1 text-ink-400">{icon}</div>}
      <p className="text-sm font-medium text-ink-100">{title}</p>
      {subtitle && <p className="max-w-sm text-xs text-ink-400">{subtitle}</p>}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-bad/30 bg-bad/10 px-3.5 py-2.5 text-sm text-bad">
      <span className="mt-0.5">⚠</span>
      <span>{message}</span>
    </div>
  );
}
