import { useEffect, useRef, useState } from "react";
import { apiErrorMessage, runReasoning } from "../../api/client";
import type {
  DatasetProfile,
  EvidenceOut,
  FindingClassification,
  FindingOut,
  HypothesisOut,
  LimitationOut,
  ReasonResponse,
  RecommendationOut,
  UncertaintyOut,
} from "../../types";
import { MarkdownMessage } from "../MarkdownMessage";
import { Badge, Button, Card, CardHeader, ErrorBanner, Spinner } from "../primitives";

const SUGGESTIONS = [
  "What's driving the biggest change in this data, and how confident can we be?",
  "Is there a real relationship here, or could it just be correlation?",
  "What should we do next based on this data, and what are the risks?",
  "What don't we know yet from this dataset?",
];

// /api/reason runs 3 structured LLM calls plus tool execution -- meaningfully
// slower than /api/chat's single call. Cycling through real pipeline stages
// (mirrors orchestrator.py: parse -> plan -> execute -> verify -> synthesize)
// gives honest progressive feedback instead of a bare spinner.
const LOADING_STAGES = [
  "Parsing your question…",
  "Planning which tools to run…",
  "Executing analysis tools…",
  "Verifying findings against evidence…",
  "Checking for causal-safety and epistemic issues…",
  "Synthesizing a grounded answer…",
];

const CLASSIFICATION_META: Record<FindingClassification, { label: string; tone: "good" | "accent" | "stat" | "hypo" | "warn" | "neutral" }> = {
  FACT: { label: "Fact", tone: "good" },
  CALCULATED_RESULT: { label: "Calculated", tone: "accent" },
  STATISTICAL_RESULT: { label: "Statistical", tone: "stat" },
  HYPOTHESIS: { label: "Hypothesis", tone: "hypo" },
  ASSUMPTION: { label: "Assumption", tone: "warn" },
  UNKNOWN: { label: "Unknown", tone: "neutral" },
};

const SEVERITY_META: Record<string, { label: string; tone: "bad" | "warn" | "neutral"; className: string }> = {
  blocks_conclusion: { label: "Blocks conclusion", tone: "bad", className: "border-bad/40 bg-bad/10" },
  reduces_confidence: { label: "Reduces confidence", tone: "warn", className: "border-warn/30 bg-warn/5" },
  minor: { label: "Minor", tone: "neutral", className: "border-ink-700/60 bg-ink-900/30" },
};

const HYPOTHESIS_STATUS_META: Record<string, { label: string; tone: "good" | "warn" | "bad" | "neutral" | "accent" }> = {
  untested: { label: "Untested", tone: "neutral" },
  supported: { label: "Supported", tone: "good" },
  weakly_supported: { label: "Weakly supported", tone: "warn" },
  unsupported: { label: "Unsupported", tone: "bad" },
  contradicted: { label: "Contradicted", tone: "bad" },
  inconclusive: { label: "Inconclusive", tone: "warn" },
};

const CONFIDENCE_META: Record<string, { label: string; tone: "good" | "warn" | "bad" }> = {
  high: { label: "High confidence", tone: "good" },
  medium: { label: "Medium confidence", tone: "warn" },
  low: { label: "Low confidence", tone: "bad" },
};

interface Turn {
  question: string;
  response?: ReasonResponse;
  error?: string;
}

export function ReasoningTab({ profile }: { profile: DatasetProfile }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stageIndex, setStageIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!loading) {
      setStageIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setStageIndex((i) => Math.min(i + 1, LOADING_STAGES.length - 1));
    }, 2800);
    return () => clearInterval(timer);
  }, [loading]);

  const ask = async (question: string) => {
    if (!question.trim() || loading) return;
    const nextTurns: Turn[] = [...turns, { question }];
    setTurns(nextTurns);
    setInput("");
    setLoading(true);
    try {
      const response = await runReasoning(profile.dataset_id, question);
      setTurns([...nextTurns.slice(0, -1), { question, response }]);
    } catch (err) {
      setTurns([...nextTurns.slice(0, -1), { question, error: apiErrorMessage(err) }]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }));
    }
  };

  return (
    <div className="flex h-[calc(100vh-7.5rem)] flex-col">
      <div ref={scrollRef} className="flex-1 space-y-6 overflow-y-auto pb-4">
        {turns.length === 0 && (
          <Card className="p-6">
            <p className="text-sm font-medium text-ink-100">Ask a grounded, evidence-classified question about “{profile.filename}”.</p>
            <p className="mt-1 text-xs text-ink-400">
              This runs the full reasoning pipeline — plan, execute, verify, and synthesize — and classifies every
              conclusion as fact, calculated result, statistical result, hypothesis, assumption, or unknown, with the
              real computed evidence behind it. Slower than the AI Analyst chat, but fully traceable.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => ask(s)}
                  className="rounded-lg border border-ink-700/60 bg-ink-900/40 px-3 py-2.5 text-left text-xs text-ink-300 transition-colors hover:border-accent/50 hover:text-ink-100"
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="space-y-3">
            <div className="flex justify-end">
              <div className="max-w-2xl rounded-xl2 bg-accent px-4 py-2.5 text-sm text-white">{turn.question}</div>
            </div>

            {turn.error && <ErrorBanner message={turn.error} />}
            {turn.response && <ReasoningResult response={turn.response} />}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-xs text-ink-400">
            <Spinner className="h-3.5 w-3.5" />
            {LOADING_STAGES[stageIndex]}
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="flex items-center gap-2 border-t border-ink-700/60 pt-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question for the full reasoning pipeline…"
          disabled={loading}
          className="flex-1 rounded-lg border border-ink-700 bg-ink-900/60 px-3.5 py-2.5 text-sm text-ink-50 placeholder:text-ink-400 focus:border-accent focus:outline-none disabled:opacity-60"
        />
        <Button type="submit" disabled={loading || !input.trim()}>
          {loading ? <Spinner className="h-4 w-4" /> : "Reason"}
        </Button>
      </form>
    </div>
  );
}

function ReasoningResult({ response }: { response: ReasonResponse }) {
  const [traceOpen, setTraceOpen] = useState(false);
  const evidenceById = new Map<string, EvidenceOut>(response.evidence.map((e) => [e.id, e]));

  return (
    <div className="w-full space-y-4">
      <Card className="p-4">
        <div className="mb-2 flex items-center gap-2">
          <Badge tone="neutral">{response.intent}</Badge>
        </div>
        <MarkdownMessage content={response.answer} />
      </Card>

      {response.principle_violations.length > 0 && (
        <div className="flex items-start gap-2 rounded-lg border border-bad/40 bg-bad/10 px-3.5 py-2.5 text-sm text-bad">
          <span className="mt-0.5">⚑</span>
          <div>
            <p className="font-medium">Internal consistency check flagged this answer</p>
            <ul className="mt-1 space-y-0.5 text-xs text-bad/90">
              {response.principle_violations.map((v, i) => (
                <li key={i}>{v}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {response.findings.length > 0 && (
        <Card>
          <CardHeader title="Findings" subtitle="Every conclusion classified by how it was established, with its evidence and uncertainty." />
          <div className="divide-y divide-ink-700/40">
            {response.findings.map((f) => (
              <FindingRow key={f.id} finding={f} evidenceById={evidenceById} />
            ))}
          </div>
        </Card>
      )}

      {response.limitations.length > 0 && (
        <Card>
          <CardHeader title="Limitations" subtitle="Shown by default — these constrain how far the findings above can be trusted." />
          <div className="space-y-2 px-5 py-4">
            {[...response.limitations]
              .sort((a, b) => severityRank(a.severity) - severityRank(b.severity))
              .map((l, i) => (
                <LimitationRow key={i} limitation={l} />
              ))}
          </div>
        </Card>
      )}

      {response.hypotheses.length > 0 && (
        <Card>
          <CardHeader title="Hypotheses" subtitle="Competing explanations considered, with their evidentiary status." />
          <div className="divide-y divide-ink-700/40">
            {response.hypotheses.map((h) => (
              <HypothesisRow key={h.id} hypothesis={h} />
            ))}
          </div>
        </Card>
      )}

      {response.recommendation && <RecommendationCard recommendation={response.recommendation} />}

      <div className="rounded-lg border border-ink-700/60 bg-ink-900/30">
        <button
          onClick={() => setTraceOpen((o) => !o)}
          className="flex w-full items-center justify-between px-4 py-2.5 text-xs font-medium text-ink-300 hover:text-ink-100"
        >
          <span>
            Reasoning trace &amp; tools used
            {response.tools_used.length > 0 && <span className="ml-2 text-ink-500">({response.tools_used.length} tool calls)</span>}
          </span>
          <span className={`transition-transform ${traceOpen ? "rotate-180" : ""}`}>⌄</span>
        </button>
        {traceOpen && (
          <div className="space-y-3 border-t border-ink-700/60 px-4 py-3">
            {response.tools_used.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {response.tools_used.map((t, i) => (
                  <Badge key={i} tone="accent">
                    {t}
                  </Badge>
                ))}
              </div>
            )}
            {response.reasoning_trace.length > 0 && (
              <ol className="space-y-1 text-xs text-ink-400">
                {response.reasoning_trace.map((step, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-ink-600">{i + 1}.</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function severityRank(severity: string): number {
  if (severity === "blocks_conclusion") return 0;
  if (severity === "reduces_confidence") return 1;
  return 2;
}

function FindingRow({ finding, evidenceById }: { finding: FindingOut; evidenceById: Map<string, EvidenceOut> }) {
  const meta = CLASSIFICATION_META[finding.classification] ?? CLASSIFICATION_META.UNKNOWN;
  const linkedEvidence = finding.supporting_evidence.map((id) => evidenceById.get(id)).filter((e): e is EvidenceOut => !!e);

  return (
    <div className="px-5 py-3.5">
      <div className="flex items-start gap-2.5">
        <Badge tone={meta.tone}>{meta.label}</Badge>
        <p className="flex-1 text-sm text-ink-100">{finding.statement}</p>
        {finding.cross_checked && (
          <span title="Cross-checked against a second source" className="text-xs text-good">
            ✓
          </span>
        )}
      </div>

      {finding.uncertainty && <UncertaintyLine uncertainty={finding.uncertainty} />}

      {linkedEvidence.length > 0 && (
        <div className="mt-2 space-y-1.5 pl-0.5">
          {linkedEvidence.map((e) => (
            <div key={e.id} className="rounded-md border border-ink-700/50 bg-ink-900/40 px-3 py-2 text-xs">
              <div className="flex flex-wrap items-center gap-1.5 text-ink-400">
                <span className="font-medium text-ink-300">{e.source_tool}</span>
                {e.metric && <span>· {e.metric}</span>}
                {e.sample_size != null && <span>· n={e.sample_size.toLocaleString()}</span>}
              </div>
              <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 tabular-nums text-ink-200">
                {salientEntries(e.result_summary).map(([k, v]) => (
                  <span key={k}>
                    <span className="text-ink-500">{k}:</span> {v}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function UncertaintyLine({ uncertainty }: { uncertainty: UncertaintyOut }) {
  const levelTone: Record<UncertaintyOut["level"], "good" | "warn" | "bad" | "neutral"> = {
    known: "good",
    estimated: "warn",
    uncertain: "warn",
    unavailable: "bad",
  };
  const parts: string[] = [];
  if (uncertainty.point_estimate != null) parts.push(`point estimate ${formatNum(uncertainty.point_estimate)}`);
  if (uncertainty.interval_low != null && uncertainty.interval_high != null) {
    parts.push(`interval [${formatNum(uncertainty.interval_low)}, ${formatNum(uncertainty.interval_high)}]`);
  }
  if (uncertainty.confidence_level != null) parts.push(`${(uncertainty.confidence_level * 100).toFixed(0)}% confidence level`);
  if (uncertainty.method) parts.push(uncertainty.method);

  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1.5 pl-0.5 text-xs text-ink-400">
      <Badge tone={levelTone[uncertainty.level]}>{uncertainty.level}</Badge>
      {parts.length > 0 && <span>{parts.join(" · ")}</span>}
    </div>
  );
}

function LimitationRow({ limitation }: { limitation: LimitationOut }) {
  const meta = SEVERITY_META[limitation.severity] ?? SEVERITY_META.reduces_confidence;
  const isBlocking = limitation.severity === "blocks_conclusion";
  return (
    <div className={`rounded-lg border px-3.5 py-2.5 ${meta.className}`}>
      <div className="flex items-start gap-2">
        <Badge tone={meta.tone}>{meta.label}</Badge>
        <Badge tone="neutral">{limitation.category.replace(/_/g, " ")}</Badge>
      </div>
      <p className={`mt-1.5 ${isBlocking ? "text-sm font-medium text-ink-50" : "text-sm text-ink-200"}`}>{limitation.text}</p>
    </div>
  );
}

function HypothesisRow({ hypothesis }: { hypothesis: HypothesisOut }) {
  const meta = HYPOTHESIS_STATUS_META[hypothesis.status] ?? HYPOTHESIS_STATUS_META.untested;
  // A causal hypothesis that isn't cleanly "supported" must not read as confident --
  // hedge visually (muted text, no strong tone) rather than let a neutral badge imply
  // more certainty than the evidence backs.
  const causalHedge = hypothesis.is_causal && hypothesis.status !== "supported";

  return (
    <div className="px-5 py-3.5">
      <div className="flex flex-wrap items-start gap-2">
        {hypothesis.is_causal && <Badge tone={causalHedge ? "warn" : "accent"}>Causal claim</Badge>}
        <Badge tone={meta.tone}>{meta.label}</Badge>
      </div>
      <p className={`mt-1.5 text-sm ${causalHedge ? "italic text-ink-300" : "text-ink-100"}`}>{hypothesis.description}</p>
      <div className="mt-1 text-xs text-ink-500">
        {hypothesis.evidence_for.length} evidence for · {hypothesis.evidence_against.length} evidence against
      </div>
    </div>
  );
}

function RecommendationCard({ recommendation }: { recommendation: RecommendationOut }) {
  const confMeta = recommendation.confidence ? CONFIDENCE_META[recommendation.confidence] : null;
  return (
    <Card className="border-accent/30">
      <CardHeader
        title="Recommendation"
        action={
          confMeta ? (
            <Badge tone={confMeta.tone}>{confMeta.label}</Badge>
          ) : (
            <Badge tone="neutral">No confidence rating</Badge>
          )
        }
      />
      <div className="space-y-3 px-5 py-4">
        <p className="text-sm text-ink-100">{recommendation.recommendation}</p>

        {!recommendation.confidence && (
          <p className="rounded-md border border-ink-700/60 bg-ink-900/40 px-3 py-2 text-xs italic text-ink-400">
            The evidence gathered here is insufficient to assign a confidence rating — this is a deliberate signal,
            not a missing value. Treat this recommendation as a starting hypothesis, not a settled conclusion.
          </p>
        )}

        {recommendation.expected_business_effect && (
          <p className="text-xs text-ink-300">
            <span className="font-medium text-ink-200">Expected effect:</span> {recommendation.expected_business_effect}
          </p>
        )}

        {recommendation.assumptions.length > 0 && (
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide text-ink-400">Assumptions</p>
            <ul className="mt-1 space-y-0.5 text-xs text-ink-300">
              {recommendation.assumptions.map((a, i) => (
                <li key={i}>· {a}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendation.risks.length > 0 && (
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide text-warn">Risks</p>
            <ul className="mt-1 space-y-0.5 text-xs text-ink-300">
              {recommendation.risks.map((r, i) => (
                <li key={i}>· {r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}

// Picks a handful of scalar (number/string/boolean) top-level fields out of an
// arbitrary tool result_summary so the UI shows real computed values instead of
// a raw JSON dump. Prefers common "the actual number" key names when present.
const PREFERRED_KEYS = ["value", "result", "mean", "median", "sum", "count", "total", "correlation", "coefficient", "p_value", "score"];

function salientEntries(summary: Record<string, unknown>): Array<[string, string]> {
  const entries = Object.entries(summary).filter(([, v]) => v === null || ["number", "string", "boolean"].includes(typeof v));
  if (entries.length === 0) {
    // Nothing scalar at the top level -- fall back to reporting array lengths /
    // presence of nested objects, still short-circuiting a raw dump.
    return Object.entries(summary)
      .slice(0, 3)
      .map(([k, v]) => [k, Array.isArray(v) ? `${v.length} items` : "…"]);
  }
  entries.sort((a, b) => {
    const ai = PREFERRED_KEYS.indexOf(a[0].toLowerCase());
    const bi = PREFERRED_KEYS.indexOf(b[0].toLowerCase());
    const ar = ai === -1 ? PREFERRED_KEYS.length : ai;
    const br = bi === -1 ? PREFERRED_KEYS.length : bi;
    return ar - br;
  });
  return entries.slice(0, 4).map(([k, v]) => [k, formatValue(v)]);
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return formatNum(v);
  return String(v);
}

function formatNum(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 3 });
}
