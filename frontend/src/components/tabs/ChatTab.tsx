import { useEffect, useRef, useState } from "react";
import { apiErrorMessage, sendChatMessage } from "../../api/client";
import type { ChartData, ChatMessage, DatasetProfile, ToolCallRecord } from "../../types";
import { ChartRenderer } from "../ChartRenderer";
import { MarkdownMessage } from "../MarkdownMessage";
import { Badge, Button, Card, Spinner } from "../primitives";

const SUGGESTIONS = [
  "Analyze this dataset and summarize the key findings.",
  "What are the biggest anomalies in this data?",
  "Which categories or groups perform best?",
  "Which columns are strongly correlated?",
];

const TOOL_LABELS: Record<string, string> = {
  profile_dataset: "Dataset profile",
  describe_data: "Statistics",
  filter_data: "Filter",
  group_and_aggregate: "Aggregation",
  compare_periods: "Period comparison",
  correlation_analysis: "Correlation analysis",
  detect_anomalies: "Anomaly detection",
  generate_chart: "Chart",
  generate_business_insights: "Business insights",
  generate_report: "Report",
};

interface Turn extends ChatMessage {
  toolCalls?: ToolCallRecord[];
  charts?: ChartData[];
  error?: boolean;
}

export function ChatTab({ profile }: { profile: DatasetProfile }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [slowLoading, setSlowLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!loading) {
      setSlowLoading(false);
      return;
    }
    const timer = setTimeout(() => setSlowLoading(true), 5000);
    return () => clearTimeout(timer);
  }, [loading]);

  const send = async (message: string) => {
    if (!message.trim() || loading) return;
    const nextTurns: Turn[] = [...turns, { role: "user", content: message }];
    setTurns(nextTurns);
    setInput("");
    setLoading(true);
    try {
      const history = turns.map(({ role, content }) => ({ role, content }));
      const res = await sendChatMessage(profile.dataset_id, message, history);
      setTurns([...nextTurns, { role: "assistant", content: res.answer, toolCalls: res.tool_calls, charts: res.charts }]);
    } catch (err) {
      const message = apiErrorMessage(err) || "Something went wrong while analyzing your data. Please try again.";
      setTurns([...nextTurns, { role: "assistant", content: message, error: true }]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }));
    }
  };

  return (
    <div className="flex h-[calc(100vh-7.5rem)] flex-col">
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto pb-4">
        {turns.length === 0 && (
          <Card className="p-6">
            <p className="text-sm font-medium text-ink-100">Ask the analyst anything about “{profile.filename}”.</p>
            <p className="mt-1 text-xs text-ink-400">
              Answers are grounded in real calculations — every number is produced by a Python tool, never guessed.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-lg border border-ink-700/60 bg-ink-900/40 px-3 py-2.5 text-left text-xs text-ink-300 transition-colors hover:border-accent/50 hover:text-ink-100"
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>
        )}

        {turns.map((turn, i) => (
          <div key={i} className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-2xl ${turn.role === "user" ? "" : "w-full"}`}>
              <div
                className={`rounded-xl2 px-4 py-2.5 ${
                  turn.role === "user"
                    ? "bg-accent text-sm text-white"
                    : turn.error
                      ? "border border-bad/30 bg-bad/10 text-sm text-bad"
                      : "border border-ink-700/60 bg-ink-850/60 text-ink-100"
                }`}
              >
                {turn.role === "assistant" && !turn.error ? <MarkdownMessage content={turn.content} /> : turn.content}
              </div>

              {!!turn.toolCalls?.length && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {turn.toolCalls.map((tc, j) => (
                    <Badge key={j} tone="accent">
                      {TOOL_LABELS[tc.tool] ?? tc.tool}
                    </Badge>
                  ))}
                </div>
              )}

              {!!turn.charts?.length && (
                <div className="mt-3 grid grid-cols-1 gap-3">
                  {turn.charts.map((chart, j) => (
                    <Card key={j} className="p-3">
                      <ChartRenderer chart={chart} />
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-xs text-ink-400">
            <Spinner className="h-3.5 w-3.5" />
            {slowLoading ? "Still analyzing your data… this can take a few extra seconds." : "Analyzing your data…"}
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex items-center gap-2 border-t border-ink-700/60 pt-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your data…"
          disabled={loading}
          className="flex-1 rounded-lg border border-ink-700 bg-ink-900/60 px-3.5 py-2.5 text-sm text-ink-50 placeholder:text-ink-400 focus:border-accent focus:outline-none disabled:opacity-60"
        />
        <Button type="submit" disabled={loading || !input.trim()}>
          {loading ? <Spinner className="h-4 w-4" /> : "Send"}
        </Button>
      </form>
    </div>
  );
}
