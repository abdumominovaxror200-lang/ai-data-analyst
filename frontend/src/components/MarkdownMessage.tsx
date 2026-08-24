import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { PropsWithChildren } from "react";

export function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown-message text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }: PropsWithChildren<object>) => (
            <div className="table-scroll">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
