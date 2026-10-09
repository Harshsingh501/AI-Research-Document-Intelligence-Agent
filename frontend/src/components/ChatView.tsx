import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import type { ConversationDetail, DocumentOut, MessageOut, Mode } from "../types";

export interface LiveAnswer {
  stage: string;
  text: string; // answer text streamed so far
}

const STAGE_LABEL: Record<string, string> = {
  condensing: "Rewriting your question with context…",
  retrieving: "Searching documents…",
  generating: "Writing answer…",
  agent: "Agent is choosing tools…",
};

interface Props {
  conversation: ConversationDetail | null;
  documents: DocumentOut[];
  sending: boolean;
  live: LiveAnswer | null; // progress of the in-flight answer
  elapsed: number; // seconds since the current request started
  error: string | null;
  selectedMessageId: string | null;
  onSelectMessage: (id: string) => void;
  onSend: (question: string, mode: Mode, documentId: string | null) => void;
}

// Turn "...heads [2]." into text with clickable citation chips. Handles the
// grouped form "[1, 3]" the model sometimes produces.
function renderWithCitations(text: string, onCite: () => void): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    parts.push(text.slice(last, m.index));
    for (const n of m[1].split(/\s*,\s*/)) {
      parts.push(
        <button key={`${m.index}-${n}`} className="cite" onClick={onCite} title={`Source ${n}`}>
          {n}
        </button>,
      );
    }
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  return parts;
}

function ConfidenceBadge({ message }: { message: MessageOut }) {
  const c = message.meta?.confidence;
  if (!c) return null;
  return <span className={`badge conf-${c}`}>{c === "none" ? "not covered" : `${c} confidence`}</span>;
}

export function ChatView({
  conversation,
  documents,
  sending,
  live,
  elapsed,
  error,
  selectedMessageId,
  onSelectMessage,
  onSend,
}: Props) {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("ask");
  const [documentId, setDocumentId] = useState<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const readyDocs = documents.filter((d) => d.status === "ready");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages.length, sending]);

  function submit(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (q.length < 3 || sending || !conversation) return;
    onSend(q, mode, documentId || null);
    setQuestion("");
  }

  if (!conversation) {
    return (
      <main className="chat empty">
        <p className="muted">Pick a chat on the left or start a new one.</p>
      </main>
    );
  }

  return (
    <main className="chat">
      <div className="messages">
        {conversation.messages.length === 0 && (
          <p className="muted center">Ask anything about your uploaded documents.</p>
        )}
        {conversation.messages.map((m) => (
          <article
            key={m.id}
            className={`msg ${m.role} ${m.id === selectedMessageId ? "selected" : ""}`}
            onClick={() => m.role === "assistant" && onSelectMessage(m.id)}
          >
            <div className="msg-body">
              {m.role === "assistant" ? renderWithCitations(m.content, () => onSelectMessage(m.id)) : m.content}
            </div>
            {m.role === "assistant" && (
              <footer className="msg-meta">
                <ConfidenceBadge message={m} />
                {m.meta?.mode && <span className="badge mode">{m.meta.mode}</span>}
                {m.meta?.fallback_used && <span className="badge warn">fallback</span>}
                {m.sources && m.sources.length > 0 && (
                  <span className="muted">{m.sources.length} sources</span>
                )}
              </footer>
            )}
          </article>
        ))}
        {sending && (
          <article className="msg assistant pending">
            {live?.text ? (
              <div className="msg-body">
                {live.text}
                <span className="caret" />
              </div>
            ) : null}
            <div className={live?.text ? "msg-meta muted" : "msg-body muted"}>
              {STAGE_LABEL[live?.stage ?? ""] ?? "Thinking…"} {elapsed}s
              {!live?.text && <span className="hint"> (local CPU inference takes 30–90 s)</span>}
            </div>
          </article>
        )}
        {error && <div className="error">{error}</div>}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={readyDocs.length ? "Ask a question…" : "Upload a document first"}
          rows={2}
          disabled={sending}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(e);
            }
          }}
        />
        <div className="composer-row">
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} disabled={sending}>
              <option value="ask">ask (RAG, faster)</option>
              <option value="agent">agent (tools)</option>
            </select>
          </label>
          <label>
            Scope
            <select value={documentId} onChange={(e) => setDocumentId(e.target.value)} disabled={sending}>
              <option value="">all documents</option>
              {readyDocs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename}
                </option>
              ))}
            </select>
          </label>
          <button className="btn primary" type="submit" disabled={sending || question.trim().length < 3}>
            Send
          </button>
        </div>
      </form>
    </main>
  );
}
