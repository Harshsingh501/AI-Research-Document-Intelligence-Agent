import type { MessageOut } from "../types";

interface Props {
  message: MessageOut | null;
}

export function SourcesPanel({ message }: Props) {
  if (!message || message.role !== "assistant") {
    return (
      <aside className="panel sources">
        <header className="panel-header">
          <h2>Sources</h2>
        </header>
        <p className="muted">Select an answer to see where it came from.</p>
      </aside>
    );
  }

  const meta = message.meta ?? {};
  const cited = new Set(meta.citations ?? []);
  const sources = message.sources ?? [];

  return (
    <aside className="panel sources">
      <header className="panel-header">
        <h2>Sources</h2>
      </header>

      <dl className="diag">
        {meta.standalone_question && (
          <>
            <dt>Searched for</dt>
            <dd>“{meta.standalone_question}”</dd>
          </>
        )}
        {meta.short_circuited && (
          <>
            <dt>Fast path</dt>
            <dd>No passage cleared the relevance threshold, so the model was not called.</dd>
          </>
        )}
        {meta.fallback_used && (
          <>
            <dt>Fallback</dt>
            <dd>{meta.fallback_reason}</dd>
          </>
        )}
        {meta.steps && meta.steps.length > 0 && (
          <>
            <dt>Agent steps</dt>
            <dd>
              <ol className="steps">
                {meta.steps.map((s, i) => (
                  <li key={i}>
                    <code>{s.tool}</code>
                    {Object.keys(s.arguments).length > 0 && (
                      <span className="muted"> {JSON.stringify(s.arguments)}</span>
                    )}
                  </li>
                ))}
              </ol>
            </dd>
          </>
        )}
      </dl>

      {sources.length === 0 && <p className="muted">No passages were retrieved for this answer.</p>}

      <ul className="source-list">
        {sources.map((s) => (
          <li key={s.chunk_id} className={`source ${cited.has(s.index) ? "cited" : ""}`}>
            <div className="source-head">
              <span className="source-index">{s.index}</span>
              <span className="source-file" title={s.filename}>
                {s.filename}
              </span>
              {s.page_number != null && <span className="muted">p. {s.page_number}</span>}
              <span className="score-num" title="cosine similarity">
                {s.score.toFixed(2)}
              </span>
            </div>
            <div className="score">
              <div className="score-bar" style={{ width: `${Math.max(0, Math.min(1, s.score)) * 100}%` }} />
            </div>
            <p className="excerpt">{s.excerpt}</p>
          </li>
        ))}
      </ul>
    </aside>
  );
}
