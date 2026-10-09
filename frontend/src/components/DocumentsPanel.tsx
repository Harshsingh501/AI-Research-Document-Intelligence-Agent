import { useRef } from "react";
import type { DocumentOut } from "../types";

interface Props {
  documents: DocumentOut[];
  uploading: boolean;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
}

function formatSize(bytes: number) {
  return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

export function DocumentsPanel({ documents, uploading, onUpload, onDelete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <section className="panel documents">
      <header className="panel-header">
        <h2>Documents</h2>
        <button
          className="btn small"
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
          title="Upload a PDF (max 25 MB)"
        >
          {uploading ? "Uploading…" : "+ PDF"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onUpload(f);
            e.target.value = ""; // allow re-uploading the same file
          }}
        />
      </header>

      {documents.length === 0 && <p className="muted">No documents yet. Upload a PDF to get started.</p>}

      <ul className="doc-list">
        {documents.map((d) => (
          <li key={d.id} className={`doc status-${d.status}`} title={d.error_message ?? d.filename}>
            <span className={`dot ${d.status}`} />
            <div className="doc-main">
              <div className="doc-name">{d.filename}</div>
              <div className="doc-meta muted">
                {d.status === "ready" && `${d.page_count} pages · `}
                {d.status === "failed" ? "failed" : d.status !== "ready" ? `${d.status}…` : formatSize(d.size_bytes)}
              </div>
            </div>
            <button className="icon" onClick={() => onDelete(d.id)} title="Delete">
              ×
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
