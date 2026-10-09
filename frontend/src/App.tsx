// App = the one place that owns server state (documents, conversations, the
// open conversation) and talks to the API. Components below it are mostly
// "props in, events out", which keeps them easy to reason about.

import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./api";
import { ChatView, type LiveAnswer } from "./components/ChatView";
import { ConversationList } from "./components/ConversationList";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { SourcesPanel } from "./components/SourcesPanel";
import type { ConversationDetail, ConversationOut, DocumentOut, Mode } from "./types";
import "./App.css";

const POLL_MS = 3000;

// The open chat lives in the URL hash (#/c/<id>) so a refresh or a shared
// link lands on the same conversation. No router library needed for one route.
function idFromHash(): string | null {
  const m = /^#\/c\/([0-9a-f-]{36})$/.exec(window.location.hash);
  return m ? m[1] : null;
}

function errorMessage(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [uploading, setUploading] = useState(false);
  const [conversations, setConversations] = useState<ConversationOut[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(idFromHash);
  const [conversation, setConversation] = useState<ConversationDetail | null>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [live, setLive] = useState<LiveAnswer | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [apiDown, setApiDown] = useState(false);

  // --- Loading ----------------------------------------------------------------

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
      setApiDown(false);
    } catch {
      setApiDown(true);
    }
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.listConversations());
    } catch (e) {
      setError(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
    void refreshConversations();
  }, [refreshDocuments, refreshConversations]);

  // Poll while any document is still being ingested. Ingestion runs as a
  // background task on the server, so the only way to learn it finished is to ask.
  const needsPoll = documents.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!needsPoll) return;
    const t = setInterval(() => void refreshDocuments(), POLL_MS);
    return () => clearInterval(t);
  }, [needsPoll, refreshDocuments]);

  // Keep the URL hash and the selection in sync, both directions
  useEffect(() => {
    const want = selectedId ? `#/c/${selectedId}` : "";
    if (window.location.hash !== want) history.replaceState(null, "", want || window.location.pathname);
  }, [selectedId]);
  useEffect(() => {
    const onHash = () => setSelectedId(idFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Load the selected conversation's messages
  useEffect(() => {
    if (!selectedId) {
      setConversation(null);
      return;
    }
    let cancelled = false;
    api
      .getConversation(selectedId)
      .then((c) => {
        if (cancelled) return;
        setConversation(c);
        const lastAssistant = [...c.messages].reverse().find((m) => m.role === "assistant");
        setSelectedMessageId(lastAssistant?.id ?? null);
      })
      .catch((e) => setError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  // Elapsed-seconds ticker while waiting for an answer
  const startedAt = useRef(0);
  useEffect(() => {
    if (!sending) return;
    startedAt.current = Date.now();
    setElapsed(0);
    const t = setInterval(() => setElapsed(Math.round((Date.now() - startedAt.current) / 1000)), 1000);
    return () => clearInterval(t);
  }, [sending]);

  // --- Actions ----------------------------------------------------------------

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);
    try {
      await api.uploadDocument(file);
      await refreshDocuments();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setUploading(false);
    }
  }

  async function handleDeleteDocument(id: string) {
    if (!confirm("Delete this document and all its chunks?")) return;
    try {
      await api.deleteDocument(id);
      await refreshDocuments();
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  async function handleNewConversation() {
    try {
      const c = await api.createConversation();
      setConversations((prev) => [c, ...prev]);
      setSelectedId(c.id);
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  async function handleDeleteConversation(id: string) {
    if (!confirm("Delete this chat?")) return;
    try {
      await api.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (selectedId === id) setSelectedId(null);
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  async function handleSend(question: string, mode: Mode, documentId: string | null) {
    if (!conversation) return;
    setSending(true);
    setLive({ stage: "retrieving", text: "" });
    setError(null);
    try {
      const res = await api.sendMessageStream(
        conversation.id,
        { question, mode, document_id: documentId },
        {
          onStatus: (s) => setLive((l) => ({ stage: s.stage, text: l?.text ?? "" })),
          onToken: (t) => setLive((l) => ({ stage: "generating", text: (l?.text ?? "") + t })),
        },
      );
      setConversation((prev) =>
        prev ? { ...prev, messages: [...prev.messages, res.user_message, res.assistant_message] } : prev,
      );
      setSelectedMessageId(res.assistant_message.id);
      await refreshConversations(); // title and ordering may have changed
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSending(false);
      setLive(null);
    }
  }

  const selectedMessage = conversation?.messages.find((m) => m.id === selectedMessageId) ?? null;

  return (
    <div className="layout">
      <nav className="sidebar">
        <h1 className="brand">DocIntel</h1>
        {apiDown && (
          <div className="error">
            Cannot reach the API at {api.API_URL}. Start it with <code>uvicorn app.main:app --reload</code>.
          </div>
        )}
        <DocumentsPanel
          documents={documents}
          uploading={uploading}
          onUpload={handleUpload}
          onDelete={handleDeleteDocument}
        />
        <ConversationList
          conversations={conversations}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onNew={handleNewConversation}
          onDelete={handleDeleteConversation}
        />
      </nav>

      <ChatView
        conversation={conversation}
        documents={documents}
        sending={sending}
        live={live}
        elapsed={elapsed}
        error={error}
        selectedMessageId={selectedMessageId}
        onSelectMessage={setSelectedMessageId}
        onSend={handleSend}
      />

      <SourcesPanel message={selectedMessage} />
    </div>
  );
}
